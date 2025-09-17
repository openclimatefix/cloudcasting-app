"""Runs metric calculations on cloudcasting for a given input day and appends to zarr store

This app expects these environmental variables to be available:
 - SATELLITE_ICECHUNK_ARCHIVE (str): Path at which ground truth satellite data can be found
 - CLOUDCASTING_PREDICTION_DIRECTORY (str): The directory where the cloudcasting forecasts are 
   saved
 - METRIC_ZARR_PATH (str): The path where the metric values will be saved

 If the SATELLITE_ICECHUNK_ARCHIVE is an s3 path, then the environment variables 
 AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY and AWS_REGION must also be set.
"""

import os
import re
import fsspec
import numpy as np
import pandas as pd
from tqdm import tqdm

import xarray as xr
import icechunk
from loguru import logger

# ---------------------------------------------------------------------------

# The forecast produces these horizon steps
FORECAST_STEPS = pd.timedelta_range(start="15min", end="180min", freq="15min")
# The forecast is run at this frequency
FORECAST_FREQ = pd.Timedelta("30min")


def open_icechunk(path: str) -> xr.Dataset:
    """Open an icechunk store to xarray Dataset
    
    Args:
        path: The path to the local or s3 icechunk store
    """

    if path.startswith("s3://"):
        bucket, _, path = path.removeprefix("s3://").partition("/")
        store = icechunk.s3_storage(
            bucket=bucket,
            prefix=path,
            access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
            secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
            region=os.environ["AWS_REGION"],
        )
    else:
        store = icechunk.local_filesystem_storage(path=path)

    repo = icechunk.Repository.open(store)
    session = repo.readonly_session("main")
    return xr.open_zarr(session.store)


def app(date: pd.Timestamp | None = None) -> None:
    """Runs metric calculations on cloudcasting for a given input day and appends to zarr store

    Args:
        date: The day for which the cloudcasting predictions will be scored.
    """

    # Unpack environmental variables
    sat_path = os.environ["SATELLITE_ICECHUNK_ARCHIVE"]
    prediction_dir = os.environ["CLOUDCASTING_PREDICTION_DIRECTORY"]
    metric_zarr_path = os.environ["METRIC_ZARR_PATH"]


    now = pd.Timestamp.now(tz="UTC").replace(tzinfo=None)

    # Default to yesterday
    if date is None:
        date = now.floor("1D") - pd.Timedelta("1D")
    
    start_dt =  date.floor("1D")
    end_dt = date.floor("1D") + pd.Timedelta("1D")

    if now <= end_dt + FORECAST_STEPS.max():
        raise Exception(
            f"We cannot score forecast with init-time {end_dt} until after the last valid-time."
        )

    # Open the satellite data store
    ds_sat = open_icechunk(path=sat_path)

    # Slice to only the timesteps we need for scoring
    ds_sat = ds_sat.sel(time=slice(start_dt, end_dt + FORECAST_STEPS.max()))

    # It is better to preload if we have the RAM space
    # - This eliminates any costs of repeatedly streaming data from the bucket
    # - It's also faster
    ds_sat = ds_sat.compute()

    # Find recent forecasts
    date_string = start_dt.strftime("%Y-%m-%d")
    remote_path = f"{prediction_dir}/{date_string}*.zarr"
    fs, path = fsspec.core.url_to_fs(remote_path)

    file_list = fs.glob(path)

    # Filter forecasts
    # - We only score forecasts we have the satellite data for
    # - If we are missing one satellite image we will skip scoring all forecasts require that
    forecasts_to_score = []

    for file in file_list:
        # Find the datetime of this forecast
        match = re.search(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}', file)
        assert match

        # Check the satellite data required to score it is present
        init_time = pd.Timestamp(match.group(0))
        if np.isin(init_time + FORECAST_STEPS, ds_sat.time).all():
            forecasts_to_score.append(file)
        else:
            logger.warn(f"Cannot score {file} due to missing satellite data")

    ds_mae_list = []

    for file in tqdm(forecasts_to_score):
        ds_forecast = xr.open_zarr(fs.get_mapper(file)).compute()

        valid_times = pd.Timestamp(ds_forecast.init_time.item()) + ds_forecast.step

        ds_forecast = (
            ds_forecast
            .assign_coords(time=valid_times)
            .swap_dims({"step":"time"})
        )

        ds_sat_sel = ds_sat.sel(
            time=ds_forecast.time,
            x_geostationary=ds_forecast.x_geostationary,
            y_geostationary=ds_forecast.y_geostationary,
            variable=ds_forecast.variable,
        )

        da_mae = np.abs(
            (ds_sat_sel.data - ds_forecast.sat_pred)
            .swap_dims({"time":"step"})
            .drop_vars("time")
        )

        # Create reductions of the full MAE matrix
        da_mae_step = da_mae.mean(dim=("x_geostationary", "y_geostationary", "variable"))
        da_mae_variable = da_mae.mean(dim=("x_geostationary", "y_geostationary", "step"))
        da_mae_spatial = da_mae.mean(dim=("step", "variable"))

        ds_mae_reductions = xr.Dataset(
            {
                "mae_step": da_mae_step,
                "mae_variable": da_mae_variable,
                "mae_spatial": da_mae_spatial,
            }
        )

        ds_mae_list.append(ds_mae_reductions)

    # Concat all the MAE scores and in-fill missing init times with NaNs
    # - Filling with NaNs makes the chunking easier
    ds_all_maes = xr.concat(ds_mae_list, dim="init_time")
    expected_init_times = pd.date_range(start_dt, end_dt, freq=FORECAST_FREQ, inclusive="left")
    ds_all_maes = ds_all_maes.reindex(init_time=expected_init_times, method=None)

    # Chunk the data ready for saving
    ds_all_maes = ds_all_maes.chunk(
        {
            "x_geostationary": -1, 
            "y_geostationary": -1, 
            "step": -1, 
            "variable": -1, 
            "init_time": 48
            }
    )

    # If it exists, open the archive of MAE values and check the coordinates against them
    fs, stripped = fsspec.core.url_to_fs(metric_zarr_path)
    if fs.exists(stripped):
        ds_maes_archive = xr.open_zarr(metric_zarr_path)

        if np.isin(ds_all_maes.init_time, ds_maes_archive.init_time).any():
            raise Exception("init-times in new MAEs already exist in MAE store")
        
        for coord in ["variable", "step", "x_geostationary", "y_geostationary"]:
            if not ds_maes_archive[coord].identical(ds_all_maes[coord]):
                raise Exception("Found differences in coord: {coord}")
                
        ds_all_maes.to_zarr(metric_zarr_path, mode="a-", append_dim="init_time")

    else:
        ds_all_maes.to_zarr(metric_zarr_path, mode="w")



if __name__ == "__main__":
    app()
