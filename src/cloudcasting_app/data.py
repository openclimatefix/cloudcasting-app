import logging
import shutil
import os

import fsspec
import numpy as np
import pandas as pd
import zipfile
import torch
import xarray as xr
from ocf_data_sampler.select.geospatial import lon_lat_to_geostationary_area_coords

logger = logging.getLogger(__name__)

sat_5_path = "sat_5_min.zarr"
sat_15_path = "sat_15_min.zarr"
sat_path = "sat.zarr"


lon_min = -16
lon_max = 10
lat_min = 45
lat_max = 70


channel_order = [
    "IR_016",
    "IR_039",
    "IR_087",
    "IR_097",
    "IR_108",
    "IR_120",
    "IR_134",
    "VIS006",
    "VIS008",
    "WV_062",
    "WV_073",
]


def crop_input_area(ds):
    (x_min, x_max), (y_min, y_max) = lon_lat_to_geostationary_area_coords(
        [lon_min, lon_max],
        [lat_min, lat_max],
        ds.data,
    )

    ds = ds.isel(x_geostationary=slice(None, None, -1))  # x-axis is in decreasing order

    ds = ds.sel(
        x_geostationary=slice(x_min, None),
        y_geostationary=slice(y_min, None),
    ).isel(
        x_geostationary=slice(0,614),
        y_geostationary=slice(0,372),
    )

    ds = ds.isel(x_geostationary=slice(None, None, -1))  # flip back
    assert len(ds.x_geostationary)==614
    assert len(ds.y_geostationary)==372

    return ds


def prepare_satellite_data(t0: pd.Timestamp):

    # Download the 5 and/or 15 minutely satellite data
    download_all_sat_data()

    # Select between the 5/15 minute satellite data sources
    combine_5_and_15_sat_data()

    # Check the required expected timestamps are available
    check_required_timestamps_available(t0)

    # Load data the data for more preprocessing
    ds = xr.open_zarr(sat_path)

    # Crop the input area to expected
    ds = crop_input_area(ds)

    # Reorder channels
    ds = ds.sel(variable=channel_order)

    # Scale the satellite data from 0-1
    ds = ds / 1023

    # Resave
    ds = ds.compute()
    if os.path.exists(sat_path):
        shutil.rmtree(sat_path)
    ds.to_zarr(sat_path)


def download_all_sat_data() -> bool:
    """Download the sat data and return whether it was successful

    Returns:
        bool: Whether the download was successful
    """
    # Clean out old files
    logging.debug("Cleaning out old satellite data")
    for loc in [sat_path, sat_5_path, sat_15_path]:
        if os.path.exists(loc):
            shutil.rmtree(loc)

    # Set variable to track whether the satellite download is successful
    sat_available = False

    # download 5 minute satellite data
    sat_5_dl_path = os.environ["SATELLITE_ZARR_PATH"]
    fs, _ = fsspec.core.url_to_fs(sat_5_dl_path)
    if fs.exists(sat_5_dl_path):
        sat_available = True
        logger.info("Downloading 5-minute satellite data")
        fs.get(sat_5_dl_path, "sat_5_min.zarr.zip")
        with zipfile.ZipFile("sat_5_min.zarr.zip", "r") as zip_ref:
            zip_ref.extractall(sat_5_path)
        os.remove("sat_5_min.zarr.zip")
    else:
        logger.info("No 5-minute data available")

    # Also download 15-minute satellite if it exists
    sat_15_dl_path = sat_5_dl_path.replace(".zarr", "_15.zarr")
    if fs.exists(sat_15_dl_path):
        sat_available = True
        logger.info("Downloading 15-minute satellite data")
        fs.get(sat_15_dl_path, "sat_15_min.zarr.zip")
        with zipfile.ZipFile("sat_15_min.zarr.zip", "r") as zip_ref:
            zip_ref.extractall(sat_15_path)
        os.remove("sat_15_min.zarr.zip")
    else:
        logger.info("No 15-minute data available")

    return sat_available


def check_required_timestamps_available(t0: pd.Timestamp):
    available_timestamps = get_satellite_timestamps(sat_path)

    # Need 12 timestamps of 15 minutely data up to and including time t0
    expected_timestamps = pd.date_range(t0-pd.Timedelta("165min"), t0, freq="15min")

    timestamps_available = np.isin(expected_timestamps, available_timestamps)

    if not timestamps_available.all():
        missing_timestamps = expected_timestamps[~timestamps_available]
        raise Exception(
            "Some required timestamps missing\n"
            f"Required timestamps: {expected_timestamps}\n"
            f"Available timestamps: {timestamps_available}\n"
            f"Missing timestamps: {missing_timestamps}",
        )


def get_satellite_timestamps(sat_zarr_path: str) -> pd.DatetimeIndex:
    """Get the datetimes of the satellite data

    Args:
        sat_zarr_path: The path to the satellite zarr

    Returns:
        pd.DatetimeIndex: All available satellite timestamps
    """
    ds_sat = xr.open_zarr(sat_zarr_path)
    return pd.to_datetime(ds_sat.time.values)


def combine_5_and_15_sat_data() -> None:
    """Select and/or combine the 5 and 15-minutely satellite data and move it to the expected path"""
    # Check which satellite data exists
    exists_5_minute = os.path.exists(sat_5_path)
    exists_15_minute = os.path.exists(sat_15_path)

    if not exists_5_minute and not exists_15_minute:
        raise FileNotFoundError("Neither 5- nor 15-minutely data was found.")

    # Find the delay in the 5- and 15-minutely data
    if exists_5_minute:
        datetimes_5min = get_satellite_timestamps(sat_5_path)
        logger.info(
            f"Latest 5-minute timestamp is {datetimes_5min.max()}. "
            f"All the datetimes are: \n{datetimes_5min}",
        )
    else:
        logger.info("No 5-minute data was found.")

    if exists_15_minute:
        datetimes_15min = get_satellite_timestamps(sat_15_path)
        logger.info(
            f"Latest 5-minute timestamp is {datetimes_15min.max()}. "
            f"All the datetimes are: \n{datetimes_15min}",
        )
    else:
        logger.info("No 15-minute data was found.")

    # If both 5- and 15-minute data exists, use the most recent
    if exists_5_minute and exists_15_minute:
        use_5_minute = datetimes_5min.max() > datetimes_15min.max()
    else:
        # If only one exists, use that
        use_5_minute = exists_5_minute

    # Move the selected data to the expected path
    if use_5_minute:
        logger.info("Using 5-minutely data.")
        os.system(f"mv {sat_5_path} {sat_path}")
    else:
        logger.info("Using 15-minutely data.")
        os.system(f"mv {sat_15_path} {sat_path}")


def get_input_data(ds: xr.Dataset, t0: pd.Timestamp):

    # Slice the data
    required_timestamps = pd.date_range(t0-pd.Timedelta("165min"), t0, freq="15min")
    ds_sel = ds.reindex(time=required_timestamps)

    # Load the data
    ds_sel = ds_sel.compute(scheduler="single-threaded")

    # Convert to arrays
    X = ds_sel.data.values.astype(np.float32)

    # Convert NaNs to -1
    X = np.nan_to_num(X, nan=-1)

    return torch.Tensor(X)
