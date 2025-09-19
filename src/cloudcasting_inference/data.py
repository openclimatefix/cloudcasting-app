import logging
import shutil
import os

import fsspec
import numpy as np
import pandas as pd
import zarr
import torch
import xarray as xr
from ocf_data_sampler.select.geospatial import lon_lat_to_geostationary_area_coords


xr.set_options(keep_attrs=True)

logger = logging.getLogger(__name__)

sat_5_path = "sat_5_min.zarr.zip"
sat_15_path = "sat_15_min.zarr.zip"
sat_path = "sat.zarr"

lon_min = -16
lon_max = 10
lat_min = 45
lat_max = 70

x_size = 614
y_size = 372

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

def get_satellite_timestamps(sat_zarr_path: str) -> pd.DatetimeIndex:
    """Get the datetimes of the satellite data

    Args:
        sat_zarr_path: The path to the satellite zarr

    Returns:
        pd.DatetimeIndex: All available satellite timestamps
    """
    with zarr.storage.ZipStore(sat_zarr_path) as store:
        ds = xr.open_zarr(store)
    return pd.to_datetime(ds.time.values)


def crop_input_area(ds: xr.Dataset) -> xr.Dataset:

    x_min, y_min = lon_lat_to_geostationary_area_coords(lon_min, lat_min, ds.data.attrs["area"])

    # x-axis is expected to be in decreasing order
    # y-axis is expected to be in ascending order
    assert ds.x_geostationary.values[0] > ds.x_geostationary.values[1]
    assert ds.y_geostationary.values[0] < ds.y_geostationary.values[1]
    
    ds = ds.isel(x_geostationary=slice(None, None, -1))

    ds = (
        ds
        .sel(x_geostationary=slice(x_min, None), y_geostationary=slice(y_min, None))
        .isel(x_geostationary=slice(0, x_size), y_geostationary=slice(0, y_size))
    )

    assert len(ds.x_geostationary)==x_size
    assert len(ds.y_geostationary)==y_size

    return ds.isel(x_geostationary=slice(None, None, -1))  # flip back


def get_input_data(ds: xr.Dataset, t0: pd.Timestamp) -> torch.Tensor:
    """Get the input data required to run the model for init-time t0"""

    # Slice the data
    required_timestamps = pd.date_range(t0-pd.Timedelta("165min"), t0, freq="15min")
    ds = ds.reindex(time=required_timestamps)

    # Convert to arrays
    X = ds.data.values.astype(np.float32)

    # Convert NaNs to -1
    X = np.nan_to_num(X, nan=-1)

    return torch.Tensor(X)


class SatelliteDownloader:

    def __init__(self):
        self.use_5_minute = None

    def prepare_satellite_data(self, t0: pd.Timestamp) -> None:

        # Download the 5 and/or 15 minutely satellite data
        self.download_all_sat_data()

        # Select between the 5/15 minute satellite data sources
        ds = self.combine_5_and_15_sat_data()

        # Check the required expected timestamps are available
        self.check_required_timestamps_available(ds, t0)

        # Crop the input area to expected
        ds = crop_input_area(ds)

        # Reorder channels
        ds = ds.sel(variable=channel_order)

        # Reshape to (channel, time, height, width)
        ds = ds.transpose("variable", "time", "y_geostationary", "x_geostationary")

        # Resave
        ds.to_zarr(sat_path)

    def download_all_sat_data(self) -> None:
        """Download the sat data"""
        # Clean out old files
        logger.info("Cleaning out old satellite data")
        for loc in [sat_path, sat_5_path, sat_15_path]:
            if os.path.exists(loc):
                shutil.rmtree(loc)

        sat_5_dl_path = os.getenv("SATELLITE_ZARR_PATH")
        sat_15_dl_path = os.getenv("SATELLITE_15_ZARR_PATH")

        for remote_path, local_path, label in [
            (sat_5_dl_path, sat_5_path, "5-min"), 
            (sat_15_dl_path, sat_15_path, "15-min"),
        ]:  
            if remote_path is not None:
                fs, _ = fsspec.core.url_to_fs(remote_path)
                if fs.exists(remote_path):
                    logger.info(f"Downloading {label} satellite data")
                    fs.get(remote_path, local_path)
                else:
                    logger.info(f"No {label} data available to download")

    def combine_5_and_15_sat_data(self) -> xr.Dataset:
        """Select and/or combine the 5 and 15-minutely satellite data"""
        # Check which satellite data exists
        exists_5_minute = os.path.exists(sat_5_path)
        exists_15_minute = os.path.exists(sat_15_path)

        if not (exists_5_minute or exists_15_minute):
            raise FileNotFoundError("Neither 5- nor 15-minutely data was found.")

        # Find the delay in the 5- and 15-minutely data
        if exists_5_minute:
            datetimes_5min = get_satellite_timestamps(sat_5_path)
            logger.info(
                f"Latest 5-minute timestamp is {datetimes_5min.max()}. "
                f"All the datetimes are: \n{datetimes_5min}",
            )

        if exists_15_minute:
            datetimes_15min = get_satellite_timestamps(sat_15_path)
            logger.info(
                f"Latest 5-minute timestamp is {datetimes_15min.max()}. "
                f"All the datetimes are: \n{datetimes_15min}",
            )

        # If both 5- and 15-minute data exists, use the most recent
        if exists_5_minute and exists_15_minute:
            use_5_minute = datetimes_5min.max() > datetimes_15min.max()
        else:
            # If only one exists, use that
            use_5_minute = exists_5_minute

        # Store the choice in satellite data
        self.use_5_minute = use_5_minute

        # Move the selected data to the expected path
        if use_5_minute:
            logger.info("Using 5-minutely data.")
            selected_path = sat_5_path
        else:
            logger.info("Using 15-minutely data.")
            selected_path = sat_15_path

        # Open and return the satellite data
        with zarr.storage.ZipStore(selected_path) as store:
            ds = xr.open_zarr(store).compute()

        return ds

    @staticmethod
    def check_required_timestamps_available(ds: xr.Dataset, t0: pd.Timestamp) -> None:
        available_timestamps =  pd.to_datetime(ds.time.values)

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
