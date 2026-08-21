"""Functions to download and process satellite data."""

import shutil

import icechunk
import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger

# The maximum gap size which will be filled via interpolation
MAXIMUM_INTERPOLATION_GAP = pd.Timedelta("15min")
# The assumed frequency of satellite image inputs required by all models
IMAGE_FREQUENCY = pd.Timedelta("5min")


def open_satellite_data(icechunk_path: str, region: str) -> xr.Dataset | None:
    """Open the satellite data from the given icechunk path.

    Args:
        icechunk_path: The local or s3 path to the icechunk store holding the satellite data
        region: The s3 region the store is in. Unused for a local store
    """
    if icechunk_path.startswith("s3://"):
        bucket, _, prefix = icechunk_path.removeprefix("s3://").partition("/")
        store = icechunk.s3_storage(
            bucket=bucket,
            prefix=prefix,
            from_env=True,
            region=region,
        )
    else:
        store = icechunk.local_filesystem_storage(path=icechunk_path)

    try:
        repo = icechunk.Repository.open(store)
        session = repo.readonly_session("main")
        ds = xr.open_zarr(session.store)
    except icechunk.IcechunkError as e:
        logger.error(f"Error opening icechunk repository: {e}")
        ds = None

    return ds


def fill_1d_bool_gaps(x: np.ndarray, max_gap: int) -> np.ndarray:
    """Fill consecutive False elements if their number is less than the gap_size.

    Args:
        x: A 1-dimensional boolean array
        max_gap: integer of the maximum gap size which will be filled with True

    Returns:
        A 1-dimensional boolean array

    Examples:
        >>> x = np.array([0, 1, 0, 0, 1, 0, 1, 0])
        >>> fill_1d_bool_gaps(x, max_gap=2).astype(int)
        array([0, 1, 1, 1, 1, 1, 1, 0])

        >>> x = np.array([1, 0, 0, 0, 1, 0, 1, 0])
        >>> fill_1d_bool_gaps(x, max_gap=2).astype(int)
        array([1, 0, 0, 0, 1, 1, 1, 0])
    """
    should_fill = np.zeros(len(x), dtype=bool)

    i_start = None

    last_b = False
    for i, b in enumerate(x):
        if last_b and not b:
            i_start = i
        elif b and not last_b and i_start is not None:
            if i - i_start <= max_gap:
                should_fill[i_start:i] = True
            i_start = None
        last_b = b

    return np.logical_or(should_fill, x)


def interpolate_missing_satellite_timestamps(ds: xr.Dataset, max_gap: pd.Timedelta) -> xr.Dataset:
    """Linearly interpolate missing satellite timestamps.

    The max gap is inclusive of timestamps either side. E.g. if max gap is 15 minutes and the
    satellite includes timestamps 12:00 and 12:15, then 12:05 and 12:10 will be filled. If the max
    gap was 10 minutes, then none of the timestamps would be filled. A max gap of 5 minutes will do
    nothing since the normal spacing is already 5 minutes.

    Args:
        ds: The satellite data
        max_gap: The maximum gap size which will be filled via interpolation.
    """
    # If any of these times are missing, we will try to interpolate them
    dense_times = pd.date_range(ds.time.min().item(), ds.time.max().item(), freq=IMAGE_FREQUENCY)

    # Create mask array of which timestamps are available
    time_is_available = np.isin(dense_times, ds.time)

    # If all the requested times are present we avoid running interpolation
    if time_is_available.all():
        logger.info("No gaps in the required satellite sequence - no interpolation run")
        return ds

    # If less than 2 of the buffer requested times are present we cannot infill
    elif time_is_available.sum() < 2:
        logger.warning("Cannot run interpolate infilling with less than 2 time steps available")
        return ds

    else:
        logger.info("Some requested times are missing - running interpolation")

        # Find the timestamps which are within max gap size
        max_gap_steps = int(max_gap / IMAGE_FREQUENCY) - 1
        interpolatable_time_mask = fill_1d_bool_gaps(time_is_available, max_gap_steps)

        time_can_be_filled = np.logical_and(interpolatable_time_mask, ~time_is_available)

        if not interpolatable_time_mask.all():
            logger.info(
                "After interpolation the following times will still be missing:\n"
                f"{dense_times[~interpolatable_time_mask].values}",
            )

        if not time_can_be_filled.any():
            return ds

        logger.info(
            f"The following times were filled by interpolation:\n"
            f"{dense_times[time_can_be_filled].values}",
        )

        # Only the missing timestamps are interpolated. Interpolating the whole dense sequence
        # promotes it to float64 and allocates several copies of the full array inside scipy,
        # which is enough to exhaust the memory of the production container
        ds_fill = ds.interp(
            time=dense_times[time_can_be_filled],
            method="linear",
            assume_sorted=True,
        )

        # interp always returns float64, so cast back before concatenating to avoid promoting the
        # data which did not need filling
        for name, var in ds.data_vars.items():
            ds_fill[name] = ds_fill[name].astype(var.dtype)

        return xr.concat([ds, ds_fill], dim="time").sortby("time")


def satellite_inputs_available(
    interval_start: pd.Timestamp,
    interval_end: pd.Timestamp,
    sat_datetimes: pd.DatetimeIndex | None,
) -> bool:
    """Checks whether the model can be run given the current satellite delay.

    Args:
        interval_start: The init-time of the forecast
        interval_end: The end-time of the forecast
        sat_datetimes: The available satellite timestamps

    Returns:
        bool: Whether the satellite data satisfies that specified in the config
    """
    # In case no satellite is available
    if sat_datetimes is None:
        return False

    else:
        expected_datetimes = pd.date_range(interval_start, interval_end, freq=IMAGE_FREQUENCY)

        # Check if any of the expected datetimes are missing
        missing_time_steps = np.setdiff1d(expected_datetimes, sat_datetimes, assume_unique=True)

        available = len(missing_time_steps) == 0

        if len(missing_time_steps) > 0:
            logger.info(
                f"Some satellite timesteps in interval {interval_start}-{interval_end} missing:"
                f"\n{missing_time_steps}"
            )

        return available


class SatelliteDownloader:
    """Class to download and process satellite data."""

    def __init__(
        self,
        interval_start: pd.Timestamp,
        interval_end: pd.Timestamp,
        source_path: str,
        s3_region: str,
        destination_path: str,
    ) -> None:
        """Class to download and process satellite data."""
        self.interval_start = interval_start
        self.interval_end = interval_end
        self.source_path = source_path
        self.s3_region = s3_region
        self.destination_path = destination_path

    def process(self, ds: xr.Dataset) -> xr.Dataset:
        """Apply all processing steps to the satellite data in order to match the training data.

        Args:
            ds: The satellite data

        Returns:
            xr.Dataset: The processed satellite data
        """
        # Interpolate missing satellite timestamps
        ds = interpolate_missing_satellite_timestamps(ds, max_gap=MAXIMUM_INTERPOLATION_GAP)

        return ds

    def resave(self, ds: xr.Dataset) -> None:
        """Resave the satellite data to the destination path."""
        # Overwrite the old data
        shutil.rmtree(self.destination_path, ignore_errors=True)

        save_chunk_dict = {
            "x_geostationary": 100,
            "y_geostationary": 100,
            "time": 6,
            "channel": -1,
        }

        # Clear old encoding
        for v in list(ds.variables.keys()):
            ds[v].encoding.clear()

        ds.chunk(save_chunk_dict).to_zarr(self.destination_path)

    def run(self) -> None:
        """Download, process, and save the satellite data."""
        ds = open_satellite_data(
            icechunk_path=self.source_path,
            region=self.s3_region,
        )

        if ds is None:
            raise ValueError(f"Could not open the satellite data at {self.source_path}")

        logger.info(
            f"Satellite data contains times:\n...\n{ds.time.values[-24:]}",
        )

        # We slice the data to the required time window for the model, plus a buffer to allow for
        # interpolation of missing timestamps
        start_dt = self.interval_start - MAXIMUM_INTERPOLATION_GAP
        end_dt = self.interval_end + MAXIMUM_INTERPOLATION_GAP

        ds = (
            ds.sortby("time")
            .drop_duplicates("time", keep="last")
            .sel(time=slice(start_dt, end_dt))[
                # Filter out unused variables
                ["data"]
            ]
            # Load into memory for processing
            .load()
        )

        if len(ds.time) == 0:
            raise ValueError(
                f"No satellite data available between {start_dt} and {end_dt}.",
            )

        ds = self.process(ds)

        if satellite_inputs_available(
            interval_start=self.interval_start,
            interval_end=self.interval_end,
            sat_datetimes=ds.time.to_index(),
        ):
            self.resave(ds)
        else:
            raise ValueError("Satellite data is not available for the required time window.")
