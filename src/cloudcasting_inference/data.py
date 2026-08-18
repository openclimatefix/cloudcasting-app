"""Functions to download and process satellite data."""

import logging
import shutil

import icechunk
import numpy as np
import pandas as pd
import xarray as xr


logger = logging.getLogger(__name__)

# The maximum gap size which will be filled via interpolation
MAXIMUM_INTERPOLATION_GAP = pd.Timedelta("15min")
# The assumed frequency of satellite image inputs required by all models
IMAGE_FREQUENCY = pd.Timedelta("5min")


def open_satellite_data(s3_icechunk_path: str, region: str) -> xr.Dataset | None:
    """Open the satellite data from the given s3 icechunk path.

    Args:
        s3_icechunk_path: The s3 path to the icechunk containing the satellite
        region: The s3 region where the icechunk is stored
    """
    bucket, _, path = s3_icechunk_path.removeprefix("s3://").partition("/")

    store = icechunk.s3_storage(
        bucket=bucket,
        prefix=path,
        from_env=True,
        region=region,
    )

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
    timestamp_available = np.isin(dense_times, ds.time)

    # If all the requested times are present we avoid running interpolation
    if timestamp_available.all():
        logger.info("No gaps in the required satellite sequence - no interpolation run")
        return ds

    # If less than 2 of the buffer requested times are present we cannot infill
    elif timestamp_available.sum() < 2:
        logger.warning("Cannot run interpolate infilling with less than 2 time steps available")
        return ds

    else:
        logger.info("Some requested times are missing - running interpolation")

        # Run the interpolation to all 5-minute timestamps between the first and last
        ds_interp = ds.interp(time=dense_times, method="linear", assume_sorted=True)

        # Find the timestamps which are within max gap size
        max_gap_steps = int(max_gap / IMAGE_FREQUENCY) - 1
        valid_fill_times = fill_1d_bool_gaps(timestamp_available, max_gap_steps)

        # Mask the timestamps outside the max gap size
        valid_fill_times_xr = xr.zeros_like(ds_interp.time, dtype=bool)
        valid_fill_times_xr.values[:] = valid_fill_times
        ds_sat_filtered = ds_interp.where(valid_fill_times_xr, drop=True)

        time_was_filled = np.logical_and(valid_fill_times_xr, ~timestamp_available)

        if time_was_filled.any():
            infilled_times = time_was_filled.where(time_was_filled, drop=True)
            logger.info(
                f"The following times were filled by interpolation:\n{infilled_times.time.values}",
            )

        if not valid_fill_times_xr.all():
            not_infilled_times = valid_fill_times_xr.where(~valid_fill_times_xr, drop=True)
            logger.info(
                "After interpolation the following times are still missing:\n"
                f"{not_infilled_times.time.values}",
            )

        return ds_sat_filtered


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
                f"\n{missing_time_steps}")

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
            s3_icechunk_path=self.source_path,
            region=self.s3_region,
        )

        logger.info(
            f"Satellite data contains times:"
            f"\n...\n{ds.time.values[-24:]}",
        )

        # We slice the data to the required time window for the model, plus a buffer to allow for
        # interpolation of missing timestamps
        start_dt = self.interval_start - MAXIMUM_INTERPOLATION_GAP
        end_dt = self.interval_end + MAXIMUM_INTERPOLATION_GAP

        ds = (
            ds
            .sortby("time")
            .drop_duplicates("time", keep="last")
            .sel(time=slice(start_dt, end_dt))
            # Filter out unused variables
            [["data"]]
            # Load into memory for processing
            .load()
        )

        if len(ds.time) == 0:
            logger.warning("No satellite data available in recent window.")
            return

        ds = self.process(ds)

        if satellite_inputs_available(
            interval_start=self.interval_start,
            interval_end=self.interval_end,
            sat_datetimes=ds.time.to_index(),
        ):
            self.resave(ds)
        else:
            raise ValueError("Satellite data is not available for the required time window.")

            