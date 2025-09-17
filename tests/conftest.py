from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import zarr
from cloudcasting_metrics.app import FORECAST_STEPS, FORECAST_FREQ
import icechunk
from icechunk.xarray import to_icechunk

xr.set_options(keep_attrs=True)

@pytest.fixture()
def init_time():
    return pd.Timestamp.now(tz="UTC").replace(tzinfo=None).floor("30min")


def make_sat_data(times: pd.DatetimeIndex):

    # Load dataset which only contains coordinates, but no data
    shell_path = f"{Path(__file__).parent}/test_data/non_hrv_shell.zarr.zip"
    with zarr.storage.ZipStore(shell_path, mode="r") as store:
        ds = xr.open_zarr(store)

    # Remove original time dim
    ds = ds.drop_vars("time")

    # Add new times so they lead up to present
    ds = ds.expand_dims(time=times)

    # Add data to dataset
    ds["data"] = xr.DataArray(
        np.zeros([len(ds[c]) for c in ds.xindexes]),
        coords=[ds[c] for c in ds.xindexes],
    )

    # Add stored attributes to DataArray
    ds.data.attrs = ds.attrs["_data_attrs"]
    del ds.attrs["_data_attrs"]


    # # This is important to avoid saving errors
    for v in list(ds.variables.keys()):
        ds[v].encoding.clear()

    return ds


@pytest.fixture()
def sat_5_data(init_time):
    times = pd.date_range(
        init_time - pd.Timedelta("3h"),
        init_time,
        freq=f"5min",
    )
    return make_sat_data(times)


@pytest.fixture()
def today():
    return pd.Timestamp.now(tz="UTC").replace(tzinfo=None).floor("1D")


@pytest.fixture()
def init_times_tuple(today) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    init_times_day1 = pd.date_range(today - pd.Timedelta("2D"), freq=FORECAST_FREQ, periods=3)
    init_times_day2 = pd.date_range(today - pd.Timedelta("1D"), freq=FORECAST_FREQ, periods=3)
    return init_times_day1, init_times_day2


@pytest.fixture()
def forecast_directory(tmp_path_factory, init_times_tuple) -> str:
    pred_dir = str(tmp_path_factory.mktemp("pred_dir"))

    all_init_times = [t for ts in init_times_tuple for t in ts]

    for init_time in all_init_times:
        ds_pred = make_sat_data(times=init_time+FORECAST_STEPS)
        ds_pred = ds_pred.assign_coords(step=("time", FORECAST_STEPS))
        ds_pred = ds_pred.swap_dims({"time":"step"}).drop_vars("time")
        ds_pred = ds_pred.expand_dims({"init_time": [init_time]})
        ds_pred = ds_pred.rename({"data": "sat_pred"})

        zarr_path = init_time.strftime(f"{pred_dir}/%Y-%m-%dT%H:%M.zarr")
        ds_pred.to_zarr(zarr_path)

    yield pred_dir

@pytest.fixture()
def sat_icechunk_path(tmp_path, init_times_tuple) -> str:
    sat_icechunk_path = str(tmp_path / "sat.icechunk")

    all_init_times = [t for ts in init_times_tuple for t in ts]

    sat_times = set([t for init_time in all_init_times for t in (init_time+FORECAST_STEPS)])
    sat_times = pd.to_datetime(sorted(sat_times))
    
    ds_sat = make_sat_data(sat_times)

    store = icechunk.local_filesystem_storage(sat_icechunk_path)
    repo = icechunk.Repository.create(store)
    session = repo.writable_session(branch="main")

    to_icechunk(ds_sat, session)
    session.commit("Commit test data")

    yield sat_icechunk_path
