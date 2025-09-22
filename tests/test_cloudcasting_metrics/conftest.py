import pandas as pd
import pytest
from cloudcasting_metrics.app import FORECAST_STEPS, FORECAST_FREQ
from tests.utils import get_sat_shell, make_sat_data
import icechunk
from icechunk.xarray import to_icechunk


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


@pytest.fixture()
def sat_shell():
    return get_sat_shell()
