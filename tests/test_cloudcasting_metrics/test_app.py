import os
import pandas as pd
import xarray as xr
from cloudcasting_metrics.app import app
from cloudcasting_metrics.app import FORECAST_STEPS, FORECAST_FREQ



def test_app(tmp_path, forecast_directory, sat_icechunk_path, today, init_times_tuple, sat_5_data):

    mae_path = str(tmp_path / "mae.zarr")

    os.environ["SATELLITE_ICECHUNK_ARCHIVE"] = sat_icechunk_path
    os.environ["PREDICTION_SAVE_DIRECTORY"] = forecast_directory
    os.environ["METRIC_ZARR_PATH"] = mae_path

    # Run once for "2 days ago" - this creates an MAE zarr
    app(date=today-pd.Timedelta("2D"))

    # Check the expected init-times are there
    ds_mae = xr.open_zarr(mae_path).compute()
    expected_init_times = pd.date_range(
        today-pd.Timedelta("2D"), 
        today-pd.Timedelta("1D"), 
        freq=FORECAST_FREQ, 
        inclusive="left",
    )
    assert (ds_mae.init_time.values==expected_init_times).all()

    # Check that the expected init-times are not NaN and the others are NaN
    ds_nan = ds_mae.isnull().mean(dim=("variable", "x_geostationary", "y_geostationary", "step"))
    non_nan_init_times = init_times_tuple[0]
    nan_init_times = [t for t in expected_init_times if t not in non_nan_init_times]
    for v in ds_nan.data_vars:
        assert not ds_nan[v].sel(init_time=non_nan_init_times).any()
        assert ds_nan[v].sel(init_time=nan_init_times).all()

    # Run once for "1 days ago" - this appends to the MAE zarr
    app(date=today-pd.Timedelta("1D"))

    # Check the expected init-times are there
    ds_mae = xr.open_zarr(mae_path).compute()
    expected_init_times = pd.date_range(
        today-pd.Timedelta("2D"), 
        today, 
        freq=FORECAST_FREQ, 
        inclusive="left",
    )
    assert (ds_mae.init_time.values==expected_init_times).all()

    # Check that the expected init-times are not NaN and the others are NaN
    ds_nan = ds_mae.isnull().mean(dim=("variable", "x_geostationary", "y_geostationary", "step"))
    non_nan_init_times = [t for ts in init_times_tuple for t in ts]
    nan_init_times = [t for t in expected_init_times if t not in non_nan_init_times]
    for v in ds_nan.data_vars:
        assert not ds_nan[v].sel(init_time=non_nan_init_times).any()
        assert ds_nan[v].sel(init_time=nan_init_times).all()

    # Check the other coordinates and data_vars
    assert set(ds_mae.data_vars)=={"mae_step", "mae_variable", "mae_spatial"}
    assert (ds_mae.step.values==FORECAST_STEPS).all()

    for coord in ["x_geostationary", "y_geostationary", "variable"]:
        assert (ds_mae[coord].values==sat_5_data[coord].values).all()
