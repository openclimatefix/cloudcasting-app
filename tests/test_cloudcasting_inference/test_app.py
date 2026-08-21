import os

import numpy as np
import xarray as xr

from cloudcasting_inference.app import app
from tests.utils import get_sat_shell


def test_app(sat_icechunk_path, tmp_path, init_time):

    os.environ["SATELLITE_ICECHUNK_PATH"] = sat_icechunk_path
    os.environ["S3_REGION"] = "eu-west-1"
    os.environ["PREDICTION_SAVE_DIRECTORY"] = str(tmp_path)

    app()

    # Check the two output files have been created
    latest_zarr_path = f"{tmp_path}/latest.zarr"
    t0_string_zarr_path = init_time.strftime(f"{tmp_path}/%Y-%m-%dT%H:%M.zarr")
    assert os.path.exists(latest_zarr_path)
    assert os.path.exists(t0_string_zarr_path)

    # Load the predictions and check them
    ds_y_hat = xr.open_zarr(latest_zarr_path)

    assert "sat_pred" in ds_y_hat
    assert list(ds_y_hat.sat_pred.dims) == [
        "init_time_utc",
        "channel",
        "step",
        "y_geostationary",
        "x_geostationary",
    ]

    # Make sure all the coords are correct
    sat_shell = get_sat_shell()
    assert ds_y_hat.init_time_utc == init_time
    assert len(ds_y_hat.step) == 12
    assert (ds_y_hat.channel == sat_shell.channel).all()
    assert (ds_y_hat.x_geostationary == sorted(sat_shell.x_geostationary)).all()
    assert (ds_y_hat.y_geostationary == sorted(sat_shell.y_geostationary)).all()

    # Make sure all of the predictions are finite
    assert np.isfinite(ds_y_hat.sat_pred).all()

    # The predictions are stored as float16
    assert ds_y_hat.sat_pred.dtype == np.float16
