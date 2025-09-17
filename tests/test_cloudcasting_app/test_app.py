import os

import numpy as np
import xarray as xr
import zarr

from cloudcasting_app.app import app


def test_app(sat_5_data, tmp_path, init_time):

    os.chdir(tmp_path)

    # In production sat zarr is zipped
    os.environ["SATELLITE_ZARR_PATH"] = "temp_sat.zarr.zip"

    os.environ["OUTPUT_PREDICTION_DIRECTORY"] = f"{tmp_path}"

    with zarr.storage.ZipStore("temp_sat.zarr.zip", mode="x") as store:
        sat_5_data.to_zarr(store)

    app()

    # Check the two output files have been created
    latest_zarr_path = f"{tmp_path}/latest.zarr"
    t0_string_zarr_path = init_time.strftime(f"{tmp_path}/%Y-%m-%dT%H:%M.zarr")
    assert os.path.exists(latest_zarr_path)
    assert os.path.exists(t0_string_zarr_path)

    # Load the predictions and check them
    ds_y_hat = xr.open_zarr(latest_zarr_path)

    assert "sat_pred" in  ds_y_hat
    assert (
        list(ds_y_hat.sat_pred.dims)==
        ["init_time", "variable", "step", "y_geostationary", "x_geostationary"]
    )

    # Make sure all the coords are correct
    assert ds_y_hat.init_time == init_time
    assert len(ds_y_hat.step)==12
    assert (ds_y_hat.x_geostationary==sat_5_data.x_geostationary).all()
    assert (ds_y_hat.y_geostationary==sat_5_data.y_geostationary).all()

    # Make sure all of the predictions are finite
    assert np.isfinite(ds_y_hat.sat_pred).all()
