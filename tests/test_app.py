import tempfile
import zarr
import os
import xarray as xr
import pandas as pd
import numpy as np


from cloudcasting_app.app import app


def test_app(sat_5_data, tmp_path, test_t0):

    os.chdir(tmp_path)

    # In production sat zarr is zipped
    os.environ["SATELLITE_ZARR_PATH"] = "temp_sat.zarr.zip"

    os.environ["OUTPUT_PREDICTION_ZARR_PATH"] = "sat_prediction.zarr.zip"

    with zarr.storage.ZipStore("temp_sat.zarr.zip", mode="x") as store:
        sat_5_data.to_zarr(store)    

    app()

    # Load the zarr and check it
    ds_y_hat = xr.open_zarr(os.environ["OUTPUT_PREDICTION_ZARR_PATH"])
    
    assert "sat_pred" in  ds_y_hat
    assert (
        list(ds_y_hat.sat_pred.coords)==
        ["init_time", "step", "variable", "x_geostationary", "y_geostationary"]
    )
    
    # Make sure all the coords are correct
    assert ds_y_hat.init_time == test_t0
    assert len(ds_y_hat.step)==12
    assert (ds_y_hat.x_geostationary==sat_5_data.x_geostationary).all()
    assert (ds_y_hat.y_geostationary==sat_5_data.y_geostationary).all()
    
    # Make sure non of the predictions are NaN
    assert not ds_y_hat.sat_pred.isnull().any()    