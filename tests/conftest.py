import os
import pytest

import numpy as np
import pandas as pd
import xarray as xr
import fsspec


xr.set_options(keep_attrs=True)

@pytest.fixture()
def test_t0():
    return pd.Timestamp.now(tz=None).floor("30min")


def make_sat_data(test_t0, freq_mins):
    
    # Load dataset which only contains coordinates, but no data
    shell_path = f"{os.path.dirname(os.path.abspath(__file__))}/test_data/non_hrv_shell.zarr.zip"
    
    ds = xr.open_zarr(fsspec.get_mapper(f"zip::{shell_path}"))

    # Remove original time dim 
    ds = ds.drop_vars("time")

    # Add new times so they lead up to present 
    times = pd.date_range(
        test_t0 - pd.Timedelta("3h"),
        test_t0,
        freq=f"{freq_mins}min",
    )
    ds = ds.expand_dims(time=times)

    # Add data to dataset
    ds["data"] = xr.DataArray(
        np.zeros([len(ds[c]) for c in ds.xindexes]),
        coords=[ds[c] for c in ds.xindexes],
    )
    
    # Add stored attributes to DataArray
    ds.data.attrs = ds.attrs["_data_attrs"]
    del ds.attrs["_data_attrs"]

    return ds


@pytest.fixture()
def sat_5_data(test_t0):
    return make_sat_data(test_t0, freq_mins=5)