from pathlib import Path

import icechunk
import numpy as np
import pandas as pd
import xarray as xr
import zarr
from icechunk.xarray import to_icechunk

xr.set_options(keep_attrs=True)


def get_sat_shell():
    shell_path = f"{Path(__file__).parent}/test_data/non_hrv_shell.zarr.zip"
    with zarr.storage.ZipStore(shell_path, mode="r") as store:
        ds = xr.open_zarr(store)

    # Remove original time dim
    ds = ds.drop_vars("time")

    # The shell was saved with the channels under `variable`, but the production satellite store
    # names that dimension `channel` - see `sat_pred.dataset.open_sat_data`
    return ds.rename({"variable": "channel"})


def make_sat_data(times: pd.DatetimeIndex) -> xr.Dataset:

    # Load dataset which only contains coordinates, but no data
    ds = get_sat_shell()

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


def write_icechunk(ds: xr.Dataset, path: str) -> str:
    """Write a dataset to a new local icechunk store, as the apps expect to read it."""
    store = icechunk.local_filesystem_storage(path)
    repo = icechunk.Repository.create(store)
    session = repo.writable_session(branch="main")

    to_icechunk(ds, session)
    session.commit("Commit test data")

    return path
