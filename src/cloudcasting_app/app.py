"""The main script for running the cloudcasting model in production

This app expects these environmental variables to be available:
    SATELLITE_ZARR_PATH (str): The path of the input satellite data
    PREDICTION_SAVE_DIRECTORY (str): The path of the directory to save the predictions to
"""

import os
from importlib.metadata import PackageNotFoundError, version

import fsspec
import hydra
import pandas as pd
import torch
import xarray as xr
import yaml
from huggingface_hub import snapshot_download
from safetensors.torch import load_model
from loguru import logger

from cloudcasting_app.data import SatelliteDownloader, sat_path, get_input_data

# Get package version
try:
    __version__ = version("cloudcasting-app")
except PackageNotFoundError:
    __version__ = "v?"


# ---------------------------------------------------------------------------

# Model will use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model revision on huggingface
REPO_ID = "openclimatefix/cloudcasting_uk"
REVISION = "47643e89000e64e0150f7359ccc0cb6524948712"


def app(t0=None):
    """Inference function for production

    Args:
        t0 (datetime): Datetime at which forecast is made
    """
    logger.info(f"Using `cloudcasting_app` version: {__version__}", version=__version__)

    # ---------------------------------------------------------------------------
    # 0. If inference datetime is None, round down to last 30 minutes
    if t0 is None:
        t0 = pd.Timestamp.now(tz="UTC").replace(tzinfo=None).floor("30min")
    else:
        t0 = pd.to_datetime(t0).floor("30min")

    logger.info(f"Making forecast for init time: {t0}")

    # ---------------------------------------------------------------------------
    # 1. Prepare the input data
    logger.info("Downloading satellite data")
    satellite_downloader = SatelliteDownloader()
    satellite_downloader.prepare_satellite_data(t0)

    # ---------------------------------------------------------------------------
    # 2. Load model
    logger.info("Loading model")

    hf_download_dir = snapshot_download(
        repo_id=REPO_ID,
        revision=REVISION,
    )

    with open(f"{hf_download_dir}/model_config.yaml", encoding="utf-8") as f:
        model = hydra.utils.instantiate(yaml.safe_load(f))

    model = model.to(device)

    load_model(
        model,
        filename=f"{hf_download_dir}/model.safetensors",
        strict=True,
    )

    model.eval()

    # ---------------------------------------------------------------------------
    # 3. Get inference inputs
    logger.info("Preparing inputs")

    # Get inputs
    ds = xr.open_zarr(sat_path).compute()

    X = get_input_data(ds, t0)

    # Convert to tensor, expand into batch dimension, and move to device
    X = X[None, ...].to(device)

    # ---------------------------------------------------------------------------
    # 4. Make predictions
    logger.info("Making predictions")

    with torch.no_grad():
        y_hat = model(X).cpu().numpy()

    # ---------------------------------------------------------------------------
    # 5. Save predictions
    logger.info("Saving predictions")
    da_y_hat = xr.DataArray(
        y_hat,
        dims=["init_time", "variable", "step", "y_geostationary", "x_geostationary"],
        coords={
            "init_time": [t0],
            "variable": ds.variable,
            "step": pd.timedelta_range(start="15min", end="180min", freq="15min"),
            "y_geostationary": ds.y_geostationary,
            "x_geostationary": ds.x_geostationary,
        },
    )

    ds_y_hat = da_y_hat.to_dataset(name="sat_pred")
    ds_y_hat.sat_pred.attrs.update(ds.data.attrs)

    # Save predictions to the latest path and to path with timestring
    out_dir = os.environ["PREDICTION_SAVE_DIRECTORY"]

    if satellite_downloader.use_5_minute:
        latest_zarr_path = f"{out_dir}/latest.zarr"
        t0_string_zarr_path = t0.strftime(f"{out_dir}/%Y-%m-%dT%H:%M.zarr")
    else:
        latest_zarr_path = f"{out_dir}/latest_0-deg.zarr"
        t0_string_zarr_path = t0.strftime(f"{out_dir}/%Y-%m-%dT%H:%M_0-deg.zarr")

    fs = fsspec.open(out_dir).fs
    for path in [latest_zarr_path, t0_string_zarr_path]:

        # Remove the path if it exists already
        if fs.exists(path):
            logger.info(f"Removing path: {path}")
            fs.rm(path, recursive=True)

        ds_y_hat.to_zarr(path)
