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

from cloudcasting_inference.data import SatelliteDownloader
from sat_pred.dataset import SatelliteDataset

# Get package version
try:
    __version__ = version("cloudcasting-inference")
except PackageNotFoundError:
    __version__ = "v?"


# ---------------------------------------------------------------------------

# Model will use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model revision on huggingface
REPO_ID = "openclimatefix-models/cloudcasting_uk"
REVISION = "06e7ee93366a0d1dbb52c937840319460b0b51db"


def sanitize_t0(t0: pd.Timestamp | None) -> pd.Timestamp:
    """Sanitize the input t0 to be a pandas Timestamp and round down to the nearest 30 minutes."""
    if t0 is None:
        t0 = pd.Timestamp.now(tz="UTC").replace(tzinfo=None)
    else:
        t0 = pd.to_datetime(t0)
    return t0.floor("30min")


def get_model(repo_id: str, revision: str) -> torch.nn.Module:
    """Download the model from huggingface and load it into memory"""

    hf_download_dir = snapshot_download(
        repo_id=repo_id,
        revision=revision,
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

    return model


def get_input_data(t0: pd.Timestamp, history_mins: int) -> torch.Tensor:
    """Get the input data for the model

    Returns:
        torch.Tensor: The input data for the model
    """
    dataset = SatelliteDataset(
            zarr_path=os.environ["PREDICTION_SAVE_DIRECTORY"],
            time_periods=[[None, None]],
            history_mins=history_mins,
            forecast_mins=0,
            sample_freq_mins=15,
            channels=None, # ChannelConfigInput
            preshuffle=False,
        )

    X, _ = dataset._get_datetime(t0)

    return X.unsqueeze(0).to(device)

@torch.no_grad()
def app(t0=None):
    """Inference function for production

    Args:
        t0 (datetime): Datetime at which forecast is made
    """
    logger.info(f"Using `cloudcasting-app` version: {__version__}", version=__version__)

    t0 = sanitize_t0(t0)
    logger.info(f"Making forecast for init time: {t0}")


    logger.info("Downloading satellite data")
    SatelliteDownloader(
        interval_start=t0-pd.Timedelta(minutes=165),
        interval_end=t0,
        source_path=os.environ["SATELLITE_ZARR_PATH"],
        s3_region=os.environ["S3_REGION"],
        destination_path=os.environ["PREDICTION_SAVE_DIRECTORY"],
    ).run()

    logger.info("Loading model")
    model = get_model(REPO_ID, REVISION)

    logger.info("Preparing inputs")
    X = get_input_data(t0, history_mins=model.history_mins)


    logger.info("Making predictions")
    y_hat = model(X).squeeze(0).cpu().numpy()

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
