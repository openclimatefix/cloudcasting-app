"""The main script for running the cloudcasting model in production.

The runtime configuration is loaded from environment variables - see
`cloudcasting_inference.settings.AppSettings` for the full list.
"""

import contextlib
import os
import tempfile
from collections.abc import Iterator
from importlib.metadata import version

import fsspec
import hydra
import numpy as np
import pandas as pd
import torch
import xarray as xr
import yaml
from huggingface_hub import snapshot_download
from loguru import logger
from numpy.typing import NDArray
from safetensors.torch import load_model
from sat_pred.channels import ChannelConfig, parse_channel_config
from sat_pred.dataset import SatelliteDataset

from cloudcasting_inference.data import SatelliteDownloader
from cloudcasting_inference.settings import AppSettings

__version__ = version("cloudcasting-app")

# ---------------------------------------------------------------------------

# Model will use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model revision on huggingface
REPO_ID = "openclimatefix-models/cloudcasting_uk"
REVISION = "06e7ee93366a0d1dbb52c937840319460b0b51db"


def sanitize_t0(t0: pd.Timestamp | None) -> pd.Timestamp:
    """Sanitize the input t0 to be a pandas Timestamp and round down to the nearest 30 minutes."""
    t0 = pd.Timestamp.now(tz="UTC").replace(tzinfo=None) if t0 is None else pd.to_datetime(t0)
    return t0.floor("30min")


def get_model(repo_id: str, revision: str) -> tuple[torch.nn.Module, dict]:
    """Download the model from huggingface and load it into memory.

    Returns:
        The model, and the config of the data it was trained on. The data config gives the
        history, forecast horizon, time resolution and channels the model expects, so the inputs
        built here always match what the model was trained to read.
    """
    hf_download_dir = snapshot_download(
        repo_id=repo_id,
        revision=revision,
    )

    with open(f"{hf_download_dir}/model_config.yaml", encoding="utf-8") as f:
        model = hydra.utils.instantiate(yaml.safe_load(f))

    with open(f"{hf_download_dir}/data_config.yaml", encoding="utf-8") as f:
        data_config = yaml.safe_load(f)

    model = model.to(device)

    load_model(
        model,
        filename=f"{hf_download_dir}/model.safetensors",
        strict=True,
    )

    model.eval()

    return model, data_config


def get_input_tensor(dataset: SatelliteDataset, t0: pd.Timestamp) -> torch.Tensor:
    """Get the input data for the model.

    Args:
        dataset: The dataset to get the input data from
        t0: Datetime at which forecast is made

    Returns:
        torch.Tensor: The input data for the model, with a leading batch dimension of 1
    """
    X, _ = dataset._get_datetime(t0)

    return torch.from_numpy(X).unsqueeze(0).to(device)


def prediction_to_dataset_like(
    y_hat: NDArray,
    t0: pd.Timestamp,
    forecast_mins: int,
    sample_freq_mins: int,
    da: xr.DataArray,
    model_address: str,
) -> xr.Dataset:
    """Convert the model output to a DataArray with the same coordinates as the input."""
    da_y_hat = xr.DataArray(
        y_hat,
        dims=["init_time_utc", "channel", "step", "y_geostationary", "x_geostationary"],
        coords={
            "init_time_utc": [t0],
            "channel": da.channel,
            "step": pd.timedelta_range(
                start=f"{sample_freq_mins}min",
                end=f"{forecast_mins}min",
                freq=f"{sample_freq_mins}min",
            ),
            "y_geostationary": da.y_geostationary,
            "x_geostationary": da.x_geostationary,
        },
    )

    attrs_dict = dict(da.attrs)

    ds_y_hat = da_y_hat.to_dataset(name="sat_pred")
    ds_y_hat.attrs.update(attrs_dict)
    ds_y_hat.attrs["model_address"] = model_address
    return ds_y_hat


@contextlib.contextmanager
def scratch_directory(parent_dir: str | None, t0: pd.Timestamp) -> Iterator[str]:
    """Create the directory which the downloaded inputs are saved to.

    If a parent directory is given, the per-run directory created inside it is named from the
    forecast init-time and is left in place after the run. Otherwise a temporary directory is used
    and is deleted on exit.

    Args:
        parent_dir: The directory to create the per-run directory in, if any
        t0: Datetime at which forecast is made
    """
    if parent_dir is None:
        with tempfile.TemporaryDirectory(prefix="cloudcasting-app-") as scratch_dir:
            yield scratch_dir
    else:
        os.makedirs(parent_dir, exist_ok=True)
        yield tempfile.mkdtemp(
            prefix=t0.strftime("cloudcasting-app-%Y%m%dT%H%M%SZ-"),
            dir=parent_dir,
        )


@torch.no_grad()
def app(t0: pd.Timestamp | None = None) -> None:
    """Inference function for production.

    Sets up the scratch directory used for the downloaded inputs, then runs the forecast.

    Args:
        t0: Datetime at which forecast is made. Defaults to the current time
    """
    logger.info(f"Using `cloudcasting-app` version: {__version__}")

    settings = AppSettings()

    t0 = sanitize_t0(t0)
    logger.info(f"Making forecast for init time: {t0}")

    with scratch_directory(settings.scratch_dir, t0) as scratch_dir:
        logger.info(f"Using scratch directory: {scratch_dir}")
        _run_forecast(settings, t0, scratch_dir)


def _run_forecast(settings: AppSettings, t0: pd.Timestamp, scratch_dir: str) -> None:
    """Download the inputs, run the model, and save the predictions.

    Args:
        settings: The application settings
        t0: Datetime at which forecast is made
        scratch_dir: Directory for downloaded inputs and temporary files
    """
    logger.info("Loading model")
    model, data_config = get_model(REPO_ID, REVISION)

    logger.info("Downloading satellite data")
    # ocf_data_sampler.find_contiguous_time_periods which SatelliteDataset uses internally discards
    # any time periods which aren't <= the history_mins, so we need to download a bit more than that
    # to ensure we get enough data. This can be removed once the new version of ocf_data_sampler
    # is released and SatelliteDataset is updated to use it.
    history_mins = data_config["history_mins"] + data_config["sample_freq_mins"]
    sat_path = f"{scratch_dir}/rss.zarr"
    SatelliteDownloader(
        interval_start=t0 - pd.Timedelta(minutes=history_mins),
        interval_end=t0,
        source_path=settings.satellite_icechunk_path,
        s3_region=settings.s3_region,
        destination_path=sat_path,
    ).run()

    logger.info("Preparing inputs")
    channel_config: ChannelConfig = parse_channel_config(data_config["channels"])
    dataset = SatelliteDataset(
        zarr_path=sat_path,
        time_periods=[[None, None]],
        history_mins=data_config["history_mins"],
        forecast_mins=0,
        sample_freq_mins=data_config["sample_freq_mins"],
        channels=channel_config,
        preshuffle=False,
    )
    X = get_input_tensor(dataset, t0)

    logger.info("Making predictions")
    # The normaliser works on (channel, time, y, x) samples, so denormalise the single sample
    # before putting the init-time axis back on
    y_hat = model(X).squeeze(0).cpu().numpy()
    # Saved as float16, matching the backtest store these forecasts are scored against
    y_hat = channel_config.normaliser.denormalise(y_hat)[np.newaxis].astype(np.float16)

    logger.info("Saving predictions")

    ds_y_hat = prediction_to_dataset_like(
        y_hat=y_hat,
        t0=t0,
        forecast_mins=data_config["forecast_mins"],
        sample_freq_mins=data_config["sample_freq_mins"],
        da=dataset.da,
        model_address=f"{REPO_ID}@{REVISION}",
    )

    # Save predictions to the latest path and to path with timestring
    out_dir = settings.prediction_save_directory

    latest_zarr_path = f"{out_dir}/latest.zarr"
    t0_string_zarr_path = t0.strftime(f"{out_dir}/%Y-%m-%dT%H:%M.zarr")

    fs = fsspec.open(out_dir).fs
    for path in [latest_zarr_path, t0_string_zarr_path]:
        # Remove the path if it exists already
        if fs.exists(path):
            logger.info(f"Removing path: {path}")
            fs.rm(path, recursive=True)

        ds_y_hat.to_zarr(path)
