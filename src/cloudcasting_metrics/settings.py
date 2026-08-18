"""Runtime configuration loaded from environment."""

from pydantic_settings import BaseSettings


class AppSettings(BaseSettings):
    """Runtime configuration loaded from environment.

    Input data paths

        - SATELLITE_ICECHUNK_PATH: Path at which ground truth satellite data can be found.
        - S3_REGION: The AWS region for the satellite data S3 bucket. Unused if
          SATELLITE_ICECHUNK_PATH is a local path.
        - PREDICTION_SAVE_DIRECTORY: The directory where the cloudcasting forecasts are saved.

    Output data paths

        - METRIC_ZARR_PATH: The path where the metric values will be saved.

    If SATELLITE_ICECHUNK_PATH is an s3 path, then the environment variables
    AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY must also be set. These are read directly from
    the environment by icechunk rather than through these settings.
    """

    # Input data paths
    satellite_icechunk_path: str
    s3_region: str
    prediction_save_directory: str

    # Output data paths
    metric_zarr_path: str
