"""Runtime configuration loaded from environment."""

from pydantic_settings import BaseSettings


class AppSettings(BaseSettings):
    """Runtime configuration loaded from environment.

    Input data paths

        - SATELLITE_ICECHUNK_PATH: The s3 path of the icechunk store holding the input
          satellite data.
        - S3_REGION: The AWS region for the satellite data S3 bucket.

    Output data paths

        - PREDICTION_SAVE_DIRECTORY: The path of the directory to save the predictions to.

    Other settings

        - SCRATCH_DIR: If set, this directory is used as the parent for per-run scratch
          directories named from the forecast init time. These per-run directories are not cleaned
          up by the app. Else, the system temp directory will be used and cleaned up after the run.
    """

    # Input data paths
    satellite_icechunk_path: str
    s3_region: str

    # Output data paths
    prediction_save_directory: str

    # Other settings
    scratch_dir: str | None = None
