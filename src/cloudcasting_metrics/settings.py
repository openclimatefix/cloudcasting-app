"""Runtime configuration loaded from environment."""

from pydantic_settings import BaseSettings


class AppSettings(BaseSettings):
    """Runtime configuration loaded from environment.

    Input data paths

        - SATELLITE_ICECHUNK_ARCHIVE: Path at which ground truth satellite data can be found.
        - PREDICTION_SAVE_DIRECTORY: The directory where the cloudcasting forecasts are saved.

    Output data paths

        - METRIC_ZARR_PATH: The path where the metric values will be saved.

    If SATELLITE_ICECHUNK_ARCHIVE is an s3 path, then the environment variables
    AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY and AWS_REGION must also be set. These are read
    directly from the environment by icechunk rather than through these settings.
    """

    # Input data paths
    satellite_icechunk_archive: str
    prediction_save_directory: str

    # Output data paths
    metric_zarr_path: str
