# cloudcasting-inference

This package is used to run inference on the model. This model takes previous frames of EUMETSAT 
satellite images and forecasts the future frames to come.

The repo associated with training the models run here is https://github.com/openclimatefix/sat_pred

The model checkpoints are hosted at:
https://huggingface.co/openclimatefix/cloudcasting_uk

## Environment Variables

The following environment variables are used in the app:

- `SATELLITE_ICECHUNK_PATH`: The s3 path to the icechunk store holding the satellite data.
- `S3_REGION`: The AWS region of the bucket the satellite icechunk store is in.
- `PREDICTION_SAVE_DIRECTORY`: The directory where predictions will be saved. 

The satellite store is read with the credentials in the environment, so `AWS_ACCESS_KEY_ID` and
`AWS_SECRET_ACCESS_KEY` must also be set.

### Optional Environment Variables

- `SCRATCH_DIR`: If set, the satellite data downloaded for each run is saved to a directory inside
this one named from the forecast init-time, and is left in place after the run. Otherwise the
system temp directory is used and is cleaned up when the run finishes.

## Example usage

### Running the app locally

It is possible to run the app locally by setting the required environment variables listed at the
top of the [inference app](src/cloudcasting_inference/app.py), these should point to the relevant 
paths for loading satellite data and saving predicitons.

You can then run the app using `uv run cloudcasting-inference`.s
