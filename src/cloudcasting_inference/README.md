# cloudcasting-inference

This package is used to run inference on the model. This model takes previous frames of EUMETSAT 
satellite images and forecasts the future frames to come.

The repo associated with training the models run here is https://github.com/openclimatefix/sat_pred

The model checkpoints are hosted at:
https://huggingface.co/openclimatefix/cloudcasting_uk

## Environment Variables

The following environment variables are used in the app:

- `SATELLITE_ZARR_PATH`: The path to the satellite data in Zarr format.
- `PREDICTION_SAVE_DIRECTORY`: The directory where predictions will be saved. 

### Optional Environment Variables

- `SATELLITE_15_ZARR_PATH`: The path to the 15 minute satellite data in Zarr format. If 
this is not set then the `SATELLITE_ZARR_PATH` is used by `.zarr` is repalced with `_15.zarr`

## Example usage

### Running the app locally

It is possible to run the app locally by setting the required environment variables listed at the
top of the [inference app](src/cloudcasting_inference/app.py), these should point to the relevant 
paths for loading satellite data and saving predicitons.

You can then run the app using `uv run cloudcasting-inference`.s
