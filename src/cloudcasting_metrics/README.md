# cloudcasting-metrics

This package is used to score the predictions made by the model. By default, the scoring is run for 
all init-times made the day before.

## Environment Variables

The following environment variables are used in the app:

- `SATELLITE_ICECHUNK_ARCHIVE`: The path to the satellite archive in icechunk formast
- `PREDICTION_SAVE_DIRECTORY`: The directory where the cloudcasting predictions are saved. 
  i.e. set to the same as `PREDICTION_SAVE_DIRECTORY` in `cloudcasting_inference`.
- `METRIC_ZARR_PATH`: Where to save metrics zarr
