# cloudcasting-metrics

This package is used to score the predictions made by the cloudcasting model and save the scores to 
a zarr store at the location `METRIC_ZARR_PATH`. By default, the scoring is run for all init-times 
made the day before running.

## Environment Variables

The following environment variables are used in the app:

- `SATELLITE_ICECHUNK_ARCHIVE`: The path to the satellite archive in icechunk formast
- `PREDICTION_SAVE_DIRECTORY`: The directory where the cloudcasting predictions are saved. 
  i.e. set to the same as `PREDICTION_SAVE_DIRECTORY` in `cloudcasting_inference`.
- `METRIC_ZARR_PATH`: Where to save metrics zarr
