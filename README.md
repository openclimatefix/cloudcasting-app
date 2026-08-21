# cloudcasting-app
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-4-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->
 
[![contributors badge](https://img.shields.io/github/contributors/openclimatefix/cloudcasting-app?color=FFFFFF)](https://github.com/openclimatefix/cloudcasting-app/graphs/contributors)
[![ease of contribution: hard](https://img.shields.io/badge/ease%20of%20contribution:%20hard-bb2629)](https://github.com/openclimatefix#how-easy-is-it-to-get-involved)

This repo is used to run the OCF-ATI cloudcasting model live in production and meausure its 
performance. The model takes previous frames of EUMETSAT satellite images and forecasts the future 
frames to come.

This repo contains two different packages:
 - `cloudcasting_inference`: Used to run inference
 - `cloudcasting_metrics`: Used to score the predictions against ground truth

## Installation

## Setup / Installation

Both packages will be installed simultaneously using:

```bash
git clone https://github.com/openclimatefix/cloudcasting-app
cd cloudcasting-app
pip install .
```

## Usage and environmental variables

See the READMEs in `src/cloudcasting_inference` and `src/cloudcasting_metrics`.

## Development

### Running the test suite

The test suite is via pytest and can be run from command line using:

```
pytest
```

This will run tests for both packages.
 

*Part of the [Open Climate Fix](https://github.com/orgs/openclimatefix/people) community.*

[![OCF Logo](https://cdn.prod.website-files.com/62d92550f6774db58d441cca/6324a2038936ecda71599a8b_OCF_Logo_black_trans.png)](https://openclimatefix.org)
