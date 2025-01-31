# cloudcasting-app
 
[![contributors badge](https://img.shields.io/github/contributors/openclimatefix/ocf-template?color=FFFFFF)](https://github.com/openclimatefix/ocf-template/graphs/contributors)
[![ease of contribution: hard](https://img.shields.io/badge/ease%20of%20contribution:%20hard-bb2629)](https://github.com/openclimatefix#how-easy-is-it-to-get-involved)

This repo is used to run the OCF-ATI cloudcasting model live in production. This model takes
previous frames of EUMETSAT satellite images and forecasts the future frames to come.

The repo associated with training the models run here is https://github.com/openclimatefix/sat_pred

The model checkpoints are hosted at:
https://huggingface.co/openclimatefix/cloudcasting_uk


## Installation

## Setup / Installation

```bash
git clone https://github.com/openclimatefix/cloudcasting-app
cd cloudcasting-app
pip install .
```


## Example usage

### Running the app locally

It is possbile to run the app locally by setting the required environment variables listed at the top of the [app](src/cloudcasting_app/app.py), these should point to the relevant paths for loading satellite data 
and sving predicitons.

## Development

### Running the test suite

The test suite is via pytest and can be run from command line using

```
pytest
```
 

## Contributing and community

[![issues badge](https://img.shields.io/github/issues/openclimatefix/ocf-template?color=FFAC5F)](https://github.com/openclimatefix/ocf-template/issues?q=is%3Aissue+is%3Aopen+sort%3Aupdated-desc)

- PR's are welcome! See the [Organisation Profile](https://github.com/openclimatefix) for details on contributing
- Find out about our other projects in the [here](https://github.com/openclimatefix/.github/tree/main/profile)
- Check out the [OCF blog](https://openclimatefix.org/blog) for updates
- Follow OCF on [LinkedIn](https://uk.linkedin.com/company/open-climate-fix)


## Contributors

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

---

*Part of the [Open Climate Fix](https://github.com/orgs/openclimatefix/people) community.*

[![OCF Logo](https://cdn.prod.website-files.com/62d92550f6774db58d441cca/6324a2038936ecda71599a8b_OCF_Logo_black_trans.png)](https://openclimatefix.org)
