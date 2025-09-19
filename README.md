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
 - `cloudcasting_app`: Used to run inference
 - `cloudcasting_metrics`: Used to score the predictions against ground truth

## Installation

Both packages will be installed simultaneously using

## Setup / Installation

```bash
git clone https://github.com/openclimatefix/cloudcasting-app
cd cloudcasting-app
pip install .
```


## Usage and environmental variables

See the READMEs in `src/cloudcasting_app` and `src/cloudcasting_metrics`.

## Development

### Running the test suite

The test suite is via pytest and can be run from command line using.

```
pytest
```

This will run tests for both packages.
 

## Contributing and community

[![issues badge](https://img.shields.io/github/issues/openclimatefix/cloudcasting-app?color=FFAC5F)](https://github.com/openclimatefix/cloudcasting-app/issues?q=is%3Aissue+is%3Aopen+sort%3Aupdated-desc)

- PR's are welcome! See the [Organisation Profile](https://github.com/openclimatefix) for details on contributing
- Find out about our other projects in the [here](https://github.com/openclimatefix/.github/tree/main/profile)
- Check out the [OCF blog](https://openclimatefix.org/blog) for updates
- Follow OCF on [LinkedIn](https://uk.linkedin.com/company/open-climate-fix)


## Contributors

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/dfulu"><img src="https://avatars.githubusercontent.com/u/41546094?v=4?s=100" width="100px;" alt="James Fulton"/><br /><sub><b>James Fulton</b></sub></a><br /><a href="https://github.com/openclimatefix/cloudcasting-app/commits?author=dfulu" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/devsjc"><img src="https://avatars.githubusercontent.com/u/47188100?v=4?s=100" width="100px;" alt="devsjc"/><br /><sub><b>devsjc</b></sub></a><br /><a href="#infra-devsjc" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/peterdudfield"><img src="https://avatars.githubusercontent.com/u/34686298?v=4?s=100" width="100px;" alt="Peter Dudfield"/><br /><sub><b>Peter Dudfield</b></sub></a><br /><a href="https://github.com/openclimatefix/cloudcasting-app/commits?author=peterdudfield" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://sumana-2705.github.io/Sumana-Portfolio/"><img src="https://avatars.githubusercontent.com/u/110307215?v=4?s=100" width="100px;" alt="Sumana Sree Angajala"/><br /><sub><b>Sumana Sree Angajala</b></sub></a><br /><a href="#infra-sumana-2705" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

---

*Part of the [Open Climate Fix](https://github.com/orgs/openclimatefix/people) community.*

[![OCF Logo](https://cdn.prod.website-files.com/62d92550f6774db58d441cca/6324a2038936ecda71599a8b_OCF_Logo_black_trans.png)](https://openclimatefix.org)
