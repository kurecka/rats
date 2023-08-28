[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)
[![Build Status](https://travis-ci.org/bsamseth/cpp-project.svg?branch=master)](https://travis-ci.org/bsamseth/cpp-project)
[![Build status](https://ci.appveyor.com/api/projects/status/g9bh9kjl6ocvsvse/branch/master?svg=true)](https://ci.appveyor.com/project/bsamseth/cpp-project/branch/master)
[![Coverage Status](https://coveralls.io/repos/github/bsamseth/cpp-project/badge.svg?branch=master)](https://coveralls.io/github/bsamseth/cpp-project?branch=master)
[![codecov](https://codecov.io/gh/bsamseth/cpp-project/branch/master/graph/badge.svg)](https://codecov.io/gh/bsamseth/cpp-project)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/eb004322b0d146239a57eb242078e179)](https://www.codacy.com/app/bsamseth/cpp-project?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=bsamseth/cpp-project&amp;utm_campaign=Badge_Grade)
[![Language grade: C/C++](https://img.shields.io/lgtm/grade/cpp/g/bsamseth/cpp-project.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/bsamseth/cpp-project/context:cpp)
[![Total alerts](https://img.shields.io/lgtm/alerts/g/bsamseth/cpp-project.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/bsamseth/cpp-project/alerts/)
[![license](https://img.shields.io/badge/license-Unlicense-blue.svg)](https://github.com/bsamseth/cpp-project/blob/master/LICENSE)
[![Lines of Code](https://tokei.rs/b1/github/bsamseth/cpp-project)](https://github.com/Aaronepower/tokei)
[![Average time to resolve an issue](http://isitmaintained.com/badge/resolution/bsamseth/cpp-project.svg)](http://isitmaintained.com/project/bsamseth/cpp-project "Average time to resolve an issue")
[![Percentage of issues still open](http://isitmaintained.com/badge/open/bsamseth/cpp-project.svg)](http://isitmaintained.com/project/bsamseth/cpp-project "Percentage of issues still open")

# Risk-Aware Tree Search
This repository contains code base for development and testing of risk-aware tree search based methods.

## Usage: help
Build and run `./rats --help`.

## Rollout method
Currently only rollout method is constant zeros.

## Algorithms
- randomized: Play random action
- c0: Play action 0 constantly
- c1: Play action 1 constantly
- ts: Simple tree search. Tries to achieve the threshold at each state.
- pts: Pareto tree search. OUT OF ORDER
- dts: Dual tree search. Implements the dual algorithm CC-POMCP.
  - Does not match the decribed algorithm perfectyl. In particular, it does not use LP, just thresholding to select best greedy action.

## Todo
- [x] Pytorch binding
- [ ] Ray distributed computing
- [ ] Evironment management
- [ ] Experiment configuration files
- [ ] Mongo reporting

## Building

Make sure to have `cmake` installed

``` bash
> cd build
> cmake .. -DCMAKE_BUILD_TYPE=[Debug | Coverage | Release]
> make
> ./rats_app ...
> make test      # Makes and runs the tests.
> make coverage  # Generate a coverage report.
> make doc       # Generate html documentation.
```

In order to build and install as a python package, run the following commands:
```bash
> pip install .
```
The command above will build the code using cmake and install the package in the current environment as `rats` package.

## Install pytjon packages
We use `conda` to maintain python packages. Make sure you have it installed.
The environment is described in `conda_env.yaml`.

To create new environment according to the env file use:
```bash
conda env create -f conda_env.yaml
```

To activate the environment use:
```bash
conda activate rats
```

To update your environment according to the env file use:
```bash
conda env update -f conda_env.yaml
```

To export the description of your current conda environment use:
```bash
conda env export > conda_env.yml
```
