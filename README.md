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


## Algorithms
- randomized: Play random action
- primal_uct: Simple tree search. Tries to achieve the threshold at each state.
- pareto_uct: Pareto tree search.
- dual_uct: Dual tree search. Implements the dual algorithm CC-POMCP.

## Todo
- [x] Pytorch binding
- [x] Link LP library
- [x] Implement insteraction with AI gym with [pybind11](https://pybind11.readthedocs.io/en/stable/advanced/embedding.html#executing-python-code)
- [x] Ray distributed computing
- [x] Evironment management
- [x] Experiment configuration files
- [x] CSV reporting

## Building

Make sure to have `cmake` and the following c++ libraries installed: [`OR-tools`](https://github.com/google/or-tools), spdlog, pybind11, eigen

``` bash
> cd build
> cmake .. -DCMAKE_BUILD_TYPE=[Debug | Release]
> make
> ./regtest      # Runs the tests.
> make doc       # Generate html documentation.
```

In order to build and install as a python package, run the following command:
```bash
> pip install .
```
The command above will build the code using cmake and install the package in the current environment as `rats` package.

## Install OR-tools
We use `OR-tools` as LP solver. Make sure you have it installed.
Download the latest version from [here](https://github.com/google/or-tools).
The following commands should work:
```bash
> cd or-tools
> cmake -S . -B build -DBUILD_DEPS:BOOL=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-O2"
> sudo cmake --build build --config Release --target install  # install the libraries in the system
```

## Install python packages
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

<!-- To export the description of your current conda environment use:
```bash
conda env export --no-build --from-history | grep -v prefix > conda_env.yaml
``` -->


## Run experiment through ray
If you have your own ray cluster running or want to run an experiment on a local machine, you can use the following command:
```
python experiment.py -m +task=train_runs +agent=pareto_uct ++agent.exploration_constant=1,5,15 ++risk_thd=0,0.1,0.2,0.3,0.5 ++agent.sim_time_limit=10,50,200 +env=large_hw ++metadata.tag=predictors ++task.num_episodes=300 ++gamma=0.999 ++agent.risk_exploration_ratio=0.01,0.1,1
```

If you want to run to the experiment on erinys cluster, you can connect to erinys02 and run a command similar to the following:
```
ray job submit --no-wait -- sh -c 'cd /work/rats/pyrats && python experiment.py -m +task=indep_runs +agent=dual_uct ++agent.exploration_constant=0.1,1,5 ++agent.sim_time_limit=10,50,100 +env=slide_hw ++metadata.tag=dual-slide_hw ++task.num_episodes=30 ++gamma=0.99999 ++risk_thd=0,0.16,0.2'
```
