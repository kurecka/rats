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
- tuct: Threshold tree search. Implements the Pareto curve-based algorithm T-UCT.
- ccpomcp: Dual tree search. Implements the dual algorithm CC-POMCP.
- ramcp: Risk-aware POMCP. Implements the LP-based algorithm RAMCP.

## Building
Build the docker image accoring to the specified Dockerfile.

## Maintaining Ray cluster
You can use `raylite` as a simplified version of malfunctioning `ray` to start and manage ray cluster. 
To install `raylite` you can use the following command:
```sh
cd raylite
pip install -e .
```

## Run experiment through ray
If you have your own ray cluster running or want to run an experiment on a local machine, you can use the following command:
```
python experiment.py -m +task=train_runs +agent=tuct ++agent.exploration_constant=1,5,15 ++risk_thd=0,0.1,0.2,0.3,0.5 ++agent.sim_time_limit=10,50,200 +env=large_hw ++metadata.tag=predictors ++task.num_episodes=300 ++gamma=0.999 ++agent.risk_exploration_ratio=0.01,0.1,1
```

If you want to run to the experiment on erinys cluster, you can connect to erinys02 and run a command similar to the following:
```
ray job submit --no-wait -- sh -c 'cd /work/rats/pyrats && python experiment.py -m +task=indep_runs +agent=ccpomcp ++agent.exploration_constant=0.1,1,5 ++agent.sim_time_limit=10,50,100 +env=slide_hw ++metadata.tag=dual-slide_hw ++task.num_episodes=30 ++gamma=0.99999 ++risk_thd=0,0.16,0.2'
```
