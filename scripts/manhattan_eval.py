from eval import print_time_estimation, eval_solvers, ask_tag, prepare_output_dir

from rats import envs
import ray
from rats import agents
import pprint
import yaml
import json

from _rats import LP_solver
import time
import numpy as np
from multiprocessing import Pool
from pathlib import Path
from itertools import product
import pandas as pd
from rats.utils import set_log_level
from manhattan_dataset.manhattan_dataset import ManhattanDataset
import asyncio

if __name__ == "__main__":
    ray.init(address="auto")

    agents = [agents.ParetoUCT, agents.RAMCP, agents.DualUCT]
    agent_repetitions = 100
    max_depth = 100
    time_limits = [500] # 1000, 2000]
    dataset_path = '/work/rats/scripts/manhattan_dataset/dense_dataset.txt'
    instances = ManhattanDataset(dataset_path).get_maps()

    grid_desc = {
        'env': [ envs.Manhattan ],
        'c': [5.0, 10.0, 15.0],
        'capacity' : [ 10000 ],
        'period' : [ 50, 100 ],
        'cons_thd' : [ 10, 20 ],
        'radius' : [ 0.2, 0.4 ],
        'instance': instances,
    }

    params_tuples = product(*[grid_desc[key] for key in grid_desc])
    params_grid = [dict(zip(grid_desc.keys(), values)) for values in params_tuples]

    tag = "ManhattanTestLong"
    if tag:
        tag = '-' + tag

    output_dir = Path("/work/rats/outputs/" + time.strftime("%Y%m%d-%H%M%S") + tag)
    grid_desc.copy()
    grid_desc.pop('instance')
    metadata = {
        'agents': agents,
        'agent_repetitions': agent_repetitions,
        'max_depth': max_depth,
        'time_limits': time_limits,
        'params_grid': grid_desc,
        'dataset_path': dataset_path,
        'num_instances': len(instances),
    }
    prepare_output_dir(output_dir, metadata)
    print_time_estimation(agents, agent_repetitions, time_limits, params_grid, max_depth)

    asyncio.run(eval_solvers(
        agent_list=agents,
        time_limits=time_limits,
        max_depth=max_depth,
        params_grid=params_grid,
        agent_repetitions=agent_repetitions,
        output_dir=output_dir,
        run_lp=False
    ))
    ray.shutdown()
