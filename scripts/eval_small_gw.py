import ray
import rats
from itertools import product
import time
from pathlib import Path
import asyncio


from gridworld_generator.dataset import GridWorldDataset
from eval import (
    eval_solvers,
    prepare_output_dir,
    ask_tag,
    print_time_estimation,
    desc2grid,
    desc2metadata,
)


def RolloutTUCT(*args, **kwargs):
    return rats.agents.TUCT(*args, **kwargs, rollout=True)


def RolloutRAMCP(*args, **kwargs):
    return rats.agents.RAMCP(*args, **kwargs, rollout=True)


def RolloutCCPOMCP(*args, **kwargs):
    return rats.agents.CCPOMCP(*args, **kwargs, rollout=True)


if __name__ == "__main__":
    ray.init()

    agents = [
        RolloutTUCT,
        RolloutRAMCP,
        RolloutCCPOMCP,
    ]
    agent_repetitions = 300
    max_depth = 100
    time_limits = [5, 10, 25]
    dataset_paths = [
        '/work/rats/scripts/gridworld_generator/GW_SMALL.txt',
        '/work/rats/scripts/gridworld_generator/GW_OLD.txt',
    ]
    instances = []
    for dataset_path in dataset_paths:
        instances += GridWorldDataset(dataset_path, base=len(instances)+1).get_maps()
    grid_desc = [
        {
            'env': [rats.envs.Avoid],
            'c': [0, 0.15, 0.35],
            'trap_prob': [0.2, 0.5],
            'slide_prob': [0, 0.2],
            'instance': instances,
        }, {
            'env': [rats.envs.SoftAvoid],
            'c': [0, 0.15, 0.3, 0.45, 0.6, 0.75],
            'trap_prob': [0.2],
            'slide_prob': [0, 0.2],
            'instance': instances,
        }
    ]

    params_grid = desc2grid(grid_desc)

    tag = 'GWSmall'

    output_dir = Path("/work/rats/outputs/" + time.strftime("%Y%m%d-%H%M%S") + tag)
    metadata = {
            'agents': agents,
            'agent_repetitions': agent_repetitions,
        'max_depth': max_depth,
        'time_limits': time_limits,
        'params_grid': desc2metadata(grid_desc),
        'dataset_paths': dataset_paths,
        'num_instances': len(instances),
    }
    prepare_output_dir(output_dir, metadata)
    print_time_estimation(agents, agent_repetitions, time_limits, params_grid, max_depth)

    asyncio.run(eval_solvers(
        agent_list=agents,
        time_limits=time_limits,
        params_grid=params_grid,
        agent_repetitions=agent_repetitions,
        output_dir=output_dir,
        max_depth=max_depth,
        gamma=0.99,
    ))
    ray.shutdown()
