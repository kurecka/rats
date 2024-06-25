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


def RolloutParetoUCT(*args, **kwargs):
    return rats.agents.ParetoUCT(*args, **kwargs, rollout=True)


def RolloutRAMCP(*args, **kwargs):
    return rats.agents.RAMCP(*args, **kwargs, rollout=True)


def RolloutDualUCT(*args, **kwargs):
    return rats.agents.DualUCT(*args, **kwargs, rollout=True)


if __name__ == "__main__":
    ray.init(address="auto")

    agents = [
        rats.agents.ParetoUCT, RolloutParetoUCT,
        rats.agents.RAMCP, RolloutRAMCP,
        rats.agents.DualUCT, RolloutDualUCT,
    ]
    agent_repetitions = 200
    max_depth = 500
    time_limits = [25, 50, 100]
    dataset_paths = [
        'gridworld_generator/HW_LARGE.txt',
    ]
    instances = []
    for dataset_path in dataset_paths:
        instances += GridWorldDataset(dataset_path, base=len(instances)+1).get_maps()
    grid_desc = [
        {
            'env': [rats.envs.Hallway],
            'c': [0, 0.2, 0.5],
            'trap_prob': [0.05],
            'slide_prob': [0, 0.2],
            'instance': instances,
        }, 
        {
            'env': [rats.envs.ContHallway],
            'c': [0.2, 0.5, 0.8],
            'trap_prob': [0.05],
            'slide_prob': [0, 0.2],
            'instance': instances,
        }
    ]

    params_grid = desc2grid(grid_desc)

    tag = 'HWLarge_Fast'

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
    ))
    ray.shutdown()
