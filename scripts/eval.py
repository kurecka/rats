#!/usr/bin/python3

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
from gridworld_generator.dataset import GridWorldDataset
import asyncio


@ray.remote
def eval_agent_config(agent, time_limit, params, max_depth=100, gamma=0.99, exploration_constant=5):
    """
    Performs a single run of the agent on the environment

    Returns: (collected (non-discounted) reward, collected (non-discounted) penalty, total_time, steps)
    """
    params = params.copy()
    env = params.pop('env')
    instance_name, instance = params.pop('instance')
    c = params.pop('c')

    e = env(**instance, **params)
    h = envs.EnvironmentHandler(e, max_depth)

    a = agent(
        h,
        max_depth=max_depth, num_sim=0, sim_time_limit=time_limit,
        exploration_constant=exploration_constant,
        risk_thd=c, gamma=gamma,
    )

    a.reset()

    steps = 0
    start_time = time.time()
    while not a.get_handler().is_over():
        a.play()
        steps += 1

    h = a.get_handler()
    rew = h.get_reward()
    p = h.get_penalty()
    total_time = (time.time() - start_time)
    res = {
        'agent': agent.__name__,
        'time_limit': time_limit,
        'c': c,
        'env': env.__name__,
        'instance': instance_name,
        'reward': rew,
        'penalty': p,
        'time': total_time,
        'steps': steps,
    }
    res.update(params)
    return res


@ray.remote
def eval_lp_config(params, gamma=0.99):
    """
    Evaluates LP solver on environment

    Returns: (reward, time, feasible)
    """

    params = params.copy()
    env = params.pop('env')
    instance_name, instance = params.pop('instance')
    c = params.pop('c')

    e = env(**instance, **params)
    start = time.time()
    lp_solver = LP_solver(e, c)
    lp_solver.change_gammas(gamma)

    try:
        rew = lp_solver.solve()
        p = 0

    except RuntimeError:
        rew = 0
        p = np.inf

    total_time = time.time() - start
    res = {
        'agent': 'LP',
        'time_limit': -1,
        'c': c,
        'env': env.__name__,
        'instance': instance_name,
        'reward': rew,
        'penalty': p,
        'time': total_time,
        'steps': 0,
    }
    res.update(params)
    return res


def aggregate_results(results):
    """
    Aggregates results from multiple runs of the same configuration
    """
    results = pd.DataFrame(results)
    aggregate_by = ['agent', 'time_limit', 'c', 'env', 'instance']
    aggrgate_on = {
        'reward': ['mean', 'std'],
        'penalty': ['mean', 'std'],
        'time': ['mean', 'std', 'min', 'max'],
        'steps': ['mean', 'std', 'min', 'max'],
    }
    static_cols = [col for col in results.columns if col not in aggregate_by and col not in aggrgate_on]
    aggregate_by += static_cols
    aggregated = results.groupby(aggregate_by).agg(aggrgate_on).reset_index()
    aggregated['repetitions'] = results.groupby(aggregate_by).size().values
    assert results.groupby(aggregate_by).ngroups == 1, "An aggregated group contains multiple configurations!"

    aggregated.columns = ['_'.join(c for c in col if c).strip() for col in aggregated.columns.values]
    return aggregated


def eval_config(agent, time_limit, params, agent_repetitions=1, max_depth=100):
    """
    Evaluate a single solver on the given configurations
    """
    futures = []
    for _ in range(agent_repetitions):
        if agent == 'LP':
            res = eval_lp_config.remote(params)
        else:
            res = eval_agent_config.remote(agent, time_limit, params, max_depth=max_depth)
        futures.append(res)
    return asyncio.gather(*futures)


async def process_futures(futures, output_dir):
    print(f"Processing batch of {len(futures)} futures")
    for completed in asyncio.as_completed(futures):
        result = aggregate_results(await completed)

        central_file = output_dir / "results.csv"
        if central_file.exists():
            result.to_csv(central_file, mode='a', header=False, index=False)
        else:
            result.to_csv(central_file, index=False)

    futures.clear()


async def eval_solvers(
        agent_list, time_limits, params_grid,
        agent_repetitions=100,
        max_depth=100,
        output_dir="/work/rats/rats",
    ):
    output_dir = Path(output_dir)

    def iterate_configs():
        for params in params_grid:
            for agent in agent_list:
                for time_limit in time_limits:
                    yield agent, time_limit, params

    futures = []
    # Run LP solver
    for params in params_grid:
        futures.append(eval_config('LP', None, params))
    await process_futures(futures, output_dir)

    # Run agent solvers
    for agent, time_limit, params in iterate_configs():
        futures.append(eval_config(agent, time_limit, params, agent_repetitions=agent_repetitions, max_depth=max_depth))
        if len(futures) >= 8000 / agent_repetitions:
            await process_futures(futures, output_dir)

    if futures:
        await process_futures(futures, output_dir)

    print("All configurations evaluated.")


def prepare_output_dir(output_dir: Path, metadata: dict):
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "metadata.yaml", "w") as f:
        yaml.dump(metadata, f)

    with open(output_dir / "metadata.yaml", "r") as f:
        print("Experiment description:")
        print(f.read())


def ask_tag():
    tag = input("Enter a tag for this experiment (press enter to skip): ")
    return tag


def print_time_estimation(agents, agent_repetitions, time_limits, params_grid, num_steps, cores=100):
    time_per_conf = sum(time_limits) * num_steps * agent_repetitions
    estimated_ms = time_per_conf * len(agents) * len(params_grid) / cores
    estimated_h = estimated_ms / 3600 / 1000
    print(f"Estimated max time: {estimated_h:.2f} hours", end="\n\n")


if __name__ == "__main__":
    ray.init(address="auto")

    agents = [agents.ParetoUCT, agents.RAMCP, agents.DualUCT]
    agent_repetitions = 100
    max_depth = 100
    time_limits = [5, 10]#, 25, 50]
    dataset_path = 'gridworld_generator/HW_SMALL.txt'
    instances = GridWorldDataset(dataset_path).get_maps()[:2]
    grid_desc = {
        'env': [envs.Hallway, envs.ContHallway],
        'c': [0, 0.1, 0.2, 0.35, 0.5],
        'trap_prob': [0.2, 0.7],
        'slide_prob': [0, 0.2],
        'instance': instances,
    }
    params_tuples = product(*[grid_desc[key] for key in grid_desc])
    params_grid = [dict(zip(grid_desc.keys(), values)) for values in params_tuples]

    tag = ask_tag()
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
        params_grid=params_grid,
        agent_repetitions=agent_repetitions,
        output_dir=output_dir,
    ))
    ray.shutdown()
