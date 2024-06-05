import envs
import ray
import agents

from rats import Hallway, LP_solver
import time
import numpy as np
from multiprocessing import Pool
from pathlib import Path
from itertools import product
import pandas as pd
from utils import set_log_level
from gridworld_generator.dataset import GridWorldDataset


def eval_agent_config(env, agent, c, slide, trap, time_limit):
    """
    Performs a single run of the agent on the environment
    
    Returns: (collected (non-discounted) reward, collected (non-discounted) penalty, total_time, steps)
    """
    e = envs.Hallway( env, trap, slide )
    h = envs.EnvironmentHandler(e, 100)

    a = agent(
        h,
        max_depth=100, num_sim=1000, sim_time_limit=time_limit, risk_thd=c, gamma=0.99,
        exploration_constant=5
    )

    a.reset()

    total_time = 0
    steps = 0
    while not a.get_handler().is_over():
        start_time = time.time()
        a.play()
        end_time = time.time()
        total_time += end_time - start_time
        steps += 1

    h = a.get_handler()
    rew = h.get_reward()
    p = h.get_penalty()
    return (rew, p, total_time, steps)


def eval_lp_config( env, c, slide, trap ):
    """
    Evaluates LP solver on environment

    Returns: (reward, time, feasible)
    """

    # ignore infeasible environments
    if (env.count('G') > 10):
        return (0, -1, False)

    e = envs.Hallway(env, trap, slide)
    start = time.time()
    lp_solver = LP_solver(e, c)
    lp_solver.change_gammas( 0.99 )

    feasible = True
    try:
        rew = lp_solver.solve()

    except RuntimeError:
        rew = 0
        feasible = False

    end_time = ( time.time() - start ) * 1000
    return (rew, end_time, feasible)

def eval_lp_config_parallel(args):
    """
    Wrapper for parallel evaluation of LP solver
    """
    env, c, slide, trap = args['map'][1], args['c_s'], args['p_slides'], args['p_traps']
    return eval_lp_config(env, c, slide, trap)

@ray.remote
def eval_config_parallel(args):
    """
    Wrapper for parallel evaluation of agent
    """
    env, agent_type, c, slide, trap, time_limit = args
    return eval_agent_config(env, agent_type, c, slide, trap, time_limit)


# modifies the results table, adds average of _repetitions_ repetitions of each pareto
# config ( each time limit in time_limits )
def evaluate_agent( env, filename, agent, c, p1, p2, results, time_limits, repetitions=100):
    """
    Runs agent on environment _repetitions_ times for each time_limit in time_limits
    """

    results = []

    ray_results = ray.get([eval_config_parallel.remote((env, agent, c, p1, p2, time_limit)) for time_limit in time_limits for _ in range(repetitions)])

    temp_results_list = [ray_results[i:i+repetitions] for i in range(0, len(ray_results), repetitions)]

    for temp_results, time_limit in zip(temp_results_list, time_limits):
        rews, pens, times, steps = zip(*temp_results)
        mean_time_per_step = np.sum(np.array(times)) / np.sum(np.array(steps))

        mean_r, mean_p, std_p, std_r = np.mean(rews), np.mean(pens), np.std(pens, ddof=1), np.std(rews, ddof=1)

        results.append(
            pd.DataFrame(
                [{
                    'filename': filename,
                    'c': c,
                    'p_slide': p1,
                    'p_trap': p2,
                    'time_limit': time_limit,
                    'mean_reward': mean_r,
                    'std_reward': std_r,
                    'mean_penalty': mean_p,
                    'std_penalty': std_p,
                    'mean_time_per_step': mean_time_per_step,
                    'repetitions': repetitions
                }]
            )
        )
    
    return pd.concat(results, ignore_index=True)


def eval_agents(agents_list, time_limits, params_grid, repetitions=100, output_path="/work/rats/pyrats"):
    output_path = Path(output_path)
    results = []

    for params in params_grid:
        for agent in agents_list:
            (map_name, env), c, p1, p2 = params['map'], params['c_s'], params['p_slides'], params['p_traps']

            print(f"Solving with params: c={c}, p_slide={p1}, p_trap={p2}, agent={agent.__name__}")
            print(env)
            agent_res = evaluate_agent( env, map_name, agent, c, p1, p2, results, time_limits, repetitions)

            if (output_path / f"results_{agent.__name__}.csv").exists():
                saved_res = pd.read_csv(output_path / f"results_{agent.__name__}.csv", sep=';')
                joined_res = pd.concat([saved_res, agent_res], ignore_index=True)
            else:
                joined_res = agent_res
            joined_res.to_csv(output_path / f"results_{agent.__name__}.csv", index=False, sep=';' )
            results.append(agent_res)
    return pd.concat(results, ignore_index=True)

def eval_lp(params_grid):
    """
        Evaluate LP solver on all configurations in params_grid
    """
    pool = Pool()
    res = pool.map(eval_lp_config_parallel, params_grid)

    pool.close()
    pool.join()

    return pd.DataFrame(res, columns=['reward', 'time', 'feasible'])


def eval_solvers(agent_list, time_limits, params_grid, agent_repetitions=100, output_dir="/work/rats/pyrats"):
    output_dir = Path(output_dir)

    # --------------- Evaluate LP solver ---------------
    # Prepare configuration descriptions
    benchmarks = []
    cs = []

    for params in params_grid:
        (map_name, map), c, p1, p2 = params['map'], params['c_s'], params['p_slides'], params['p_traps']

        benchmark = f"{map_name}_c{c}_slide{p1}_trap{p2}"
        benchmarks.append(benchmark)
        cs.append(c)
    
    benchmarks = pd.Series(benchmarks, name='benchmark')
    cs = pd.Series(cs, name='c')

    # Compute LP results
    res_lp = eval_lp(params_grid)

    # Save results
    res_lp = pd.concat([benchmarks, cs, res_lp], axis=1)
    res_lp.to_csv(
        output_dir / "results_lp.csv",
        columns=['benchmark', 'c', 'feasible', 'reward', 'time'],
        index=False, sep=';',
    )


    # --------------- Evaluate agents ---------------
    # Run benchmarks for agents, write out results in eval_agents function
    res_agents = eval_agents(agent_list, time_limits, params_grid, agent_repetitions, output_dir)


if __name__ == "__main__":
    agents = [agents.ParetoUCT, agents.RAMCP, agents.DualUCT]
    time_limits = [5, 10, 25, 50]
    dataset = 'gridworld_generator/HW_SMALL.txt'
    grid_desc = {
        'c_s': [0, 0.1, 0.2, 0.35, 0.5],
        'p_slides': [0.2],
        'p_traps': [0.1],
        'map': GridWorldDataset(dataset).get_maps()
    }
    params_tuples = product(*[grid_desc[key] for key in grid_desc])
    params_grid = [dict(zip(grid_desc.keys(), values)) for values in params_tuples]

    output_dir = Path("/work/rats/outputs/" + time.strftime("%Y%m%d-%H%M%S"))
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "metadata.txt", "w") as f:
        f.write("Agents: ")
        for agent in agents:
            f.write(agent.__name__ + ", ")
        f.write("\nTime limits: ")
        f.write(str(time_limits))
        f.write("\nParams grid:")
        grid_desc_copy = grid_desc.copy()
        grid_desc_copy.pop('map')
        grid_desc_copy['dataset'] = dataset
        f.write(str(grid_desc_copy))
        f.write("\n")

    with open(output_dir / "metadata.txt", "r") as f:
        print(f.read())

    ray.init(address="auto")
    eval_solvers(
        agent_list=agents,
        time_limits=time_limits,
        params_grid=params_grid,
        agent_repetitions=100,
        output_dir=output_dir,
    )
    ray.shutdown()
