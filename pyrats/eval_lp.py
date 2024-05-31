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


class GridWorldDataset:
    filenames = [
        "final_13",
        "final_14",
        "final_15",
        "final_16",
        "final_17",
        "final_18",
        "final_19",
        "final_1",
        "final_20",
        "final_2",
        "final_3",
        "final_4",
        "final_5",
        "final_6",
        "final_7",
        "final_8",
        "final_9",
    ]

    maps2 = [
'''##########
#GGGTTBTG#
#GGGTT.TT#
#..GTT.TG#
#T.TTT.T##
#..TTT.T##
#T..T..T##
##########''',

'''#############
#..T........#
#..TTGTGTGTG#
#..TT.......#
#..TTGTGTGTG#
#.B...TTTTTT#
#############''',

'''##########
#GTGT.TGT#
#..TG....#
#T.T.TBTT#
#..TG..TG#
#GT..TG..#
##########''',

'''##########
#..GT...G#
#TB.TTGTT#
#..GT.TGT#
#TG......#
#..GTGT.T#
##########''',

"""##########
#TGTGTGTG#
#BTGTGTG.#
#GT.##TGT#
#TG...TG.#
#GTG.GTG.#
##########""",

"""##########
#GTGTGTGT#
#B....G..#
#GTGT.TGT#
#TGTG...T#
#.G.GT.GT#
##########""",

"""#############
#......#....#
#..T.G.#....#
#..TG.....T.#
#..TTGTGTGTG#
#..B..TTTTTT#
#############""",

"""########
#G..TGG#
#T.B..T#
#GT#TGG#
########""",

"""########
#GTG..B#
#.T.TTT#
#.T.TTT#
#....TG#
########""",

"""#########
#TT.G..T#
#..##.#G#
#...GTTG#
#B..TTTG#
#########""",

"""#######
##GGGG#
#BTTGG#
#..TG.#
#T.#T.#
#T.#..#
#T...T#
#######""",

"""
##########
#GTGT.TGT#
#..TG#...#
#T.#.#BTT#
#.#TG#.TG#
#GT..TG..#
##########""",

"""#######
#BTTTG#
#T.T..#
#GT...#
#######""",

"""########
#GT.TTT#
#.TBTT.#
#TT.TTG#
########
""",

"""########
#G..TGG#
#T.BTTT#
#GT#TGG#
########""",

"""#########
#TT.G..##
#..TTTTG#
#...G.TG#
#B..T.TG#
#########""",

"""######
#..G.#
#G#TT#
#.TTT#
##T..#
#.B#G#
#..#T#
#TTTG#
######""",


"""#########
#GGGTTT##
#GGGTT.B#
#..GTT.T#
#T.TTT.G#
#G.TTT.T#
#T.....T#
#########""",

"""#########
#GGGTTT##
#GGGTT.B#
#..GTT.T#
#T.TTT..#
#..TTT.T#
#T.....T#
#########""",

"""##########
#GGGTTT###
#GGGTT.B##
#..GTT.T##
#T.TTT.TG#
#..TTT.T##
#T..T..TG#
##########""",
    ]

    maps = [
'''##########
#..GT...G#
#TB.TTGTT#
#..GT.TGT#
#TG......#
#..GTGT.T#
##########''',

"""##########
#TGTGTGTG#
#BTGTGTG.#
#GT.##TGT#
#TG...TG.#
#GTG.GTG.#
##########""",

"""##########
#GTGTGTGT#
#B....G..#
#GTGT.TGT#
#TGTG...T#
#.G.GT.GT#
##########""",

"""#############
#......#....#
#..T.G.#....#
#..TG.....T.#
#..TTGTGTGTG#
#..B..TTTTTT#
#############""",

"""########
#G..TGG#
#T.B..T#
#GT#TGG#
########""",

"""########
#GTG..B#
#.T.TTT#
#.T.TTT#
#....TG#
########""",

"""#########
#TT.G..T#
#..##.#G#
#...GTTG#
#B..TTTG#
#########""",

"""#######
##GGGG#
#BTTGG#
#..TG.#
#T.#T.#
#T.#..#
#T...T#
#######""",

"""
##########
#GTGT.TGT#
#..TG#...#
#T.#.#BTT#
#.#TG#.TG#
#GT..TG..#
##########""",

"""#######
#BTTTG#
#T.T..#
#GT...#
#######""",

"""########
#GT.TTT#
#.TBTT.#
#TT.TTG#
########
""",

"""########
#G..TGG#
#T.BTTT#
#GT#TGG#
########""",

"""#########
#TT.G..##
#..TTTTG#
#...G.TG#
#B..T.TG#
#########""",

"""######
#..G.#
#G#TT#
#.TTT#
##T..#
#.B#G#
#..#T#
#TTTG#
######""",


"""#########
#GGGTTT##
#GGGTT.B#
#..GTT.T#
#T.TTT.G#
#G.TTT.T#
#T.....T#
#########""",

"""#########
#GGGTTT##
#GGGTT.B#
#..GTT.T#
#T.TTT..#
#..TTT.T#
#T.....T#
#########""",

"""##########
#GGGTTT###
#GGGTT.B##
#..GTT.T##
#T.TTT.TG#
#..TTT.T##
#T..T..TG#
##########""",
    ]

    @classmethod
    def get_maps(cls):
        return list(zip(cls.filenames, cls.maps))


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

    for time_limit in time_limits:
        print( "limit:", time_limit )

        temp_results = ray.get([eval_config_parallel.remote((env, agent, c, p1, p2, time_limit)) for _ in range(repetitions)])

        rews, pens, times, steps = zip(*temp_results)
        mean_time_per_step = np.sum(np.array(times)) / np.sum(np.array(steps))

        mean_r, mean_p, std_p, std_r = np.mean(rews), np.mean(pens), np.std(pens, ddof=1), np.std(rews, ddof=1)

        feasible = ( mean_p - std_p * 1.65 <= c )
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
                    'feasible': feasible,
                    'mean_time_per_step': mean_time_per_step
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

    return pd.DataFrame(res, columns=['reward', 'time', 'computable'])


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
        columns=['benchmark', 'c', 'computable', 'reward', 'time'],
        index=False, sep=';',
    )


    # --------------- Evaluate agents ---------------
    # Run benchmarks for agents, write out results in eval_agents function
    res_agents = eval_agents(agent_list, time_limits, params_grid, agent_repetitions, output_dir)


if __name__ == "__main__":
    set_log_level("DEBUG")

    agents = [agents.ParetoUCT] #agents.RAMCP, agents.DualUCT] #, agents.ParetoUCT]
    time_limits = [5, 10, 25, 50] #, 100, 250, 500]
    grid_desc = {
        'c_s': [0.1, 1, 5, 20], #[0, 0.1, 0.2, 0.3, 0.4],
        'p_slides': [0, 0.2],
        'p_traps': [0.1, 0.7],
        'map': GridWorldDataset.get_maps()
    }
    params_tuples = product(*[grid_desc[key] for key in grid_desc])
    params_grid = [dict(zip(grid_desc.keys(), values)) for values in params_tuples]

    ray.init(address="auto")
    eval_solvers(
        agent_list=agents,
        time_limits=time_limits,
        params_grid=params_grid[:5],
        agent_repetitions=100,
        output_dir="/work/rats/outputs"
    )
    ray.shutdown()
