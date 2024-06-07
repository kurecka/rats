from rats import envs
import ray
from rats import agents

from _rats import LP_solver
from math import sqrt
from rats.utils import set_log_level
import time
import numpy as np
from multiprocessing import Pool


ray.init(address="auto")

filenames2 = [
    "final_10",
    "final_11",
    "final_12",
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

"""
settings:
    c_0 ∈ C = { 0, 0.1, 0.2, 0.3, 0.4 } ,
    p_slide ∈ S = { 0, 0.2 } , and
    p_trap ∈ R = { 0.1, 0.7 } . """

# setting global config
c_s = [ 0, 0.1, 0.2, 0.3, 0.4 ]
p_slides = [ 0, 0.2 ]
p_traps = [ 0.1, 0.7 ]
time_limits = [5, 10, 25, 50, 100, 250, 500]

# debug settings
#c_s = [0]
#p_slides = [ 0 ]
#p_traps = [ 0.1 ]
# time_limits = [5, 10, 15, 25]
agent_list = [agents.ParetoUCT, agents.DualUCT, agents.RAMCP]


def eval_config( env, agent, c, slide, trap, time_limit ):
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
    env, c, slide, trap = args
    return eval_lp_config(env, c, slide, trap)

@ray.remote
def eval_config_parallel(args):
    env, agent_type, c, slide, trap, time_limit = args
    return eval_config(env, agent_type, c, slide, trap, time_limit)


# modifies the results table, adds average of _repetitions_ repetitions of each pareto
# config ( each time limit in time_limits )
def process_agent_runs( env, filename, agent, c, p1, p2, results, time_limits, repetitions=100):

    result_row = []

    for time_limit in time_limits:
        print( "limit:", time_limit )

        temp_results = ray.get([eval_config_parallel.remote((env, agent,  c, p1, p2, time_limit)) for _ in range(repetitions)])

        rews, pens, times, steps = zip(*temp_results)
        mean_time_per_step = np.sum(np.array(times)) / np.sum(np.array(steps))

        mean_r, mean_p, std_p, std_r = np.mean(rews), np.mean(pens), np.std(pens, ddof=1), np.std(rews, ddof=1)

        feasible = ( mean_p - std_p * 1.65 <= c )
        result_row.append( (mean_r, std_r, mean_p, std_p, feasible, mean_time_per_step ) )

    line = process_pareto_line( time_limits, result_row )
    results.append( result_row )

    return line


def eval_agents(agents_list, time_limits, repetitions=100):

    results = []

    for i in range(len(maps)):
        env = maps[i]
        for c in c_s:
            for p1 in p_slides:
                for p2 in p_traps:
                    for agent in agents_list:
                        print(f"Solving with params: c={c}, p_slide={p1}, p_trap={p2}, agent={agent.__name__}")
                        print(env)
                        line = f"{filenames[i]}_c{c}_slide{p1}_trap{p2};{c}"
                        line += process_agent_runs( env, filenames[i], agent, c, p1, p2, results, time_limits, repetitions)
                        with open(f"/work/rats/pyrats/results_{agent.__name__}.csv", 'a') as file:
                            file.write( line[:-1] + "\n")

    return results

def eval_lp():

    pool = Pool()
    res = pool.map(eval_lp_config_parallel, [(env, c, p1, p2)
                                             for env in maps
                                             for c in c_s
                                             for p1 in p_slides
                                             for p2 in p_traps ] )

    pool.close()
    pool.join()
    return res


"""
    calculates cr, max_time, total infeasible, (later ncr?) for lp and all
    pareto configs.

    lp results is an array of triplets ( reward, time, feasible )
    pareto results is an array of arrays of triplets, each row of the array
    corresponds to results of each configuration.

    if time_limits = { 5, 10, 100 } then each row is [ (rew, time, feasible) x3
    for each config (sim_time_limit = t) for t in time_limits]

    env_count to override avg in cr calculation
        ( if some envs should be ignored )

    return array of triplets ( cr, max_time, total_infeasible )
"""

def get_aggregated_statistics( lp_results, pareto_results, env_count=0 ):

    pareto_config_count = len( pareto_results[0] )
    if env_count == 0:
        env_count = len( lp_results )

    lp_cr = 0
    lp_max_time = 0
    lp_infeasible = 0

    pareto_cr = [ 0 for _ in range( pareto_config_count ) ]
    pareto_infeasible = [ 0 for _ in range( pareto_config_count ) ]

    for i in range( len( lp_results ) ):
        rew, time, infeasible = lp_results[i]

        lp_cr += rew / env_count
        lp_max_time = max( lp_max_time, time )
        lp_infeasible += infeasible

        for j in range( pareto_config_count ):
            rew, penalty, infeasible = pareto_results[i][j]

            pareto_cr[j] += rew / env_count
            pareto_infeasible[j] += infeasible


    result = [ (lp_cr, lp_max_time, lp_infeasible) ]
    for i in range( pareto_config_count ):
        # 0 for max time pareto
        result.append( ( pareto_cr[i], 0, pareto_infeasible[i] ) )

    return result

# process line of pareto results into csv format
def process_pareto_line( time_limits, pareto_line ):
    line = ""

    for i in range( len(pareto_line) ):
        rew, rew_std, penalty, pen_std, feasible, steptime = pareto_line[i]
        line += f"{feasible};{rew:.2f};{rew_std:.2f};{penalty:.2f};{pen_std:.2f};{steptime:.2f};"

    return line


def eval_solvers(agent_list, time_limits = [ 5 ], pareto_repetitions=100):

    for agent in agent_list:
        with open(f"/work/rats/pyrats/results_{agent.__name__}.csv", 'w') as pfile:
            legend_str = "Benchmark;c;"
            for i in range( len(time_limits)):
                t = time_limits[i]
                legend_str += f"limit_{t}_feasible;timestep_{t}_rew;timestep_{t}_rew_std;timestep_{t}_penalty;timestep_{t}_penalty_std;timestep_{t}_steptime;"

            legend_str = legend_str[:-1] + "\n"
            pfile.write(legend_str)

    with open("/work/rats/pyrats/results.csv", 'w') as file:
        legend_str = "Benchmark;c;feasible;lp_reward; lp_time\n"
        file.write(legend_str)

        index = 0
        
        res_lp = eval_lp()

        # count feasible environments (for LP)
        feasible_count = 0

        # write out results for LP
        for name in filenames:
            for c in c_s:
                for p1 in p_slides:
                    for p2 in p_traps:
                        # env legend
                        line = f"{name}_c{c}_slide{p1}_trap{p2};"

                        # lp legend
                        line += f"{c};{res_lp[index][2]};{res_lp[index][0]:.2f};{res_lp[index][1]:.2f}\n"
                        file.write(line)

                        if res_lp[index][2]:
                            feasible_count += 1

                        index += 1
        file.flush()

        # run benchmarks for pareto, write out results in eval_agents function
        res_pareto = eval_agents(agent_list, time_limits, pareto_repetitions)

        # get cr and stuff
        stats = get_aggregated_statistics( res_lp, res_pareto, feasible_count )

        lp_cr, lp_time, lp_feas = stats[0]

        line = f"LP CR - {lp_cr:.2f}, max time - {lp_time:.2f}, total feasible - {lp_feas}\n"

        for i in range( 1, len(stats) ) :
            cr, _, feas = stats[i]
            line += f"Pareto_t={time_limits[i-1]} CR - {cr:.2f}, total feasible - {feas}\n"

        # write stats
        file.write(line)


eval_solvers(agent_list, time_limits, 100)

ray.shutdown()
