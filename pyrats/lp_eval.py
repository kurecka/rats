import envs
import agents

from rats import Hallway, LP_solver
from utils import set_log_level
import time


filenames = [
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

maps2 = [
"""##########
#TGTGTGTG#
#BTGTGTG.#
#GT.##TGT#
#TG...TG.#
#GTG.GTG.#
##########"""
]

maps = [
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

"""
settings:
    c_0 ∈ C = { 0, 0.1, 0.2, 0.3, 0.4 } ,
    p_slide ∈ S = { 0, 0.2 } , and
    p_trap ∈ R = { 0.1, 0.7 } . """


def eval_pareto_config( env, c, slide, trap ):
    e = envs.Hallway( env, trap, slide )
    h = envs.EnvironmentHandler(e, 100)

    a = agents.ParetoUCT(
        h,
        max_depth=100, num_sim=1000, sim_time_limit=5, risk_thd=c, gamma=0.99,
        exploration_constant=5
    )

    a.reset()

    while not a.get_handler().is_over():
        a.play()

    h = a.get_handler()
    rew = h.get_reward()
    p = h.get_penalty()
    return (rew, p)


def eval_lp_config( env, c, slide, trap ):
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


def eval_pareto(repetitions=100):
    c_s = [ 0, 0.1, 0.2, 0.3, 0.4 ]
    p_slides = [ 0, 0.2 ]
    p_traps = [ 0.1, 0.7 ]

    results = []

    for env in maps:
        for c in c_s:
            for p1 in p_slides:
                for p2 in p_traps:
                    print("\n\nPARETO")
                    print(f"Solving with params: c={c}, p_slide={p1}, p_trap={p2}")
                    print(env)

                    rew, p = 0, 0

                    for i in range(repetitions):
                        print(f"rep {i}")
                        r2, penalty = eval_pareto_config(env, c, p1, p2)
                        rew += r2
                        p += penalty

                    rew /= repetitions
                    p /= repetitions
                    feasible = ( p <= c )
                    results.append((rew, p, feasible))
    cr = 0
    total_infeasible = 0
    for (rew, c, f) in results:

        if f:
            cr += 1/400 * rew
        else:
            total_infeasible += 1

    return results, cr, total_infeasible

def eval_lp():
    c_s = [ 0, 0.1, 0.2, 0.3, 0.4 ]
    p_slides = [ 0, 0.2 ]
    p_traps = [ 0.1, 0.7 ]

    results = []

    for env in maps:
        for c in c_s:
            for p1 in p_slides:
                for p2 in p_traps:
                    print(f"Solving with params: c={c}, p_slide={p1}, p_trap={p2}")
                    print(env)

                    # ignore infeasible environments
                    if (env.count('G') > 10):
                        results.append((0, -1, False))
                    else:
                        results.append( eval_lp_config(env, c, p1, p2) )

    cr = 0
    max_time = 0
    total_infeasible = 0
    for (rew, t, f) in results:
        cr += 1/400 * rew
        max_time = max(max_time, t)

        if not f:
            total_infeasible += 1

    return results, cr, max_time, total_infeasible

def eval_solvers(pareto_repetitions=100):
    res_lp, cr, max_time, total_infeasible = eval_lp()
    res_pareto, cr_pareto, total_infeasible_pareto = eval_pareto(pareto_repetitions)
    print(res_lp)
    print(len(res_lp), len(res_pareto))
    print(res_pareto)

    c_s = [ 0, 0.1, 0.2, 0.3, 0.4 ]
    p_slides = [ 0, 0.2 ]
    p_traps = [ 0.1, 0.7 ]

    with open("results.csv", 'w') as file:
        file.write("Benchmark;feasible;lp_reward; lp_time;pareto_feasible;pareto_rew;pareto_penalty\n")
        index = 0
        for name in filenames:
            for c in c_s:
                for p1 in p_slides:
                    for p2 in p_traps:
                        line = f"{name}_c{c}_slide{p1}_trap{p2};"
                        line += f"{res_lp[index][2]};{res_lp[index][0]:.2f};{res_lp[index][1]:.2f};"
                        line += f"{res_pareto[index][2]};{res_pareto[index][0]:.2f};{res_pareto[index][1]}\n"
                        file.write(line)
                        index += 1

        file.write(f"Infeasible: LP - {total_infeasible}, P - {total_infeasible_pareto}\n")
        file.write(f"CR: LP - {cr:.2f}, P - {cr_pareto:.2f}\n")

eval_solvers(100)
