import envs
import agents

import numpy as np
from multiprocessing import Pool


# big configs:
# time: 250ms, 500ms, 750ms, 1000ms, 1500ms, 2000ms, 3000ms, 5000ms - RAMCP only upto 500ms, after that LP is too slow
# d: 1, 5, 10
# num_steps: 500
# gamma: 0.95
# thd: 0, 0.1, 0.2, 0.3, 0.4, 0.5
# 100 runs

# 58x20
# exactly solvable with thd=0, main focus p_trap = 1, p_slide = 0.1
# conf:
# p_trap: 0.6, 1
# p_slide: 0.4, 0.1
map_big1 = """
##########################################################
#B...TTTT.T..GG.G#.......TTTT...............#..........GG#
#...........G....#.................G........#.......TGG..#
#......GG.....G.G#........GGGGG...G.........#.......T....#
#..TT...##########.................#G.......#.......T....#
#.....G.......G..#.......TTTT.G....#........#.......T....#
#.....G.......G.G#.............G...#........#.......T...G#
#.......TTTTT..G.#.................#..G.....#...........G#
#............T...#.T.T.T...........#........#.....GG.....#
#............T..G#.................#........#.....GG...TT#
#............T...#.....T........G..#........#............#
#.................G....T...........#........#T.T.T.TT..G.#
#........TT........G...............#........#............#
#................G.G...T...........#..G.....#.......GGG..#
#......................T..........G#........#............#
#..................................#........######....TTT#
#..................GT..T...#####...T.....................#
#..................G........GGG....T.....#G#G#....GGG....#
#......................G...........T.....#G#G#...TT......#
##########################################################
"""

# 36x18 29 golds
# exactly solvable with thd = p_trap + 3 * (p_slide / 2)
# 2 * p_slide is enough but not for these algorithms
# thd = 0 -> 6 golds
# thd = p_trap -> 12 golds
# thd = p_trap + p_slide / 2 -> 27 golds
# conf:
# p_trap: 0.1, 0.3
# p_slide: 0.4, 0.1, 0
map_big2 = """
####################################
#.........G..G........G....T...G.G.#
#..................................#
#........G.G..........T....T..TTTT.#
#.................GG..T............#
######TTTTT#########..T...GG......G#
#..............#...#..TTTT#####.T..#
#B...G.........#####......######T..#
#####..........#####GT....#####....#
#.....G........#####.#######....T..#
#.....###T#....#####....G.....###.G#
#.....###T#....#####..G##....G###..#
#.....###G#....#######..TTT...T##..#
#.....G.G......T...............#...#
#..............T...T..TTT.TG...T...#
#.....TTT......T...###.............#
#............G.T...G...G.......T.GG#
####################################
"""

# 36x17 35 golds
# p_trap = 0.1 has to be small, p_slide = 0.1 also good to be small
# ~ 15 gold w/ thd = 3 * p_trap
# conf:
# p_trap: 0.1
# p_slide: 0.3, 0.1, 0
map_big3 = """
####################################
#.....................##############
#############T.GGG.T.#G...T....T..T#
#############.TTTTT.T#..T....T.TTGG#
#..GG..######.......##...G..G.T.T..#
#......######...G...##...TT...T.T.T#
#TTTTTT######T.T.T.T##G...T..TT.TGG#
#......TTG..........##.....TT.T.T.T#
#......TT.GG........#........T.TG.T#
#......######......G#T.T...TGG.T.T.#
#.G.G..######.......#.T.........TT.#
#T.T.TT######..G....#G...#.....T.T.#
#.T.T.T...###G......#G...##T.T.T.TT#
#.....T...###TT...G.T...T##.G.G.G..#
#..G..T.G.###..........TT##........#
#B....T...###G........TTT##T#T#G#TT#
####################################
"""

# pseudo-random
# conf:
# p_trap: 0.2, 0.7
# p_slide: 0.1, 0.4
map_big4 = """
##########################################################
#B...TTTT.......G#.....####TT..............G#...........G#
#....T..T........T......T..........G...####..........G...#
#....T........G.GT...#G......GG.............#...G...T....#
#G.TT.TT########.#...#G............#G.......#.T.....TG...#
#..G..G...T...G..#......TT..T.G....#........#.T.....T....#
#.....G.......G.G#...GT........G...#.....G..###.....T....#
#.......T...T..G...................#..G.....T...........G#
#.G....G..G..T...#.T.#.T..G..............TTTT......G.....#
#...T........T..G#...#...TT........T........#..........TT#
#.G.T..GT....T...#...#.T........G..T...G....#............#
#.................G....T..G.T......#........#T.T.T.TT..G.#
###.##...TT...G....T......G.T......#........#............#
#.............G..G.G...T...........#..G.....T.........G..#
#....G....T...T........T....G.....G#........T............#
#.........T...###..................#....G#####TT##....TTT#
#.....G...G........GT..T...TTT##...T.....................#
#.............G....G.........GG....T###......G.....GG....#
#...........TTT........G..................GTT####TT......#
##########################################################
"""

# GPT
# p_trap: 0.2, 0.7
# p_slide: 0.1, 0.4
map_big5 = """
##########################################################
#B.....G.......T..G#....T####T..T..........T...G#...T.T.T#
#.....T..T........T......T..........G...####..........T..#
#....T........G.GT...#G......GG.............#...G...T....#
#G.TT.TT########.#...#G............#G.......#.T.....TG...#
#..G..G...T...G..#......TT..T.G....#........#.T.....T....#
#.....G.......G.G#...GT........G...#.....G..###.....T....#
#.......T...T..G...................#..G.....T...........G#
#.G....G..G..T...#.T.#.T..G..............TTTT......G.....#
#...T........T..G#...#...TT........T........#..........TT#
#.G.T..GT....T...#...#.T........G..T...G....#............#
#.................G....T..G.T......#........#T.T.T.TT..G.#
###.##...TT...G....T......G.T......#........#............#
#.............G..G.G...T...........#..G.....T.........G..#
#....G....T...T........T....G.....G#........T............#
#.........T...###..................#....G#####TT##....TTT#
#.....G...G........GT..T...TTT##...T.....................#
#.............G....G.........GG....T###......G.....GG....#
#...........TTT........G..................GTT####TT......#
##########################################################
"""

maps = [
    map_big1,
    map_big2,
    map_big3,
    map_big4,
    map_big5
]

# settings:
time_limits = [250, 500, 750, 1000, 2000, 3000]
d = [1, 5, 10]
num_steps = 500
gamma = 0.95
thd = [0, 0.3, 0.5]
runs = 100
agent_list = [agents.ParetoUCT, agents.DualUCT, agents.RAMCP]

# map confs:
conf1 = [(0.6, 0.4), (1, 0.1)]
conf2 = [(0.1, 0.4), (0.1, 0.1), (0.1, 0)]
conf3 = [(0.1, 0.3), (0.1, 0.1), (0.1, 0)]
conf4 = [(0.2, 0.1), (0.7, 0.4)]
conf5 = [(0.2, 0.1), (0.7, 0.4)]

confs = [conf1, conf2, conf3, conf4, conf5]

def eval_config( map, agent_type, c, slide, trap, time_limit, exp_const ):
    e = envs.Hallway( map, trap, slide )
    h = envs.EnvironmentHandler(e, num_steps)

    a = agent_type(
        h,
        max_depth=num_steps, num_sim=1000, sim_time_limit=time_limit, risk_thd=c, gamma=gamma,
        exploration_constant=exp_const
    )

    a.reset()

    while not a.get_handler().is_over():
        a.play()

    h = a.get_handler()
    rew = h.get_reward()
    p = h.get_penalty()
    return (rew, p)

def eval_config_parallel(args):
    map, agent_type, c, slide, trap, time_limit, exp_const = args
    return eval_config(map, agent_type, c, slide, trap, time_limit, exp_const)

def process_config( map, agent_type, c, slide, trap, time_limit, exp_const ):
    
    pool = Pool()
    results = pool.map(eval_config_parallel, [(map, agent_type, c, slide, trap, time_limit, exp_const) for _ in range(runs)])
    pool.close()
    pool.join()

    rews, pens = zip(*results)

    return (np.mean(rews), np.mean(pens), np.std(rews, ddof=1), np.std(pens, ddof=1))

def run_eval():

    for i, m in enumerate(maps):
        for trap, slide in confs[i]:
            for c in thd:
                # new file for each map
                with open(f"large_HG_eval/results/results_map_big{i+1}_trap:{trap}_slide:{slide}_thd:{c}.csv", "w") as f:
                    f.write("agent;time_limit;exp_const;mean_reward;mean_penalty;std_reward;std_penalty;feasible;emp_feasible\n")
                    for time_limit in time_limits:
                        for agent_type in agent_list:
                            if agent_type == agents.RAMCP and time_limit > 500:
                                continue
                            for exp_const in d:
                                mean_r, mean_p, std_r, std_p = process_config( m, agent_type, c, slide, trap, time_limit, exp_const )
                                emp_feasible = mean_p <= c
                                feasible = mean_p - std_p * 1.65 <= c
                                f.write(f"{agent_type.__name__};{time_limit};{exp_const};{mean_r};{mean_p};{std_r};{std_p};{feasible};{emp_feasible}\n")

if __name__ == "__main__":
    run_eval()