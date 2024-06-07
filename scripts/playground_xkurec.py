import numpy as np

from rats import envs
from rats import agents
from rats.utils import set_log_level, set_rng_seed

# 08 09 10 11 12
# 15 16 17 18 19
# 22 23 24 25 26
# 29 30 31 32 33
# 36 37 38 39 40
# 43 44 45 46 47



map = """
#######
#BTTTG#
#..T..#
#.....#
##TT#.#
#GTTG.#
#..T..#
#######
"""

map_small = """
#######
##GGGG#
#BTTGG#
#..TG.#
#T.#T.#
#T.#..#
#T...T#
#######
"""

map_final7 = """
#########
#GGGTTT##
#GGGTT.B#
#..GTT.T#
#T.TTT..#
#..TTT.T#
#T.....T#
#########
"""

map_final9 = """
##########
#GGGTTBTG#
#GGGTT.TT#
#..GTT.TG#
#T.TTT.T##
#..TTT.T##
#T..T..T##
##########
"""

map2 = """
#######
#BTTTG#
#..T..#
#.....#
##TT#.#
#GTTG.#
#######
"""

map3 = """
#############
GGGG###.....#
GGGGTB#.###.#
GGGG#...#..G#
#############
"""

map5 = """
#####
##.##
##.##
#B.T#
##.##
##.##
##.T#
##G##
#####
"""

map_slide2 = """
#####
#TTT#
#T.T#
#B.T#
##.##
##.##
##.T#
##G##
#####
"""

# ## ## ## ## ##
# ## ## 07 ## ##
# ## ## 12 ## ##
# ## 16 17 18 ##
# ## ## 22 ## ##
# ## ## 27 ## ##
# ## ## 32 33 ##
# ## ## 37 ## ##
# ## ## ## ## ##


map6 = """
#######
#BTTTG#
#T.T..#
#GT...#
#######
"""


map7 = """
########
#G..TGG#
#T.BTTT#
#GT#TGG#
########
"""


map8 = """
#############
#..TTTTTTTTT#
#..TTGTGTGTG#
#..TT.......#
#..TTGTGTGTG#
#.B...TTTTTT#
#############
"""

map10 = """
#############
#..T........#
#..TTGTGTGTG#
#..TT.......#
#..TTGTGTGTG#
#.B...TTTTTT#
#############
"""

map9 = """
#############
#B.TTGTTTTTT#
#..T....TT.G#
#.TT.TT.TT.T#
#....TT....T#
#TGTTTTTTGTT#
#############
"""

map4 = """
##########
#TTT...GT#
#TGGTTT..#
#T.T.TG..#
#TTGGTTT.#
#T..TB...#
##########
"""

map11 = """
###########
#.G.#B#.G.#
#.#.....#.#
#G#T#T#T#G#
#G#######G#
#.#..#..#.#
#.#..#..#.#
#.#..#..#.#
#.#..#..#.#
#G#..#..#G#
#.T#.#.#T.#
#.T#.#.#T.#
#G#..#..#G#
###########
"""

map12 = """
#########
#GG#..G.#
#GG#G#TT#
#T##.TTT#
#TT##T..#
##GTT.#G#
##.#B.#T#
##.#TTTG#
#########
"""


def prep_env():
    # return envs.CCPOMCP_EX1()
    # return envs.CCPOMCP_EX2(4)
    return envs.ContHallway(map10, 0.4, 0.1)

def prep_agent(h):
    # return agents.RAMCP(
    # return agents.DualRAMCP(
    # return agents.DualUCT(
    # return agents.LambdaParetoUCT(
    return agents.ParetoUCT(
        h,
        max_depth=50, num_sim=100, risk_thd=0.5, gamma=0.99,
        exploration_constant=5, graphviz_depth=3,
        # lr=1
    )

def repeat():
    set_rng_seed(3)
    r = 0
    p = 0

    for i in range(50):
        e = prep_env()
        h = envs.EnvironmentHandler(e, 100)
        a = prep_agent(h)
        e.reset()
        a.reset()
        while not a.get_handler().is_over():
            a.play()
        h = a.get_handler()
        r += (h.get_reward() - r) / (i+1)
        p += (h.get_penalty() - p) / (i+1)
        print(f'{i}: r={r} p={p}')


def observe_run():
    set_log_level('debug')
    

    e = prep_env()
    h = envs.EnvironmentHandler(e, 10)
    a = prep_agent(h)
    e.reset()
    a.reset() 

    i = 0
    while not a.get_handler().is_over():
    # for i in range(6):
        h = a.get_handler()
        print(f'{i}: state = {h.get_current_state()}')
        print("reward:", h.get_reward())
        print("steps:", h.get_num_steps())
        print()
        a.play()
        # if i < 6:
        with open(f"../logs/tree.dot", "w") as f:
            f.write(a.get_graphviz())
        # input()
        i+=1

repeat()
# observe_run()
