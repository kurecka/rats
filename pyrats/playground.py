import numpy as np

import envs
import agents
from utils import set_log_level

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
#BTTTG#
#..T..#
#.....#
#######
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

map4 = """
##########
#TT.....T#
#T.GTTT..#
#T.T.TTG.#
#T..GTTT.#
#T..TB...#
##########
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


# set_log_level('debug')

r = 0
p = 0

for i in range(1000):
    # e = envs.Hallway(map5, 1, 0.4)
    e = envs.CCPOMCP_EX1()
    h = envs.EnvironmentHandler(e, 20)
    a = agents.ParetoUCT(
        h,
        max_depth=20, num_sim=-1, sim_time_limit=5, risk_thd=0.16, gamma=0.99,
        exploration_constant=1
    )

    e.reset()
    a.reset()
    while not a.get_handler().is_over():
        a.play()
        # print()
    h = a.get_handler()
    print(h.get_reward())
    r += (h.get_reward() - r) / (i+1)
    p += (h.get_penalty() - p) / (i+1)
    print(f'{i}: r={r} p={p}')


# # e = envs.Hallway(map, 0.1, 0)
# e = envs.CCPOMCP_EX1()
# h = envs.EnvironmentHandler(e, 100)
# # a = agents.DualUCT(
# # a = agents.DualRAMCP(
# a = agents.DualUCT(
#     h,
#     max_depth=70, num_sim=0, sim_time_limit=10, risk_thd=0.5, gamma=0.95,
#     exploration_constant=25, graphviz_depth=3
# )

# e.reset()
# a.reset() 

# i = 0
# while not a.get_handler().is_over():
# # for i in range(6):
#     print(f'{i}: {a.get_handler().get_current_state()}')
#     a.play()
#     # if i < 6:
#     with open(f"../logs/tree.dot", "w") as f:
#         f.write(a.get_graphviz())
#     input()
#     i+=1
