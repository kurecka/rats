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
#T.T.TT..#
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

set_log_level('debug')

# r = 0
# p = 0

# for i in range(100):
#     e = envs.Hallway(map5, 1, 0.4)
#     h = envs.EnvironmentHandler(e, 20)
#     a = agents.ParetoUCT(
#         h,
#         max_depth=20, num_sim=-1, sim_time_limit=50, risk_thd=0.16, gamma=0.99,
#         exploration_constant=1
#     )

#     e.reset()
#     a.reset()
#     while not a.get_handler().is_over():
#         a.play()
#         # print()
#     h = a.get_handler()
#     print(h.get_reward())
#     r += (h.get_reward() - r) / (i+1)
#     p += (h.get_penalty() - p) / (i+1)
#     print(f'{i}: r={r} p={p}')


e = envs.Hallway(map5, 1, 0.4)
h = envs.EnvironmentHandler(e, 20)
a = agents.ParetoUCT(
    h,
    max_depth=20, num_sim=-1, sim_time_limit=50, risk_thd=0.16, gamma=0.99,
    exploration_constant=1, graphviz_depth=3
)

e.reset()
a.reset()

i = 0
while not a.get_handler().is_over():
# for i in range(5):
    print(f'{i}: {a.get_handler().get_current_state()}')
    a.play()
    with open(f"../logs/tree_{i}.dot", "w") as f:
        f.write(a.get_graphviz())
    i+=1
