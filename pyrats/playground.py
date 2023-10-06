import numpy as np

import envs
import agents
from utils import set_log_level

# map = """
# #############
# #B...T#.G...#
# #.##..#TTT..#
# #..G#.......#
# #############
# """

# map = """
# #######
# #BTTTG#
# #..T..#
# #.....#
# #######
# """

# set_log_level('debug')

# e = envs.Hallway(map, 0.1)
e = envs.InvestorEnv(2, 20)
h = envs.EnvironmentHandler(e)
a = agents.DualUCT(
    h,
    max_depth=80, num_sim=8, risk_thd=0.2, gamma=1,
    exploration_constant=0.6, graphviz_depth=7
)

# for i in range(3):
#     e.reset()
#     a.reset()
#     while not a.get_handler().is_over():
#         a.play()


e.reset()
a.reset()

for i in range(3):
    a.play()
    with open(f"../logs/tree_{i}.dot", "w") as f:
        f.write(a.get_graphviz())
