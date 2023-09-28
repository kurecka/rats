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

map = """#####
#BTG#
#.T.#
#...#
#####
"""

set_log_level('info')

e = envs.Hallway(map, 0.5)
# e = envs.InvestorEnv(2, 20)
a = agents.ParetoUCT(
    envs.EnvironmentHandler(e),
    max_depth=20, num_sim=40, risk_thd=0.2, gamma=1,
    exploration_constant=1, graphviz_depth=7
)


for i in range(30):
    e.reset()
    a.reset()
    while not e.is_over():
        a.play()


e.reset()
a.reset()

for i in range(3):
    a.play()
    with open(f"../logs/tree_{i}.dot", "w") as f:
        f.write(a.get_graphviz())
