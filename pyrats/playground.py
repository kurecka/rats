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

set_log_level('debug')


e = envs.Hallway(map, 0.1)
# e = envs.InvestorEnv(2, 20)
h = envs.EnvironmentHandler(e, 100)
a = agents.ParetoUCT(
    h,
    max_depth=100, num_sim=-1, sim_time_limit=800, risk_thd=0.2, gamma=0.999,
    exploration_constant=5, graphviz_depth=3
)

# e.reset()
# a.reset()
# while not a.get_handler().is_over():
#     a.play()


# e.reset()
# a.reset()

# i = 0
# while not a.get_handler().is_over():
for i in range(30):
    print(f'{i}: {a.get_handler().get_current_state()}')
    a.play()
    with open(f"../logs/tree_{i}.dot", "w") as f:
        f.write(a.get_graphviz())
    # i+=1
