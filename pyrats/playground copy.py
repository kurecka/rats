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

r = 0
p = 0
sr = 0
sp = 0

for i in range(1):
    e = envs.Hallway(map, 0.1)
    # e = envs.InvestorEnv(2, 20)
    h = envs.EnvironmentHandler(e, 100)
    a = agents.RAMCP(
        h,
        max_depth=20, num_sim=10000, sim_time_limit=250, risk_thd=0.2, gamma=0.9999,
        exploration_constant=1
    )

    e.reset()
    a.reset()
    while not a.get_handler().is_over():
        a.play()
    h = a.get_handler()
    # print(h.get_reward())
    r += (h.get_reward() - r) / (i+1)
    sr += h.get_reward()
    p += (h.get_penalty() - p) / (i+1)
    sp += h.get_penalty()
    print(f'{i}: {r} {p}')

print(sr/100, sp/100)
# r /= 300
# p /= 300
# print(f"Average reward: {r}")
# print(f"Average penalty: {p}")

# e.reset()
# a.reset()

# i = 0
# while not a.get_handler().is_over():
# # for i in range(10):
#     print(f'{i}: {a.get_handler().get_current_state()}')
#     a.play()
#     with open(f"../logs/tree_{i}.dot", "w") as f:
#         f.write(a.get_graphviz())
#     i+=1
