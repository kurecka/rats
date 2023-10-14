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

set_log_level('info')


e = envs.Hallway(map, 0.2)
# e = envs.InvestorEnv(2, 20)
h = envs.EnvironmentHandler(e, 30)
a = agents.ParetoUCT(
    h,
    max_depth=100, num_sim=-1, sim_time_limit=200, risk_thd=0.5, gamma=0.995,
    exploration_constant=5, risk_exploration_ratio=0.05, graphviz_depth=3
)

r = 0
p = 0
wr = 0
wp = 0

for i in range(10000):
    print(i);
    e.reset()
    a.reset()
    t=0
    while not a.get_handler().is_over():
        print(f'{t}: {a.get_handler().get_current_state()}')
        a.play()
        with open(f"../logs/tree_{t}.dot", "w") as f:
            f.write(a.get_graphviz())
        t+=1
    a.train()

    h = a.get_handler()
    r += h.get_reward()
    wr += (h.get_reward() - wr) * 0.05
    p += h.get_penalty()
    wp += (h.get_penalty() - wp) * 0.05
    print()
    print(f'reward: {h.get_reward()}, penalty: {h.get_penalty()}')
    print(f'mean reward: {r / (i+1)}, mean penalty: {p / (i+1)}')
    print(f'window reward: {wr}, window penalty: {wp}')
    print(f"state curve 8: {a.get_state_curve((8, 7))}")
    print(f"state curve (9, 7): {a.get_state_curve((9, 7))}")
    print(f"state curve (15, 7): {a.get_state_curve((15, 7))}")
    print(f"state curve (17, 7): {a.get_state_curve((17, 7))}")
    print(f"state curve (10, 7): {a.get_state_curve((10, 7))}")
    print(f"state curve (23, 7): {a.get_state_curve((23, 7))}")
    print(f"state curve (23, 6): {a.get_state_curve((23, 6))}")
    print(f"state curve (24, 7): {a.get_state_curve((24, 7))}")
    print(f"state curve (24, 6): {a.get_state_curve((24, 6))}")
    print(f"state curve (12,6): {a.get_state_curve((12, 6))}")
    # input()


# e.reset()
# a.reset()

# i = 0
# while not a.get_handler().is_over():
# # for i in range(30):
#     print(f'{i}: {a.get_handler().get_current_state()}')
#     a.play()
#     with open(f"../logs/tree_{i}.dot", "w") as f:
#         f.write(a.get_graphviz())
#     i+=1
