import agents
import envs

from rats import Hallway, LP_solver, Manhattan
import numpy as np
from utils import set_log_level
import time

set_log_level("trace")

map1 = """
########
#..BT.G#
####T..#
########
"""

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
#..TT.#
#....T#
##TT#T#
#TTTTT#
#.TT..#
#GG..G#
#######
"""

map3 = """
#######
#BTTTG#
#..TT.#
#....T#
##TT#T#
#TTT.T#
#.TT..#
#GG..G#
#######
"""

map_hard = """##########
#TGTGTGTG#
#BTGTGTG.#
#GT.##TGT#
#TG...TG.#
#GTG.GTG.#
##########"""

final_13 = """
##########
#..GT...G#
#TB.TTGTT#
#..GT.TGT#
#TG......#
#..GTGT.T#
##########
"""



# pretty close starting state for easy testing
init_state = '42429690'

# three states next to each other for default orders
targets = ['42455666', '42442977', '596775930']


# periods 
periods = { target : 10 for target in targets }

# higher period for last target
periods[targets[-1]] = 20
print(periods)

# default manhattan data
# reloads = ['42431659','42430367','1061531810','42443056','1061531448','42448735','596775930','42435275','42429690','42446036','42442528','42440966','42431186','42438503','42442977',
# '42440966','1061531802','42455666']

# targets = ['42440465','42445916']
e = envs.Manhattan(1000, targets, periods, init_state, cons_thd=10)

total_rew = 0
total_pen = 0

'''
random walk environment test

for i in range(100000):
    s = e.current_state()
    # print(s)
    # print(e.possible_actions(s))
    # print(e.current_state())
    a = np.random.choice(e.possible_actions(s))
    # print("played", a)
    prev_s = s
    s, r, p, o = e.play_action(a)

    if ( r > 0 ):
        print(prev_s, "into", s)

    total_rew += r
    total_pen += p

e.animate_simulation()
print("OVER", total_rew, total_pen)
print(e.current_state())
print(e.possible_actions(e.current_state()))
'''

for i in range(1):
    # e = envs.InvestorEnv(2, 20)
    h = envs.EnvironmentHandler(e, 500)
    a = agents.RAMCP(
        h,
        max_depth=500, num_sim=1000, sim_time_limit=500, risk_thd=0.3, gamma=0.95,
        exploration_constant=5
    )

    e.reset()
    a.reset()
    while not a.get_handler().is_over():
        a.play()
        s = a.get_handler().get_current_state()
        print("Step:", a.get_handler().get_num_steps(), "State: (", s[0] % e.get_width(), ",", s[0] // e.get_width(), ")", "Reward:", a.get_handler().get_reward(), "Penalty:", a.get_handler().get_penalty())
        # print()
    h = a.get_handler()
    print(h.get_reward())
    r += (h.get_reward() - r) / (i+1)
    sr += h.get_reward()
    p += (h.get_penalty() - p) / (i+1)
    sp += h.get_penalty()
    print(f'{i}: {r} {p}')


