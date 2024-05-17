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

