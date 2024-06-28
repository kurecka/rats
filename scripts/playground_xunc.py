
from rats import envs
from rats import agents
from rats import utils
import numpy as np
import time

utils.set_log_level("trace")

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
# targets = ['42455666', '42442977', '596775930']
targets1 = ['42440966','1061531802','42455666']

targets2 = ['42443056','1061531448','42448735']
targets3 = ['42446036','42442528','42440966']
targets4 = [ '42440966','1061531802','42455666']


# default manhattan data
reloads = ['42431659','42430367','1061531810',
           '42443056','1061531448','42448735',
           '596775930','42435275','42429690',
            '42446036','42442528','42440966',
            '42431186','42438503','42442977', 
            '42440966','1061531802','42455666']

# targets = ['42440465','42445916']
init_state = "42434894"
targets = ["42443056", "42448735", "42446036", "42438503", "42429690",
	  "42442977", "42431659", "42455666","596775930", "42430367"]

period = 50

capacity = 10000
e = envs.Manhattan(targets, init_state, period, capacity, cons_thd=50.0, radius=1.0)

total_rew = 0
total_pen = 0

"""
for i in range(10):
    s = e.current_state()
    print(i, s)
    if i == 2:
        e.make_checkpoint(0)
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

e.restore_checkpoint(0)
print(e.current_state())
e.reset()

"""
r = 0
p = 0
sr = 0
sp = 0

e.reset()

for i in range(1):
    # e = envs.InvestorEnv(2, 20)
    h = envs.EnvironmentHandler(e, 500)
    e.reset()

    print(e.current_state())
    print(h.get_current_state())

    a = agents.ParetoUCT(
        h,
        max_depth=500, num_sim=1000, sim_time_limit=500, risk_thd=10.0, gamma=0.999,
        exploration_constant=5
    )

    e.reset()
    a.reset()
    print("bfore", a.get_handler().get_current_state())
    i = 0
    while not a.get_handler().is_over():
        print(f"{i}", a.get_handler().get_current_state())
        s = a.get_handler().get_current_state()
        a.play()
        #print("Step:", a.get_handler().get_num_steps(), "State: (", s, ")", "Reward:", a.get_handler().get_reward(), "Penalty:", a.get_handler().get_penalty())
        # print()
    h = a.get_handler()
    print(h.get_reward())
    r += (h.get_reward() - r) / (i+1)
    sr += h.get_reward()
    p += (h.get_penalty() - p) / (i+1)
    sp += h.get_penalty()
    print(f'{i}: {r} {p}')
    print(a.get_graphviz())
    e.animate_simulation(100, "/work/rats/outputs/ahoj.html")


