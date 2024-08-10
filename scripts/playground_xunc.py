
from rats import envs
from manhattan_dataset.manhattan_dataset import ManhattanDataset
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
init_state = "42427762"
targets = "42435261","42456683","42440022","42435250","42430378","1241742563","42457788","42435716","596775946"
period = 100
capacity = 10000


data = ManhattanDataset("manhattan_dataset/MANHATTAN.txt")

for name, env in data.get_maps():
    e = envs.Manhattan(**env, period=period, capacity=capacity, cons_thd=20.0, radius=0.5)
    h = envs.EnvironmentHandler(e, 100)
    e.reset()


    a = agents.ParetoUCT(
        h,
        max_depth=100, num_sim=1000, sim_time_limit=100, risk_thd=3.0, gamma=0.999,
        exploration_constant=5
    )

    e.reset()
    a.reset()
    i = 0
    while not a.get_handler().is_over():
        print(f"{i}", a.get_handler().get_current_state())
        s = a.get_handler().get_current_state()
        a.play()
        #print("Step:", a.get_handler().get_num_steps(), "State: (", s, ")", "Reward:", a.get_handler().get_reward(), "Penalty:", a.get_handler().get_penalty())
        # print()
    e.animate_simulation(duration=5, filename=f"/work/rats/outputs/{name}.html")




total_rew = 0

