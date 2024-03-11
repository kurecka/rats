import envs

from rats import Hallway, LP_solver
from utils import set_log_level
import time

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

map4 = """
#####
#B.##
##.T#
##.T#
##G##
#####
"""

e = envs.Hallway(map1, 0.1, 0)
s = e.current_state()
# print(s)
# print(e.outcome_probabilities(s, 2))
lp_solver = LP_solver(e, 0.05)
# lp_solver.change_gammas(0.999999)

start = time.time()
rew = lp_solver.solve()
print("Execution time: ", (time.time() - start) * 1000, "ms")

print(rew)
