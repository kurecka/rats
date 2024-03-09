import envs

from rats import Hallway, Orchestrator, LP_solver

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

e = envs.Hallway(map, 1, 0.4)
lp_solver = LP_solver(e, 1.0)
rew = lp_solver.solve()

print(rew)
