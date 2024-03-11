import envs

from rats import Hallway, LP_solver
from utils import set_log_level
import time


set_log_level('trace')

filenames = [
    "final_10",
    "final_11",
    "final_12",
    "final_13",
    "final_14",
    "final_15",
    "final_16",
    "final_17",
    "final_18",
    "final_19",
    "final_1",
    "final_20",
    "final_2",
    "final_3",
    "final_4",
    "final_5",
    "final_6",
    "final_7",
    "final_8",
    "final_9",
]

maps = [
'''##########
#GGGTTBTG#
#GGGTT.TT#
#..GTT.TG#
#T.TTT.T##
#..TTT.T##
#T..T..T##
##########''',

'''#############
#..T........#
#..TTGTGTGTG#
#..TT.......#
#..TTGTGTGTG#
#.B...TTTTTT#
#############''',

'''##########
#GTGT.TGT#
#..TG....#
#T.T.TBTT#
#..TG..TG#
#GT..TG..#
##########''',

'''##########
#..GT...G#
#TB.TTGTT#
#..GT.TGT#
#TG......#
#..GTGT.T#
##########''',

"""##########
#TGTGTGTG#
#BTGTGTG.#
#GT.##TGT#
#TG...TG.#
#GTG.GTG.#
##########""",

"""##########
#GTGTGTGT#
#B....G..#
#GTGT.TGT#
#TGTG...T#
#.G.GT.GT#
##########""",

"""#############
#......#....#
#..T.G.#....#
#..TG.....T.#
#..TTGTGTGTG#
#..B..TTTTTT#
#############""",

"""########
#G..TGG#
#T.B..T#
#GT#TGG#
########""",

"""########
#GTG..B#
#.T.TTT#
#.T.TTT#
#....TG#
########""",

"""#########
#TT.G..T#
#..##.#G#
#...GTTG#
#B..TTTG#
#########""",

 """#######
##GGGG#
#BTTGG#
#..TG.#
#T.#T.#
#T.#..#
#T...T#
#######""",

"""
##########
#GTGT.TGT#
#..TG#...#
#T.#.#BTT#
#.#TG#.TG#
#GT..TG..#
##########""",

"""#######
#BTTTG#
#T.T..#
#GT...#
#######""",

"""########
#GT.TTT#
#.TBTT.#
#TT.TTG#
########
""",

"""########
#G..TGG#
#T.BTTT#
#GT#TGG#
########""",

"""#########
#TT.G..##
#..TTTTG#
#...G.TG#
#B..T.TG#
#########""",

"""######
#..G.#
#G#TT#
#.TTT#
##T..#
#.B#G#
#..#T#
#TTTG#
######""",


"""#########
#GGGTTT##
#GGGTT.B#
#..GTT.T#
#T.TTT.G#
#G.TTT.T#
#T.....T#
#########""",

"""#########
#GGGTTT##
#GGGTT.B#
#..GTT.T#
#T.TTT..#
#..TTT.T#
#T.....T#
#########""",

"""##########
#GGGTTT###
#GGGTT.B##
#..GTT.T##
#T.TTT.TG#
#..TTT.T##
#T..T..TG#
##########""",

]

"""
settings:
    c_0 ∈ C = { 0, 0.1, 0.2, 0.3, 0.4 } ,
    p_slide ∈ S = { 0, 0.2 } , and
    p_trap ∈ R = { 0.1, 0.7 } . """


def eval_lp_config( env, c, slide, trap ):
    e = envs.Hallway(env, trap, slide)
    start = time.time()
    lp_solver = LP_solver(e, c)
    lp_solver.change_gammas( 0.99 )
    rew = lp_solver.solve()
    end_time = ( time.time() - start ) * 1000
    return (rew, end_time)


def eval_lp():
    c_s = [ 0, 0.1, 0.2, 0.3, 0.4 ]
    p_slides = [ 0, 0.2 ]
    p_traps = [ 0.1, 0.7 ]

    results = []

    for env in maps:
        for c in c_s:
            for p1 in p_slides:
                for p2 in p_traps:
                    results.append( eval_lp_config(env, c, p1, p2) )

    cr = 0
    for (rew, t) in results:
        cr += 1/400 * rew
