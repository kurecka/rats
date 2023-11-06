import numpy as np
from itertools import product
from random import shuffle
from sys import argv


def get_tile_set(char_tile_set):
    tile_set = []
    for line in char_tile_set.splitlines():
        line = line.strip()
        if line: 
            tile_set.append([ord(c) for c in line])
            
    return tile_set


def tile_set_indices(tile_set):
    width = len(tile_set[0])
    height = len(tile_set)
    return list(product(range(1, height-1), range(1, width-1)))


def match_tile(map, x, y, tile_set, i, j):
    for di, dj in product(range(-1, 2), range(-1, 2)):
        if map[x+di, y+dj] != tile_set[i+di][j+dj] and map[x+di, y+dj] != 0:
            return False
        map[x+di, y+dj] = tile_set[i+di][j+dj]
    return True

def map2str(map):
    map[map == 0] = ord(' ')
    return '\n'.join([
        line.astype(np.uint8).tobytes().decode('ascii')
        for line in map
    ])


def wfc(map, x, y, tile_set, depth=0):
    indices = tile_set_indices(tile_set)
    shuffle(indices)

    not_fixed = map == 0

    for i,j in indices:
        if not match_tile(map, x, y, tile_set, i, j):
            map[not_fixed] = 0
            continue
        zero_tiles = np.argwhere(map == 0)
        if len(zero_tiles) == 0:
            return True
        x_, y_ = zero_tiles[0]
        # print(map)
        res = wfc(map, x_, y_, tile_set, depth+1)
        if res:
            return True
        else:
            # reset non-fixed tiles
            map[not_fixed] = 0
    return False


def main(width=8, height=8):
#     char_tile_set = """
# ################
# #..TTT...G.#T.T#
# #T..T...TT.#...#
# #...T.#......G.#
# #..GT......#...#
# #.TTTTTT.#T##TT#
# #...TG...#G#...#
# ################
# """

#     char_tile_set = """
# ###############
# #...TT#...TTT.#
# #.T.#T#G...T..#
# #..G....TTTT.G#
# #..TT#T#...T..#
# #..GTTG....T..#
# #G.#TT#..TTTT.#
# #TTTG..#G.T.T.#
# #..##..#T.TGTG#
# #..#....#T.T..#
# #....#....T...#
# #.T#TTTT####..#
# #TTGTTGTTTG#..#
# #..#...T#.T#..#
# ###############
# """

    char_tile_set = """
################
#.......#......#
#.##.##.##..##.#
##.#..T..#..T..#
#..T..#....T#..#
##TT#..#T..##..#
#.#GT...T..#...#
#.#.T..TTT.#TT##
#.#G#......#G.G#
##TT##.TTT.##T##
#.T..T..GT.. TG#
#.T..T.TTT...T.#
#GT....GT... T.#
################
"""


    tile_set = get_tile_set(char_tile_set)

    map = np.zeros((height, width), dtype=np.int32)
    map[0, :] = ord('#')
    map[-1, :] = ord('#')
    map[:, 0] = ord('#')
    map[:, -1] = ord('#')

    if wfc(map, 1, 1, tile_set):
        space_indices = np.argwhere(map == ord('.'))
        b = 10000000 % len(space_indices)
        x, y = space_indices[b]
        map[x, y] = ord('B')
        print(map2str(map))
    else:
        print("No solution found")
    


if __name__ == '__main__':
    if len(argv) == 1:
        main()
    else:
        main(int(argv[1]), int(argv[2]))
