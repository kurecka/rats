import numpy as np
from itertools import product
from random import shuffle
from sys import argv
from utils import is_color_connected, map2str


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
    

char_tile_set = """
############################
#PP...........#...PPP..PPP.#
#.PPPPP..PP...#.PPP.....PP.#
#.....P..P....#...PPPPPPP..#
#..P..PPPP....#...P...PP...#
#PPPPPP..P..PP#PPPPP...P...#
#...P....P...P#...P........#
#...P....PPPPP#...PPPPPPPPP#
#..PPPPPPPP...#...PP..P....#
#..P.....PPPPP#PPPP...PPPP.#
#..PPP........#..........PP#
############################
"""

def main(width=8, height=8):
    tile_set = get_tile_set(char_tile_set)

    map = np.zeros((height, width), dtype=np.int32)
    map[0, :] = ord('#')
    map[-1, :] = ord('#')
    map[:, 0] = ord('#')
    map[:, -1] = ord('#')

    if wfc(map, width//2, height//2, tile_set):
        print(map2str(map))
        print("Connected:", is_color_connected(map, color=ord('P')))
    else:
        print("No solution found")



if __name__ == '__main__':
    if len(argv) == 1:
        main()
    else:
        main(int(argv[1]), int(argv[2]))
