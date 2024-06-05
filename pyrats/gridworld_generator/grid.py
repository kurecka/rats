import numpy as np
from sys import argv
from wfc import map2str, is_connected

def draw_grid(map, stride_x, stride_y, offset_x, offest_y, c):
    for i in range(offset_x, map.shape[0], stride_x):
        map[i, :] = c
    for i in range(offest_y, map.shape[1], stride_y):
        map[:, i] = c


def maybe_prune(map, c):
    new_map = map.copy()
    indices = (new_map == ord('P')).nonzero()
    rand_index = np.random.randint(len(indices[0]))
    x, y = indices[0][rand_index], indices[1][rand_index]
    new_map[x, y] = 0
    if is_connected(new_map):
        map[x, y] = 0

def select_beginning(map, c):
    indices = (map == c).nonzero()
    rand_index = np.random.randint(len(indices[0]))
    x, y = indices[0][rand_index], indices[1][rand_index]
    map[x, y] = ord('B')

def select_gold(map, c, count=5, gold=ord('G')):
    for _ in range(count):
        indices = (map == c).nonzero()
        rand_index = np.random.randint(len(indices[0]))
        x, y = indices[0][rand_index], indices[1][rand_index]
        map[x, y] = gold

def transform(map, c, a, pa, b):
    indices = (map == c).nonzero()
    rand_index = np.random.randint(len(indices[0]))
    for x, y in zip(indices[0], indices[1]):
        if np.random.rand() < pa:
            map[x, y] = a
        else:
            map[x, y] = b

def generate(width, height, layers=2, color=ord('P'), space_prob=0.8, wall_prob=0.5):
    map = np.zeros((width, height), dtype=np.uint8)
    for l in range(layers):
        layer = np.zeros((width, height), dtype=np.uint8)
        stride_x = np.random.randint(2, 4)
        stride_y = np.random.randint(2, 4)
        offset_x = np.random.randint(stride_x)
        offset_y = np.random.randint(stride_y)
        draw_grid(map, stride_x, stride_y, offset_x, offset_y, color)

        for _ in range(5):
            maybe_prune(map, color)

        map = np.maximum(map, layer)

    for _ in range(5):
        maybe_prune(map, color)

    select_beginning(map, color)
    select_gold(map, color, 5, ord('G'))
    transform(map, color, ord('.'), space_prob, ord('T'))
    transform(map, 0, ord('#'), wall_prob, ord('T'))

    return map

def print_random_map(width=10, height=10, space_prob=0.8, wall_prob=0.5):
    map = np.zeros((width+2, height+2), dtype=np.uint8)
    map[:] = ord('#')
    map[1:-1, 1:-1] = generate(width, height, space_prob=space_prob, wall_prob=wall_prob)
    print(map2str(map))
    print()

if __name__ == '__main__':
    # set the seed for reproducibility
    np.random.seed(42)

    space_prob = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    wall_prob = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    i = 0

    for s in space_prob:
        for w in wall_prob:
            for _ in range(int(argv[3])):
                print("Map", i)
                print("Space prob:", s)
                print("Wall prob:", w)
                print_random_map(int(argv[1]), int(argv[2]), s, w)
                i += 1
