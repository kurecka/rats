import numpy as np
import argparse
from sys import argv
from utils import map2str, is_color_connected
from typing import Dict
from itertools import product


def draw_grid(map: np.array, color: int, stride_x: int = 1, stride_y: int = 1, offset_x: int = 1, offset_y: int = 1) -> None:
    for i in range(offset_x, map.shape[0], stride_x):
        map[i, :] = color
    for i in range(offset_y, map.shape[1], stride_y):
        map[:, i] = color


def prune_color(map, color, select_from=None, stay_connected=True) -> None:
    """
        Remove a random cell of the given color, so that the color remains connected.
        If select_from is given, the cell is selected from the given set of cells.

        Args:
            map: the map (np.array[int])
            color: the color to prune (int)
            select_from: the set of cells to select from (np.array[bool])
            stay_connected: whether the color should remain connected (bool)
    """
    new_map = map.copy()
    
    indices = (new_map == color).nonzero()
    if select_from is not None:
        indices = ((new_map == color) & select_from).nonzero()
    if len(indices[0]) == 0:
        return

    rand_index = np.random.randint(len(indices[0]))
    x, y = indices[0][rand_index], indices[1][rand_index]

    new_map[x, y] = 0
    if not stay_connected or is_color_connected(new_map, color):
        map[x, y] = 0


def replace_color(map: np.array, color_from: int, color_to: int, k: int = 1) -> None:
    """
        Replace up to k random cells of color_from with color_to.
    """
    for _ in range(k):
        indices = (map == color_from).nonzero()
        rand_index = np.random.randint(len(indices[0]))
        x, y = indices[0][rand_index], indices[1][rand_index]
        map[x, y] = color_to


def transform_color(map: np.array, color_from: int, color_to_dist: Dict[int, float]) -> None:
    """
        Replace all cells of color_from with random colors from color_to_dist.

        Args:
            map: the map (np.array[int])
            color_from: the color to transform (int)
            color_to_dist: the distribution of colors to transform to (Dict[int, float])
        
        Example: transform_color(map, ord('P'), {ord('T'): 0.8, ord('G'): 0.2})
    """
    indices = (map == color_from).nonzero()
    for x, y in zip(indices[0], indices[1]):
        r = np.random.rand()
        sum = 0
        for i, (color_to, prob) in enumerate(color_to_dist.items()):
            sum += prob
            if r <= sum or i == len(color_to_dist) - 1:
                map[x, y] = color_to
                break


def frame_map(map: np.array, color: int = ord('#')) -> np.array:
    """
        Frame the map with the given color.
    """
    new_map = np.full((map.shape[0] + 2, map.shape[1] + 2), color, dtype=np.uint8)
    new_map[1:-1, 1:-1] = map
    return new_map


def generate_layerd_grid(width, height, layers=2, a=0.3, b=0.1, space_prob=0.8, wall_prob=0.5, g=0.3):
    path_color = ord('P')
    map = np.zeros((width, height), dtype=np.uint8)
    for _ in range(layers):
        layer = np.zeros((width, height), dtype=np.uint8)
        stride_x = np.random.randint(2, 4)
        stride_y = np.random.randint(2, 4)
        offset_x = np.random.randint(stride_x)
        offset_y = np.random.randint(stride_y)
        draw_grid(layer, path_color, stride_x=stride_x, stride_y=stride_y, offset_x=offset_x, offset_y=offset_y)

        grid_size = (layer == path_color).sum()
        remove = int(a * grid_size)

        for _ in range(remove):
            prune_color(layer, path_color)

        map = np.maximum(map, layer)

    map_size = (map == path_color).sum()
    remove = map_size - int(b * map_size)
    for _ in range(remove):
        prune_color(map, path_color)
    map_size = (map == path_color).sum()

    replace_color(map, path_color, ord('B'), k=1)
    if g >= 1:
        replace_color(map, path_color, ord('G'), k=int(g))
    else:
        replace_color(map, path_color, ord('G'), int(g * map_size))
    transform_color(map, path_color, {ord('.'): space_prob, ord('T'): 1-space_prob})
    transform_color(map, 0, {ord('#'): wall_prob, ord('T'): 1-wall_prob})

    return map


def generate_tree_in_grid(width, height, stride=3, b=1., space_prob=0.8, wall_prob=0.5, g=0.3):
    path_color = ord('P')
    map = np.full((width, height), path_color)
    for _ in range(int(width * height * b)):
        prune_color(map, path_color)
    
    walls = np.zeros((width, height), dtype=np.uint8)
    offset_x = np.random.randint(stride)
    offset_y = np.random.randint(stride)
    draw_grid(walls, 1, stride_x=stride, stride_y=stride, offset_x=offset_x, offset_y=offset_y)

    map[(map != path_color) & (walls == 1)] = ord('#')

    map_size = (map == path_color).sum()
    replace_color(map, path_color, ord('B'), k=1)
    if g >= 1:
        replace_color(map, path_color, ord('G'), k=int(g))
    else:
        replace_color(map, path_color, ord('G'), int(g * map_size))
    transform_color(map, path_color, {ord('.'): space_prob, ord('T'): 1-space_prob})
    transform_color(map, 0, {ord('#'): wall_prob, ord('T'): 1-wall_prob})

    return map


def parse_args():
    parser = argparse.ArgumentParser(description='Generate a gridworld map.')
    parser.add_argument('width', help='The width of the map.')
    parser.add_argument('height', help='The height of the map.')
    parser.add_argument('--gen', type=str, help='The generator to use.', default='LG', choices=['LG', 'TG'])
    parser.add_argument('-l', help='LG: The number of layers', default=2)
    parser.add_argument('-a', help='LG: The percentage of path cells to remove from each layer', default=0.1)
    parser.add_argument('-b', help='The percentage of path cells to remove from the final map', default=0.1)
    parser.add_argument('-g', help='The percentage (or number) of path cells to replace with goal cells', default=0.3)
    parser.add_argument('-e', help='The probability of a path cell changin to empty cell', default=0.8)
    parser.add_argument('-w', help='The probability of a non-path cell changing to a wall cell', default=0.5)
    parser.add_argument('--stride', help='TG: The stride of the grid', default=2)
    parser.add_argument('--seed', type=int, help='The random seed. Negative for random seed.', default=0)
    parser.add_argument('-n', type=int, help='Repeat the generation n times', default=1)
    args = parser.parse_args()

    all_args = ['width', 'height', 'a', 'b', 'g', 'e', 'w', 'l', 'stride']
    grid_args = []
    non_grid_args = []

    for arg in all_args:
        if isinstance(getattr(args, arg), str):
            val = eval(getattr(args, arg))
        else:
            val = getattr(args, arg)
        args.__setattr__(arg, val)
        if isinstance(val, list):
            grid_args.append(arg)
        else:
            non_grid_args.append(arg)
        
    for arg in args.__dict__:
        if arg not in all_args:
            non_grid_args.append(arg)

    
    grid = list(product(*[getattr(args, arg) for arg in grid_args]))

    return args, grid, grid_args, non_grid_args


def print_instance(instance_order: int, map: np.array, params: Dict, grid_args: list):
    print(f"Instance {instance_order}")
    print("Params:", end=' ')
    print(','.join(f"{key}={val}" for key,val in params.items()))
    print("GridParams:", end=' ')
    print(','.join(f"{key}={val}" for key,val in params.items() if key in grid_args))
    print("Map:")
    print(map2str(frame_map(map)))
    print()


if __name__ == '__main__':
    """
    Generate a gridworld map.

    Sample usage:
        # Generate various 10x10 maps
        python3 generator.py 10 10 -l 2 -a 0.5 -b 0.2 -w 0.7 -g 0.2 -e 0.8
        python3 generator.py 10 10 -l 2 -a 0.8 -b 0.1 -w 0.5 -g 0.5 -e 0.8

        # Generate two 6x6 maps with a single command
        python3 generator.py 6 6 -n 2

        # Generate multiple maps using a paramter grid
        python3 generator.py 6 6 -l 2 -a 0.3 -b 0.2 -w [0.2,0.5,0.8] -g 5 -e [0.3,0.5,0.8] -n 1

        # Generate a map with integral number of gold cells
        python3 generator.py 6 6 -g 5
        # Generate a map with 30% gold cells
        python3 generator.py 6 6 -g 0.3
        
        # Generate a map using the tree-in-grid generator
        python3 generator.py 10 10 --gen TG -g 5 -w 0.2 --stride 1 -b 0.4

        # Use ranom seed
        python3 generator.py 6 6 --seed -1

    """
    print('python3', *argv, end='\n\n')

    args, grid, grid_args, non_grid_args = parse_args()
    if args.seed >= 0:
        np.random.seed(args.seed)

    instance_order = 1

    
    for params in grid:
        all_params = {arg: getattr(args, arg) for arg in non_grid_args}
        for i, arg in enumerate(grid_args):
            all_params[arg] = params[i]

        for i in range(int(args.n)):
            if args.gen == 'LG':
                map = generate_layerd_grid(
                    width=all_params['width'], height=all_params['height'],
                    layers=all_params['l'],
                    a=all_params['a'], b=all_params['b'],
                    space_prob=all_params['e'], wall_prob=all_params['w'], g=all_params['g']
                )
            elif args.gen == 'TG':
                map = generate_tree_in_grid(
                    width=all_params['width'], height=all_params['height'],
                    stride=all_params['stride'], b=all_params['b'],
                    space_prob=all_params['e'], wall_prob=all_params['w'], g=all_params['g']
                )
            else:
                raise NotImplementedError(f"Generator {args.gen} is not implemented.")
            
            print_instance(instance_order, map, all_params, grid_args)
            instance_order += 1
