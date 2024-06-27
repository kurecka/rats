import numpy as np

def map2str(map):
    map[map == 0] = ord(' ')
    return '\n'.join([
        line.astype(np.uint8).tobytes().decode('ascii')
        for line in map
    ])


def is_color_connected(map, color):
    mask_visited = np.zeros_like(map, dtype=bool)
    color_count = (map == color).sum()
    if color_count == 0:
        return True

    x, y = np.argwhere(map == color)[0]
    mask_visited[x, y] = True
    stack = [(x, y)]
    
    stack_count = 0
    while stack:
        x, y = stack.pop()
        stack_count += 1
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            height, width = map.shape
            x_, y_ = x+dx, y+dy
            if 0 <= x_ < height and 0 <= y_ < width and map[x_, y_] == color and not mask_visited[x_, y_]:
                mask_visited[x_, y_] = True
                stack.append((x_, y_))
    return color_count == stack_count