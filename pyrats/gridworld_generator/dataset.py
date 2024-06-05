from pathlib import Path

def get_random_maps():
    path = 'gridworld_generator/GWRB.txt'
    files = []
    maps = []
    with open(path) as f:
        text = f.read()
        maps_data = text.split('Map')[1:]
        for i, map_data in enumerate(maps_data):
            lines = map_data.split('\n')
            space_prob = float(lines[1].split(':')[-1])
            wall_prob = float(lines[2].split(':')[-1])
            file_name = f'map{i}-{space_prob}-{wall_prob}'
            map = '\n'.join(lines[3:])
            files.append(file_name)
            maps.append(map)

    return files, maps


class RandomGridWorldDataset:
    filenames, maps = get_random_maps()

    @classmethod
    def get_maps(cls):
        return list(zip(cls.filenames, cls.maps))
