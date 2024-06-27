from pathlib import Path


class GridWorldDataset:
    def __init__(self, path='GWRB.txt', base=1):
        self.base = base
        self.names, self.maps = self.parse_maps(path)

    def parse_maps(self, path):
        names = []
        maps = []
        with open(path) as f:
            text = f.read()

        instance_data = text.split('Instance')[1:]
        for i, map_data in enumerate(instance_data):
            metadata, map_data = map_data.split('Map:')

            map = map_data.strip()

            lines = metadata.split('\n')
            params = ''
            for line in lines:
                if 'GridParams:' in line:
                    line = line.replace('GridParams:', '')
                    if line.strip():
                        params = '-' + '-'.join([param.strip() for param in line.split(',') if param.strip()])
                    break
            instance_name = f'map{i+self.base}{params}'

            names.append(instance_name)
            maps.append(map)

        return names, maps

    def get_maps(self):
        return list(zip(self.names, [{'map': map} for map in self.maps]))
