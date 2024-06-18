class ManhattanDataset:
    def __init__(self, path='manhattan_dataset/test_dataset.txt'):
        self.names, self.maps = self.parse_maps(path)

    def parse_maps(self, path):
        names = []
        maps = []
        with open(path) as f:
            text = f.read()

        instance_data = text.split('Instance')[1:]
        for i, map_data in enumerate(instance_data):
            lines = map_data.split('\n')

            map = dict()

            for line in lines:
                data = line.split(' ')

                key = data[0].lower()

                if key == 'name':
                    names.append(data[-1])

                if key == 'init_state':
                    map['init_state'] = data[-1]

                if key == 'targets':
                    targets = data[-1].split(',')
                    map['targets'] = targets

            maps.append(map)

        return names, maps


    def get_maps(self):
        return list(zip(self.names, [{'map': map} for map in self.maps]))
