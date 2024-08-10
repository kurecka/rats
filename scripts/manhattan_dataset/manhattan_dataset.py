from manhattan_dataset.manhattan_visualize import plot_manhattan_map

class ManhattanDataset:
    def __init__(self, path='manhattan_dataset/MANHATTAN.txt'):
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
        return list(zip(self.names, [{"targets" : map['targets'], "init_state" : map['init_state']} for map in self.maps]))

    def visualize_maps(self, mapfile="../rats/manhattan_res/NYC.graphml"):
        for i in range(len(self.maps)):
            map = self.maps[i]
            name = self.names[i]
            targets = map['targets']
            init_state = map['init_state']
            plot_manhattan_map(mapfile, "manhattan_dataset/" + name + '.html', init_state, targets)
