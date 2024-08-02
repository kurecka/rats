import copy
from fimdpenv import AEVEnv
import folium
from itertools import islice
from math import asin, cos, sqrt, pi, inf

import numpy as np
import networkx as nx
import time

# Calculates distance in km from latitude and longitude of two points using
# the Haversine formula,
# see: https://stackoverflow.com/questions/27928/calculate-distance-between-two-latitude-longitude-points-haversine-formula
def distance_from_gps(long1, lat1, long2, lat2):
    R = 6371
    p = pi / 180
    a = 0.5 - cos((lat2-lat1)*p)/2 + cos(lat1*p) * cos(lat2*p) * (1-cos((long2-long1)*p))/2
    return 2 * R * asin(sqrt(a))


class ManhattanEnv:

    """
        Manhattan MDP environment simulator built on top of FiMDPEnv environment AEVEnv.
        The agent moves through an area of Manhattan from 42nd to 116th street and receives orders from
        various customers. The goal of the agent is to maximize finished orders (rewards) while avoiding
        delayed deliveries (penalty).

        targets - labels of reachability targets / states of the mdp that provide orders for the agent

        init_state - initial state of the agent

        period - the cooldown on each customer order.

        capacity - the consumption capacity of the agent. Picking a direction results in a stochastic energy consumption
                   taken from the original AEVEnv benchmark. See https://arxiv.org/abs/2005.07227 - section 7.1.
                   This consumption is then subtracted from the agents capacity as well as from the periods.
                   Once the agent reaches an energy level of zero, the
                   environment terminates. Passing capacity = 0 results in a
                   non-terminating environment

        cons_thd - the delay threshold on the orders, whenever an order is
                    accepted and the agent accumulates a penalty of >= cons_thd
                    in the process of finishing the order, unit penalty is
                    received

        radius - the distance limit on accepting orders in kilometers
    """
    def __init__(self, targets, init_state, period, capacity, cons_thd=10.0, radius=2.0):
        self.env = AEVEnv.AEVEnv(capacity, targets, [], init_state,
                datafile="/work/rats/rats/manhattan_res/NYC.json",
                mapfile ="/work/rats/rats/manhattan_res/NYC.graphml")

        # maps checkpoint ids to state and history information
        self.checkpoints = dict()

        self.init_state = init_state

        self.position = self.init_state

        # last known gps position, used when evaluating distance from targets
        self.last_gps = None

        self.energy = capacity
        self.capacity = capacity

        self.cons_thd = cons_thd

        self.targets = targets
        self.period = period

        self.radius = radius

        # multiplicative factor to period of order after completing it
        self.cooldown = 5

        # decision node -> special part of state (flag) - signals that orders
        # can be accepted at this step
        self.decision_node = False

        # ctr for each target, -1 is active target (accepted order)
        self.state_of_targets = { t : period for t in self.targets }

        # ctr for each target, how much time has passed since accepting the
        # order, once the delay exceeds cons_thd, receive unit penalty
        self.delay_of_targets = { t : 0 for t in self.targets }

        # history of positions, used to animate the trajectory of the agent
        self.history = []

        # maps indices to lengths of the history, used to remove simulated actions/states from history
        # when restoring checkpoints
        self.histories = {}

        # graph with latitude and longitude information, used to evaluate
        # distance from targets and plot an animation of the trajectory.
        self.G = nx.MultiDiGraph(nx.read_graphml(self.env.mapfile))
        self.geo_data = self.G.nodes(data=True)


    """
        helper methods

        used mainly for converting between state representations of AEVEnv
        and this class
    """
    # translate state names (json labels) to cmdp states used in AEVEnv
    def name_to_state(self, name):
        return self.env.consmdp.names_dict[name]

    # translate indices in cmdp to json labels
    def state_to_name(self, state):
        return self.env.state_to_name[state]

    def target_active(self, target):
        return self.state_of_targets[target] == -1

    def can_accept_target(self, target):
        return self.state_of_targets[target] == 0

    """
        env interface methods
    """
    def name(self):
        return "ManhattanEnv"

    def num_actions(self):
        return len(self.possible_actions())

    """
        get the number of actions for the current state
        if in decision node, return indices to active targets

        -1 signals refusing any available orders.
    """
    def get_actions_for_state(self, name):
        # able to accept orders
        if ( self.decision_node ):
            return [ -1 ] + [ i for i, t in enumerate(self.targets) if self.can_accept_target(t) ]

        state_id = self.name_to_state(name)
        action_count = len(self.env.consmdp.actions_for_state(state_id))
        return list(range(action_count))

    def current_state(self):
        return self.position, self.state_of_targets, self.decision_node

    def possible_actions(self, state = None):
        # if called with default init state tuple (from c++) or None,
        # get actions for current position
        if state is None or state[0] == '':
            state_name = self.position
        else:
            state_name = state[0]
        return self.get_actions_for_state(state_name)


    def get_action(self, idx):
        return self.possible_actions()[idx]

    # TODO: not implemented
    def outcome_probabilities(self, name, action):
        pass

    """
         decreases counters on orders by _cons_, any orders that reach zero and
         are not sufficiently close (<= self.radius), are refreshed to the
         period.

         returns true if a customer is sufficiently close and his order can be accepted,
         this moves the environment into a decision_node dummy state next step.
    """
    def decrease_ctrs(self, cons):
        orders_available = False

        for t in self.targets:
            if self.target_active(t):
                continue

            self.state_of_targets[t] -= cons
            if self.state_of_targets[t] <= 0:
                dist = self.distance_to_target(t)
                new_value = self.period

                if dist <= self.radius:
                    new_value = 0
                    orders_available = True

                self.state_of_targets[t] = new_value

        return orders_available


    """
        Reloads all orders that have not been accepted (have state=0) to
        period.
    """
    def reload_ctrs(self):
        for t in self.targets:
            if self.can_accept_target(t):
                self.state_of_targets[t] = self.period


    """
        Increases the delay on all active orders by cons, returns the number of
        orders that hit cons_thd in this call (i.e. the number of delayed
        orders that were penalized)
    """
    def increase_delay(self, cons):
        delayed_orders = 0
        for t in self.targets:
            # active order that was not delayed yet
            if self.target_active(t) and self.delay_of_targets[t] < self.cons_thd:
                self.delay_of_targets[t] += cons

                # if delay is > threshold, cap it and receive penalty
                if ( self.delay_of_targets[t] >= self.cons_thd ):
                    delayed_orders += 1
                    self.delay_of_targets[t] = self.cons_thd
        return delayed_orders


    def distance_to_target(self, target):
        # some states do not have longitude and latitude information in the
        # graph file, need to work around this
        if self.position not in self.geo_data:

            # check if last known gps position is initialized
            if self.last_gps is None:
                return inf

            # use last known gps position
            else:
                state_data = self.geo_data[self.last_gps]

        # use the current position
        else:
            state_data = self.geo_data[self.position]
            self.last_gps = self.position

        state_lat = float(state_data['lat'])
        state_lon = float(state_data['lon'])

        target_data = self.geo_data[target]
        target_lat = float(target_data['lat'])
        target_lon = float(target_data['lon'])

        return distance_from_gps(state_lon, state_lat, target_lon, target_lat)


    def play_action(self, action):
        # if in decision node, handle (potential) orders
        if self.decision_node:
            reward = 0
            self.decision_node = False

            # action is an index to targets
            if action != -1:

                # incurs small negative reward for accepting order
                # reward = -1e-4

                # accept target
                self.state_of_targets[self.targets[action]] = -1
            self.reload_ctrs()

            return (self.position, self.state_of_targets, self.decision_node), reward, 0, self.is_over()

        # otherwise, proceed by moving in the underlying cmdp, recording
        # reward/penalty and adjusting periods of orders, record the state into
        # history as well as information about active orders
        self.history.append( (self.position, [self.target_active(t) for t in self.targets ] ) )

        # get linked list of actions from cmdp
        state_id = self.name_to_state(self.position)
        action_iterator = self.env.consmdp.actions_for_state(state_id)

        # get the action from iterator
        action_data = next(islice(action_iterator, action, None))

        # get successor
        next_state = np.random.choice(list(action_data.distr.keys()),
                                      p=list(action_data.distr.values()))

        # skip dummy state representing stochastic consumption, record it
        action_iterator = self.env.consmdp.actions_for_state(next_state)
        action_data = next(islice(action_iterator, 0, None))

        next_state = next(iter(action_data.distr))
        self.position = self.state_to_name(next_state)

        # if limited energy budget is set, decrease the current energy
        if self.capacity > 0:
            self.energy -= action_data.cons

        # decrease counters for targets
        self.decision_node = self.decrease_ctrs(action_data.cons)

        # reward if order is delivered
        reward = 0
        if (self.position in self.targets) and self.target_active(self.position):
            reward = 1
            self.state_of_targets[self.position] = self.period * self.cooldown

        # add delay (consumption of last action) to orders that are unfinished
        # receive penalty for each delayed order
        penalty = self.increase_delay(action_data.cons)

        return (self.position, self.state_of_targets, self.decision_node), reward, penalty, self.is_over()


    # if capacity is ==0 the environment does not terminate
    def is_over(self):
        return self.capacity > 0 and self.energy <= 0

    def make_checkpoint(self, checkpoint_id):
        counter_copy = copy.deepcopy(self.state_of_targets)
        delay_copy = copy.deepcopy(self.delay_of_targets)

        self.checkpoints[checkpoint_id] = (self.position, self.last_gps, self.energy, delay_copy, counter_copy, self.decision_node)
        self.histories[checkpoint_id] = len(self.history)

    def restore_checkpoint(self, checkpoint_id):
        self.position, self.last_gps, self.energy, self.delay_of_targets, self.state_of_targets, self.decision_node = self.checkpoints[checkpoint_id]
        history_id = self.histories[checkpoint_id]
        self.history = self.history[:history_id]

    def reset(self):
        self.energy = self.capacity
        self.position = self.init_state
        self.state_of_targets = { t : self.period for t in self.targets }
        self.delay_of_targets = { t : 0 for t in self.targets }
        self.decision_node = False
        self.history = []

    """
        Visualizes the positions in self.history on the map of manhattan.
        Taken directly from AEVEnv.

        Duration signals the maximum number of transitions visible on the map,
        i.e. 20 means that only the 20 most recent transitions will be shown.
        0 -> transitions should persist
    """
    def animate_simulation(self, duration=0, filename="map.html"):
        targets = self.targets

        init_state = self.init_state

        def is_int(s):
            try:
                int(s)
                return True
            except ValueError:
                return False


        for _, data in self.geo_data:
            data['lat'] = float(data['lat'])
            data['lon'] = float(data['lon'])

        # remove targets and trajectory states that are not present
        # in the geo json data, populate trajectory list
        for target in targets:
            if target not in list(self.G.nodes):
                targets.remove(target)

        trajectory = []

        for position, orders in self.history:
            if position not in list(self.G.nodes):
                pass
            else:
                trajectory.append((position, orders))

        # create map and fit its bounds to min/max longitude and latitude of visited states
        global_lat = []
        global_lon = []
        for node in self.G.nodes.data():
            point = [node[1]['lat'], node[1]['lon']]
            global_lat.append(point[0])
            global_lon.append(point[1])

        min_point = [min(global_lat), min(global_lon)]
        max_point =[max(global_lat), max(global_lon)]
        m = folium.Map(zoom_start=1)

        # Add Stadia Stamen Terrain tiles to the map
        folium.TileLayer(
            tiles='https://{s}.tile.openstreetmap.fr/hot/{z}/{x}/{y}.png',
            attr='copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors, Tiles style by <a href="https://www.hotosm.org/" target="_blank">Humanitarian OpenStreetMap Team</a> hosted by <a href="https://openstreetmap.fr/" target="_blank">OpenStreetMap France</a>.',
            name='OpenStreetMap.HOT'
        ).add_to(m)

        folium.LayerControl().add_to(m)

        # add initial state as a permanent marker on the map
        folium.CircleMarker(location=[self.G.nodes[init_state]['lat'], self.G.nodes[init_state]['lon']],
                        radius= 5,
                        popup = 'initial state',
                        color='black',
                        fill_color = 'black',
                        fill_opacity=1,
                        fill=True).add_to(m)


        # animate trajectory of the agent
        t = time.time()
        path = list(zip(trajectory[:-1], trajectory[1:]))

        # contains geo json data pertaining to the transitions (coords of both pts, timestamp and color of edge)
        lines = []

        # contains geo json data pertaining to the orders - their positions, status, etc.
        order_data = []

        # time difference in seconds between two lines (moves of the agent)
        # used by folium to control when the line appears/disappears
        t_edge = 1

        for pair in path:
            data1, data2 = pair

            pos1, orders1 = data1
            pos2, orders2 = data2


            # prepare metadata for each transition of the agent
            # for both points on the transition get longitude and latitude from json and timestamp them
            lines.append(dict({'coordinates':
                [[self.G.nodes[pos1]['lon'], self.G.nodes[pos1]['lat']],
                [self.G.nodes[pos2]['lon'], self.G.nodes[pos2]['lat']]],
                'dates': [time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(t)),
                           time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(t+t_edge))],
                           'color':'black'}))

            # add metadata for each target, change colors based on its status
            i = 0
            for target in targets:
                color = 'red'

                # order accepted in next step
                if orders2[i]:
                    color = 'yellow'

                # order finished in next step
                elif orders1[i]:
                    color = 'green'

                order_data.append(dict({'coordinates':
                                       [self.G.nodes[target]['lon'], self.G.nodes[target]['lat']],
                                   'dates':
                                        [time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(t+t_edge))],
                               'color': color }))
                i += 1

            t = t+t_edge

        # geojson feature for each of the agent transitions
        features = [{'type': 'Feature',
                     'geometry': {
                                'type': 'LineString',
                                'coordinates': line['coordinates'],
                                 },
                     'properties': {'times': line['dates'],
                                    'style': {'color': line['color'] }
                                    }
                     }
                for line in lines]

        # geojson features for all orders
        positions = [{
                'type': 'Feature',
                'geometry': {
                            'type':'Point',
                            'coordinates': position['coordinates']
                            },
                'properties': {
                    'times': position['dates'],
                    'style': {'color' : position['color'] },
                    'icon': 'circle',
                    'iconstyle':{
                        'fillColor': position['color'],
                        'fillOpacity': 1,
                        'stroke': 'true',
                        'radius': 5
                    }
                }
            }
         for position in order_data]


        data_lines = {'type': 'FeatureCollection', 'features': features + positions }

        # convert the seconds into an ISO timestring
        if duration != 0:
            iso_duration = f"PT{duration}S"
        else:
            iso_duration = None

        folium.plugins.TimestampedGeoJson(data_lines, period='PT1S', transition_time=400, add_last_point=False, duration=iso_duration).add_to(m)

        m.save(filename)

