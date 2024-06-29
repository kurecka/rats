
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
                   environment terminates, passing capacity = 0 results in a
                   non-terminating environment

        cons_thd - the delay threshold on the orders, each cons_thd fuel consumed during an active order incurs a penalty of 1
                   on the agent.

        radius - the distance limit on accepting orders in kilometers
    """
    def __init__(self, targets, init_state, period, capacity, cons_thd=10.0, radius=2.0):
        self.env = AEVEnv.AEVEnv(capacity, targets, [], init_state,
                datafile="/work/rats/rats/manhattan_res/NYC.json",
                mapfile ="/work/rats/rats/manhattan_res/NYC.graphml")

        # maps checkpoint ids to state and history information
        self.checkpoints = dict()

        if not init_state:
            # randomize starting state
            self.init_state = self.random_state()
        else:
            self.init_state = init_state

        self.position = self.init_state

        # last known gps position, used when evaluating distance from targets
        self.last_gps = None

        self.energy = capacity
        self.capacity = capacity

        self.cons_thd = cons_thd
        self.order_delay = 0

        self.targets = targets
        self.period = period

        # multiplier to the period, once the order is finished
        self.cooldown = 5
        self.radius = radius

        # decision node -> special part of state (flag) - signals that orders
        # can be accepted at this step
        self.decision_node = False

        # ctr for each target, -1 is active target (accepted order)
        self.state_of_targets = { t : period for t in self.targets }

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

    # generate a random state
    # TODO: can generate dummy states this way, which results in unwanted behavior
    # however, we do not use this method in any benchmarks as of now.
    def random_state(self):
        return self.state_to_name(np.random.choice(self.env.consmdp.num_states))

    def target_active(self, target):
        return self.state_of_targets[target] == -1

    def currently_delivering(self):
        for t in self.targets:
            if self.target_active(t):
                return True
        return False
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
            return [ -1 ] + [ i for i, t in enumerate(self.targets) if self.state_of_targets[t] == 0 ]

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
        return possible_actions()[idx]

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


    def reload_ctrs(self):
        for t in self.targets:
            if self.state_of_targets[t] == 0:
                self.state_of_targets[t] = self.period


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
            self.decision_node = False

            # action is an index to targets
            if action != -1:
                # accept target
                self.state_of_targets[self.targets[action]] = -1
            self.reload_ctrs()

            return (self.position, self.state_of_targets, self.decision_node), 0, 0, self.is_over()

        # otherwise, proceed by moving in the underlying cmdp, recording
        # reward/penalty and adjusting periods of orders, record the state into
        # history as well
        self.history.append( self.position )

        # get linked list of actions from cmdp
        state_id = self.name_to_state(self.position)
        action_iterator = self.env.consmdp.actions_for_state(state_id)

        # get the action from iterator
        action_data = next(islice(action_iterator, action, None))

        # get successor
        next_state = np.random.choice(list(action_data.distr.keys()),
                                      p=list(action_data.distr.values()))

        # skip dummy state, record penalty
        action_iterator = self.env.consmdp.actions_for_state(next_state)
        action_data = next(islice(action_iterator, 0, None))

        next_state = next(iter(action_data.distr))
        self.position = self.state_to_name(next_state)

        # if limited energy budget is set, decrease the current energy
        if self.capacity > 0:
            self.energy -= action_data.cons

        # if an order is active, add to delay
        if self.currently_delivering():
            self.order_delay += action_data.cons

        # decrease counters for targets
        self.decision_node = self.decrease_ctrs(action_data.cons)

        penalty = 0
        # penalize for delay
        if self.order_delay > self.cons_thd:
            penalty = 1
            self.order_delay = 0

        # reward if order is delivered
        reward = 0
        if (self.position in self.targets) and (self.state_of_targets[self.position] == -1):
            reward = 1
            self.state_of_targets[self.position] = self.period * self.cooldown

            # if this was the only order, set order delay back to zero
            if not self.currently_delivering():
                self.order_delay = 0

        return (self.position, self.state_of_targets, self.decision_node), float(reward), float(penalty), self.is_over()


    # if capacity is zero the environment does not terminate
    def is_over(self):
        return self.capacity > 0 and self.energy <= 0

    def make_checkpoint(self, checkpoint_id):
        counter_copy = copy.deepcopy(self.state_of_targets)
        self.checkpoints[checkpoint_id] = (self.position, self.last_gps, self.energy, self.order_delay, counter_copy, self.decision_node)
        self.histories[checkpoint_id] = len(self.history)

    def restore_checkpoint(self, checkpoint_id):
        self.position, self.last_gps, self.energy, self.order_delay, self.state_of_targets, self.decision_node = self.checkpoints[checkpoint_id]
        history_id = self.histories[checkpoint_id]
        self.history = self.history[:history_id]

    def reset(self):
        self.energy = self.capacity
        self.position = self.init_state
        self.state_of_targets = { t : self.period for t in self.targets }
        self.decision_node = False
        
        self.checkpoints.clear()
        self.history = []

    """
        Visualizes the positions in self.history on the map of manhattan.
        Taken directly from AEVEnv
    """
    # interval signals number of frames between each animation plot
    def animate_simulation(self, interval=100, filename="map.html"):
        """
        Obtain the animation of a simulation instance where the agent reaches
        the target state from the initial state using assigned counterstrategy
        """
        targets = self.targets
        init_state = self.init_state

        def is_int(s):
            try:
                int(s)
                return True
            except ValueError:
                return False


        # Load NYC Geodata
        for _, _, data in self.G.edges(data=True, keys=False):
            data['time_mean'] = float(data['time_mean'])
        for _, data in self.geo_data:
            data['lat'] = float(data['lat'])
            data['lon'] = float(data['lon'])

        for target in targets:
            if target not in list(self.G.nodes):
                targets.remove(target)

        trajectory = []

        # filter dummy states
        for position  in self.history:
            if position not in list(self.G.nodes):
                pass
            else:
                trajectory.append( position )

        # create baseline map
        nodes_all = {}
        for node in self.G.nodes.data():
            name = str(node[0])
            point = [node[1]['lat'], node[1]['lon']]
            nodes_all[name] = point
        global_lat = []; global_lon = []
        for name, point in nodes_all.items():
            global_lat.append(point[0])
            global_lon.append(point[1])
        min_point = [min(global_lat), min(global_lon)]
        max_point =[max(global_lat), max(global_lon)]
        m = folium.Map(zoom_start=1, tiles='cartodbpositron')
        m.fit_bounds([min_point, max_point])

        # add initial state, reload states and target states
        folium.CircleMarker(location=[self.G.nodes[init_state]['lat'], self.G.nodes[init_state]['lon']],
                        radius= 3,
                        popup = 'initial state',
                        color='green',
                        fill_color = 'green',
                        fill_opacity=1,
                        fill=True).add_to(m)

        for node in targets:
            folium.CircleMarker(location=[self.G.nodes[node]['lat'], self.G.nodes[node]['lon']],
                        radius= 3,
                        popup = 'target state',
                        color="red",
                        fill_color = "red",
                        fill_opacity=1,
                        fill=True).add_to(m)
        # Baseline time

        t = time.time()
        path = list(zip(trajectory[:-1], trajectory[1:]))
        lines = []
        current_positions = []
        for pair in path:

            # pull out positions from the history tuple
            pos1, pos2 = pair

            t_edge = 1
            lines.append(dict({'coordinates':
                [[self.G.nodes[pos1]['lon'], self.G.nodes[pos1]['lat']],
                [self.G.nodes[pos2]['lon'], self.G.nodes[pos2]['lat']]],
                'dates': [time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(t)),
                           time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(t+t_edge))],
                           'color':'black'}))
            current_positions.append(dict({'coordinates':[self.G.nodes[pos2]['lon'], self.G.nodes[pos2]['lat']],
                        'dates': [time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(t+t_edge))]}))
            t = t+t_edge

        features = [{'type': 'Feature',
                     'geometry': {
                                'type': 'LineString',
                                'coordinates': line['coordinates'],
                                 },
                     'properties': {'times': line['dates'],
                                    'style': {'color': line['color'],
                                              'weight': line['weight'] if 'weight' in line else 2
                                             }
                                    }
                     }
                for line in lines]

        positions = [{
            'type': 'Feature',
            'geometry': {
                        'type':'Point',
                        'coordinates':position['coordinates']
                        },
            'properties': {
                'times': position['dates'],
                'style': {'color' : 'white'},
                'icon': 'circle',
                'iconstyle':{
                    'fillColor': 'white',
                    'fillOpacity': 1,
                    'stroke': 'true',
                    'radius': 2
                }
            }
        }
         for position in current_positions]
        data_lines = {'type': 'FeatureCollection', 'features': features}
        data_positions = {'type': 'FeatureCollection', 'features': positions}
        folium.plugins.TimestampedGeoJson(data_lines,  transition_time=interval,
                               period='PT1S', add_last_point=False, date_options='mm:ss', duration=None).add_to(m)

        m.save(filename)
