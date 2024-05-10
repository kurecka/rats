from fimdpenv import AEVEnv
from itertools import islice
import numpy as np
import networkx as nx

# defaults from their implementation
reloads = ['42431659','42430367','1061531810','42443056','1061531448','42448735','596775930','42435275','42429690','42446036','42442528','42440966','42431186','42438503','42442977',
 '42440966','1061531802','42455666']

targets = ['42440465','42445916']


# states - labels (json), see above 
# actions - integers, (see variable aid in fimdp/core.py)

# TODO: add energy to S ? probably not, it is not relevant for the agent, only determines when the environment is over
# S - state_label, state_of_targets, decision_node
# A - index size_t
class ManhattanEnv:

    """
        capacity - starting capacity of the agent
        targets - the reachability targets, currently receive unit reward for
            hitting a state in targets
        init_state - initial state  (json label), randomize if none

        self.energy / self.capacity -> curr/max energy

        self.history -> tracks data that is recorded before calling step()
            i.e. S = (state, state_of_targets, decision_node, energy)

        MDP state consists of the current state, the counters for each target, and decision node indicator
    """
    def __init__(self, capacity, targets, periods, init_state="", cons_thd=10):
        self.env = AEVEnv.AEVEnv(capacity, targets, [], init_state,
                datafile="manhattan_res/NYC.json",
                mapfile ="manhattan_res/NYC.graphml")

        # maps int -> ( state, energy )
        self.checkpoints = dict()

        if not init_state:
            # randomize starting state
            self.init_state = self.random_state()
        else:
            self.init_state = init_state

        self.current_state = self.init_state
        self.capacity = capacity
        self.energy = capacity
        self.targets = targets
        self.periods = periods
        self.cons_thd = cons_thd
        self.decision_node = False

        # ctr for each target, -1 is active target
        self.state_of_targets = { t : self.periods[t] for t in self.targets }

        self.history = []


    """
        helper methods
    """
    # translate names (json labels) to cmdp states used in functions
    def name_to_state(self, name):
        return self.env.consmdp.names_dict[name]

    # translate indices in cmdp to json labels
    def state_to_name(self, state):
        return self.env.state_to_name[state]

    # generate a random state (TODO: use for delivery targets)
    def random_state(self):
        return self.state_to_name(np.random.choice(self.env.consmdp.num_states))

    """
        env interface methods
    """

    def name(self):
        return "ManhattanEnv"

    def num_of_actions(self):
        return len( self.possible_actions() )

    """
        get the number of actions for the current state
        if in decision node, return indices to active targets
    """
    def get_actions_for_state(self, name):
        if ( self.decision_node ):
            return [ -1 ] + [ i for i, t in enumerate(self.targets) if self.state_of_targets[t] == -1 ]

        state_id = self.name_to_state(name)
        action_count = len(self.env.consmdp.actions_for_state(state_id))
        return list(range(action_count))

    def current_state(self):
        return self.current_state

    # c++ state is a tuple of (state_name, state_of_targets, decision_node)
    def possible_actions(self, state = None):
        state_name = self.current_state if state == None else state[0]
        return self.get_actions_for_state(state_name)

    def get_action(self, idx):
        return idx

    def outcome_probabilities(self, name, action):

        dist = dict()

        state_id = self.name_to_state(name)

        # ActionData... source target, consumption, distr
        action_iterator = self.env.consmdp.actions_for_state(state_id)

        # the iterator does not have random access, pull out the action like this
        action_data = next(islice(action_iterator, action, None))

        for x in action_data.distr.keys():
            dist[self.state_to_name(x)] = float(action_data.distr[x])

        return dist
    
    def decrease_ctrs(self, cons):
        nearby = self.find_nearby_targets(self.current_state)
        for t in self.targets:
            if self.state_of_targets[t] != -1:
                self.state_of_targets[t] -= min(cons, self.state_of_targets[t])

                # if target is 0 and not nearby, reset his ctr
                if self.state_of_targets[t] == 0 and t not in nearby:
                    self.state_of_targets[t] = self.periods[t]

    def reload_ctrs(self):
        for t in self.targets:
            if self.state_of_targets[t] == 0:
                self.state_of_targets[t] = self.periods[t]

    """
        find nearby targets by their GPS coordinates,
        they should be closer than a certain threshold
    """
    def find_nearby_targets(self, state_name, radius=0.01):
        G = nx.MultiDiGraph(nx.read_graphml(self.env.mapfile))
        targets = self.targets
        data = G.nodes(data=True)[state_name]
        state_lat = float(data['lat'])
        state_lon = float(data['lon'])
        nearby = []

        for target in targets:
            target_data = G.nodes(data=True)[target]
            target_lat = float(target_data['lat'])
            target_lon = float(target_data['lon'])
            if ( (state_lat - target_lat)**2 + (state_lon - target_lon)**2 ) < radius:
                nearby.append(target)

        return nearby
        

    def play_action(self, action):

        # record relevant information (used for animating trajectory)
        self.history.append( (self.current_state, self.state_of_targets, self.decision_node) )

        # if in decision node, choose if to accept any of the ready targets (at most one)
        if self.decision_node:
            self.decision_node = False
            self.reload_ctrs()

            # action is an index to targets
            if action != -1:
                # accept target
                self.state_of_targets[self.targets[action]] = -1
            
            return (self.current_state, self.state_of_targets, self.decision_node), 0, 0, self.is_over()
        

        # get linked list of actions from cmdp
        state_id = self.name_to_state(self.current_state)
        action_iterator = self.env.consmdp.actions_for_state(state_id)

        # the iterator does not have random access, pull out the action like this
        action_data = next(islice(action_iterator, action, None))

        # get successor
        next_state = np.random.choice(list(action_data.distr.keys()),
                                      p=list(action_data.distr.values()))
        
        # skip dummy state, record penalty
        action_iterator = self.env.consmdp.actions_for_state(next_state)
        action_data = next(islice(action_iterator, 0, None))
        next_state = action_data.target # or have get from distribution ??

        self.current_state = self.state_to_name(next_state)

        # termination condition ??
        self.energy -= action_data.cons

        # decrease counters for targets
        self.decrease_ctrs(action_data.cons)

        # penalize if consumption is too high and some target is active
        penalty = (action_data.cons > self.cons_thd) * len([1 for t in self.targets if self.state_of_targets[t] == -1]) > 0

        # reward if target is reached
        reward = 0
        if (self.current_state in self.targets) and (self.state_of_targets[self.current_state] == -1):
            reward = 1
            self.state_of_targets[self.current_state] = self.periods[self.current_state]

        # move to dummy node if target ctr hits 0
        for t in self.targets:
            if self.state_of_targets[t] == 0:
                self.decision_node = True
                break

        return (self.current_state, self.state_of_targets, self.decision_node), float(reward), float(penalty), self.is_over()


    # TODO: adjust for other constraint
    def is_over(self):
        return self.energy <= 0

    def make_checkpoint(self, checkpoint_id):
        self.checkpoints[checkpoint_id] = (self.current_state, self.energy, self.state_of_targets, self.decision_node)

    def restore_checkpoint(self, checkpoint_id):
        self.current_state, self.energy, self.state_of_targets, self.decision_node = self.checkpoints[checkpoint_id]

    def reset(self):
        self.energy = self.capacity
        self.current_state = self.init_state
        self.state_of_targets = { t : self.periods[t] for t in self.targets }
        self.decision_node = False

        self.history = []

    """
        taken from AEVEnv

        call after final execution of the policy, i.e. 
        after calling play_action() as many times as you wish
    """
    # FIXME: not implemented yet
    def animate_simulation(self):
        """
        Obtain the animation of a simulation instance where the agent reaches
        the target state from the initial state using assigned counterstrategy
        """
        
        targets = self.targets
        init_state = self.init_state
        reloads = self.reloads

        def is_int(s):
            try: 
                int(s)
                return True
            except ValueError:
                return False

        
        # Load NYC Geodata
        G = nx.MultiDiGraph(nx.read_graphml(self.mapfile))
        for _, _, data in G.edges(data=True, keys=False):
            data['time_mean'] = float(data['time_mean'])
        for _, data in G.nodes(data=True):
            data['lat'] = float(data['lat'])
            data['lon'] = float(data['lon'])
        for reload in reloads:
            if reload not in list(G.nodes):
                reloads.remove(reload)
        for target in targets:
            if target not in list(G.nodes):
                targets.remove(target)

        trajectory = []

        # filter dummy states
        for state, energy, targets in self.history:
            if state not in list(G.nodes):
                pass
            else:
                trajectory.append( (state, energy, targets) )

        # create baseline map
        nodes_all = {}
        for node in G.nodes.data():
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
        folium.CircleMarker(location=[G.nodes[init_state]['lat'], G.nodes[init_state]['lon']],
                        radius= 3,
                        popup = 'initial state',
                        color='green',
                        fill_color = 'green',
                        fill_opacity=1,
                        fill=True).add_to(m)        
        for node in reloads:
            folium.CircleMarker(location=[G.nodes[node]['lat'], G.nodes[node]['lon']],
                        radius= 1,
                        popup = 'reload state',
                        color="#0f89ca",
                        fill_color = "#0f89ca",
                        fill_opacity=1,
                        fill=True).add_to(m)
        """
        this gets plotted on the fly instead 

        for node in targets:
            folium.CircleMarker(location=[G.nodes[node]['lat'], G.nodes[node]['lon']],
                        radius= 3,
                        popup = 'target state',
                        color="red",
                        fill_color = "red",
                        fill_opacity=1,
                        fill=True).add_to(m)

        """
        # Baseline time
        t = time.time()
        path = list(zip(trajectory[:-1], trajectory[1:]))
        lines = []
        current_positions = []
        for pair in path:

            t_edge = 1
            lines.append(dict({'coordinates':
                [[G.nodes[pair[0]]['lon'], G.nodes[pair[0]]['lat']],
                [G.nodes[pair[1]]['lon'], G.nodes[pair[1]]['lat']]],
                'dates': [time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(t)),
                           time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(t+t_edge))],
                           'color':'black'}))
            current_positions.append(dict({'coordinates':[G.nodes[pair[1]]['lon'], G.nodes[pair[1]]['lat']],
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
        return m

if __name__ == "__main__":
    x = ManhattanEnv(3000, targets, reloads)
    for i in range(10):
        print(x.outcome_probabilities(x.current_state, 0))
        x.play_action(np.random.choice(x.possible_actions()))

