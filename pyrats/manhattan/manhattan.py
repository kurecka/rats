from AEVEnv import AEVEnv
import numpy as np


# defaults from their implementation
reloads = ['42431659','42430367','1061531810','42443056','1061531448','42448735','596775930','42435275','42429690','42446036','42442528','42440966','42431186','42438503','42442977',
 '42440966','1061531802','42455666']

targets = ['42440465','42445916']


# states - labels (json), see above 
# actions - ActionData objects
class ManhattanEnv:

    # TODO: adjust starting state
    def __init__(self, capacity, targets, reloads=None, init_state=None):
        self.env = AEVEnv(capacity, targets, reloads, init_state)

        if init_state is None:
            # randomize starting state
            self.current_state = self.state_to_name(np.random.choice(self.env.consmdp.num_states))

        else:
            self.current_state = init_state
        self.name = "ManhattanEnv"


    # translate names (json labels) to cmdp states used in functions
    def name_to_state(self, name):
        return self.env.consmdp.names_dict[name]

    def state_to_name(self, state):
        return self.env.state_to_name[state]

    def get_actions_for_state(self, name):
        state_id = self.name_to_state(name)
        action_it = self.env.consmdp.actions_for_state(state_id)

        # hacky way to get actions out of the iterator
        return [ action for action in action_it ]

    def possible_actions(self):
        return self.get_actions_for_state(self.current_state)

    def get_action(self, idx):
        return self.possible_actions()[idx]

    def outcome_probabilities(self, name, action):

        dist = []
        for k, v in action.distr:
            print(k, v)





x = ManhattanEnv(0, targets)
print(x.possible_actions())
print(x.outcome_probabilities( x.current_state, x.get_action(0) ))

