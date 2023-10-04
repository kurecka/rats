import rats


class Agent:
    def __init__(self, env_handler, *args, **kwargs):
        self.template_type = env_handler.template_type
        class_name = self.__class__.__name__ + "__" + self.template_type
        klass = getattr(rats, class_name)
        self.agent = klass(env_handler.__pure__, *args, **kwargs)
        
    def __getattr__(self, attr):
        return getattr(self.agent, attr)

    def __str__(self):
        return str(self.agent)

    def __repr__(self):
        return repr(self.agent)

class ConstantAgent(Agent):
    pass

class RandomizedAgent(Agent):
    pass

class DualUCT(Agent):
    pass

class PrimalUCT(Agent):
    pass

class ParetoUCT(Agent):
    pass

class RAMCP(Agent):
    pass
