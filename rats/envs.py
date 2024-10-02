import _rats
from _rats import Avoid, SoftAvoid, InvestorEnv, CCPOMCP_EX1, CCPOMCP_EX2, Manhattan


class EnvironmentHandler:
    def __init__(self, env, *args, **kwargs):
        self.template_type = env.template_type()
        class_name = "EnvironmentHandler__" + self.template_type
        klass = getattr(_rats, class_name)

        self.env_handler = klass(env, *args, **kwargs)
    
    def __getattr__(self,attr):
        if attr == "__pure__":
            return self.env_handler
        return getattr(self.env_handler, attr)
