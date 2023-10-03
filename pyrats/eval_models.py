import ray
import numpy as np
import envs
import agents


# ray.init(address="auto")
ray.init()

@ray.remote
def task(agent_name, args):
    cls = getattr(agents, agent_name)
    from utils import set_log_level
    set_log_level('info')
    e = envs.InvestorEnv(2, 20)
    a = cls(envs.EnvironmentHandler(e), **args)
    # a = agents.ParetoUCT(
    #     agents.EnvironmentHandler(e),
    #     max_depth=100, num_sim=200, risk_thd=thd, gamma=1,
    #     exploration_constant=0.05, graphviz_depth=-1
    # )
    e.reset()
    a.reset()
    while not a.get_handler().is_over():
        a.play()
    
    h = a.get_handler()

    return h.get_reward(), h.get_penalty()

uct_args = {
    'max_depth': 100,
    'num_sim': 200,
    'gamma': 1,
    'exploration_constant': 0.05,
}

thds = np.linspace(0.1, 0.9, 7)
algos = [
    'ParetoUCT',
    'PrimalUCT',
    'DualUCT',
]

experiments = []

for algo in algos:
    for thd in thds:
        args = uct_args.copy()
        args['risk_thd'] = thd
        experiments.append((algo, args))

futures = {}

for agent_name, args in experiments:
    futures[(agent_name, args['risk_thd'])] = [task.remote(agent_name, args) for _ in range(500)]

for (agent_name, thd), f in futures.items():
    results = ray.get(f)
    # results = np.array(results).reshape(len(thds), -1, 2)
    results = np.array(results)
    r, p = results.mean(axis=0)
    print(f"algo={agent_name}, thd={thd:.2f} ->reward={r:.2f}, penalty={p:.2f}")
