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
#     map = """#######
# #BTTTG#
# #..T..#
# #.....#
# #######
# """
#     e = envs.Hallway(map, 0.1)
    a = cls(envs.EnvironmentHandler(e), **args)
    e.reset()
    a.reset()
    while not a.get_handler().is_over():
        a.play()
    
    h = a.get_handler()

    return h.get_reward(), h.get_penalty()

uct_args = {
    'max_depth': 100,
    'num_sim': 0,
    'sim_time_limit': 200,
    'gamma': 1,
    'exploration_constant': 5,
}

thds = np.linspace(0, 1, 11)
algos = [
    'DualUCT',
    'ParetoUCT',
    'PrimalUCT',
    'RAMCP',
]

experiments = []

for algo in algos:
    for thd in thds:
        args = uct_args.copy()
        if algo == 'DualUCT':
            # args['sim_time_limit'] = int(args['sim_time_limit'] / 2)
            args['initial_lambda'] = 50
            args['lr'] = 1
        args['risk_thd'] = thd
        experiments.append((algo, args))

futures = {}

for agent_name, args in experiments:
    futures[(agent_name, args['risk_thd'])] = [task.remote(agent_name, args) for _ in range(10)]

results_map = {}

for (agent_name, thd), f in futures.items():
    results = ray.get(f)
    # results = np.array(results).reshape(len(thds), -1, 2)
    results = np.array(results)
    mean_r, mean_p = results.mean(axis=0)
    std_r, std_p = results.std(axis=0)
    if agent_name not in results_map:
        results_map[agent_name] = {}
    results_map[agent_name][thd] = {
        'mean_r': mean_r,
        'mean_p': mean_p,
        'std_r': std_r,
        'std_p': std_p,
    }

print(args)

for agent_name, data in results_map.items():
    print(agent_name)
    for thd, d in data.items():
        print(f'thd {thd:.2f}\t{d["mean_r"]:.2f}±{d["std_r"]:.2f}\t{d["mean_p"]:.2f}±{d["std_p"]:.2f}')