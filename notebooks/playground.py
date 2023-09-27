import rats
import rats
import ray
import numpy as np

ray.init(address="auto")
# ray.init()

@ray.remote
def task(thd):
    rats.set_log_level('info')
    e = rats.InvestorEnv(2, 20)
    a = rats.RAMCP(
        rats.EnvironmentHandler(e),
        max_depth=20, num_sim=1000, risk_thd=thd, gamma=1,
        exploration_constant=0.05, graphviz_depth=-1
    )
    a.reset()
    e.reset()
    while not e.is_over():
        a.play()
    
    h = a.get_handler()

    return h.get_reward(), h.get_penalty()

thds = np.linspace(0.1, 0.9, 7)

futures = [task.remote(thd) for thd in thds for _ in range(1000)]
results = ray.get(futures)
results = np.array(results).reshape(len(thds), -1, 2)
means = results.mean(axis=1)

for thd, (r, p) in zip(thds, means):
    print(f"thd={thd:.2f}, reward={r:.2f}, penalty={p:.2f}")
