import rats
import ray
import numpy as np

ray.init(address="auto")

@ray.remote
def task():
    rats.set_log_level('info')
    e = rats.InvestorEnv(2, 20)
    a = rats.ParetoUCT(
        rats.EnvironmentHandler(e),
        max_depth=0, num_sim=1000, risk_thd=0.23, gamma=1,
        exploration_constant=1, graphviz_depth=7
    )
    a.reset()
    e.reset()
    while not e.is_over():
        a.play()
    
    h = a.get_handler()

    return h.get_reward(), h.get_penalty()

futures = [task.remote() for _ in range(1000)]
results = ray.get(futures)
print(np.mean(results, axis=0))
