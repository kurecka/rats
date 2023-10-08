import agents
import envs
import ray
from ray import tune

ray.init(address="erinys02.fi.muni.cz:6379")

def train(e, a, n_episodes=100):
    mean_reward = 0
    mean_penalty = 0

    for _ in range(n_episodes):
        a.reset()
        e.reset()
        while not a.get_handler().is_over():
            a.play()
        
        h = a.get_handler()
        
        mean_reward += h.get_reward() / n_episodes
        mean_penalty += h.get_penalty() / n_episodes

    return mean_reward, mean_penalty


map = """
#######
#BTTTG#
#..T..#
#.....#
##TT#.#
#GTTG.#
#..T..#
#######
"""

def objective(config):
    e = envs.Hallway(0.1, map)
    a = agents.ParetoUCT(
        envs.EnvironmentHandler(e),
        max_depth=100, sim_time_limit=50, risk_thd=0.2, gamma=0.99,
        exploration_constant=config['c'],
    )

    r, p = train(e, a, n_episodes=200)

    return {"reward": r, "penalty": p}


search_space = {
    # "d": tune.randint(5, 20),
    "c": tune.loguniform(1e-3, 100),
}

tuner = tune.Tuner(
    objective,
    param_space=search_space,
    tune_config=tune.TuneConfig(num_samples=500)
)

results = tuner.fit()
print(results.get_best_result(metric="penalty", mode="min").config)

# print(objective(results.get_best_result(metric="penalty", mode="min").config))