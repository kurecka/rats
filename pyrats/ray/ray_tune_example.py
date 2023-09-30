import rats
import ray
from ray import tune

ray.init(address="erinys02.fi.muni.cz:6379")

def train(e, a, n_episodes=100):
    mean_reward = 0
    mean_penalty = 0

    for _ in range(n_episodes):
        a.reset()
        e.reset()
        while not e.is_over():
            a.play()
        
        h = a.get_handler()
        
        mean_reward += h.get_reward() / n_episodes
        mean_penalty += h.get_penalty() / n_episodes

    return mean_reward, mean_penalty


def objective(config):
    rats.set_log_level('err')
    e = rats.InvestorEnv(2, 20)
    a = rats.ParetoUCT(
        rats.EnvironmentHandler(e),
        max_depth=config['d'], num_sim=100, risk_thd=0.6, gamma=0.9,
        exploration_constant=config['c'],
    )

    r, p = train(e, a, n_episodes=1000)

    return {"reward": r, "penalty": p}


search_space = {
    "d": tune.randint(5, 20),
    "c": tune.loguniform(1e-3, 1e-1),
}

tuner = tune.Tuner(
    objective,
    param_space=search_space,
    tune_config=tune.TuneConfig(num_samples=500)
)

results = tuner.fit()
print(results.get_best_result(metric="penalty", mode="min").config)

print(objective(results.get_best_result(metric="penalty", mode="min").config))