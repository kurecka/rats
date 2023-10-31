from utils import set_log_level
from omegaconf import OmegaConf
import ray
import hydra
import pandas as pd
from pathlib import Path
import sys


@ray.remote
def run(agent, env, rats_version, num_episodes=1, sys_path=[]):
    sys.path = sys_path + sys.path
    import agents
    import envs
    from rats import build_info
    assert rats_version == build_info(), "rats version mismatch {} != {}".format(rats_version, build_info())

    env_class = getattr(envs, env['class'])
    conf = OmegaConf.to_container(env, resolve=True)
    conf.pop('class')
    env = env_class(**conf)

    agent_class = getattr(agents, agent['class'])
    conf = OmegaConf.to_container(agent, resolve=True)
    conf.pop('class')
    agent = agent_class(envs.EnvironmentHandler(env), **conf)

    results = []

    for i in range(num_episodes):
        print("Episode {}/{}".format(i+1, num_episodes))
        env.reset()
        agent.reset()
        while not agent.get_handler().is_over():
            agent.play()
        if agent.is_trainable():
            agent.train()
    
        h = agent.get_handler()
        results.append((h.get_reward(), h.get_penalty()))

    return results


# @ray.remote(resources={"HeadResource": 0.1})
@ray.remote
def launch(agent, env, num_episodes, output_dir, independent_runs=True):
    from rats import build_info
    rats_version = build_info()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'rats_version.txt', 'w') as f:
        f.write(rats_version)

    results = []
    if independent_runs:
        for i in range(num_episodes):
            results.append(run.remote(agent, env, rats_version))
    else:
        results.append(run.remote(agent, env, rats_version, num_episodes))


    results = sum(ray.get(results), [])
    df = pd.DataFrame(results, columns=['reward', 'penalty'])
    df.to_csv(output_dir / 'results.csv', index=False)


def run_experiment(agent, env, num_episodes, independent_runs=True):
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    return launch.remote(agent, env, num_episodes, output_dir, independent_runs)
