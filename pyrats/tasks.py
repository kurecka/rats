import agents
import envs
from utils import set_log_level
from omegaconf import OmegaConf
import ray
import pandas as pd
import hydra
from pathlib import Path


@ray.remote
def single_run(agent, env):
    env_class = getattr(envs, env['class'])
    conf = OmegaConf.to_container(env, resolve=True)
    conf.pop('class')
    env = env_class(**conf)

    agent_class = getattr(agents, agent['class'])
    conf = OmegaConf.to_container(agent, resolve=True)
    conf.pop('class')
    agent = agent_class(envs.EnvironmentHandler(env), **conf)

    while not agent.get_handler().is_over():
        agent.play()
    
    h = agent.get_handler()
    return h.get_reward(), h.get_penalty()


def run_experiment(agent, env, num_episodes):
    results = []
    for i in range(num_episodes):
        results.append(single_run.remote(agent, env))

    df = pd.DataFrame(ray.get(results), columns=['reward', 'penalty'])
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    df.to_csv(output_dir / 'reults.csv')