import agents
import envs
from utils import set_log_level
from omegaconf import OmegaConf
import ray
import pandas as pd
import hydra
from pathlib import Path
from rats import build_info

@ray.remote
def single_run(agent, env, rats_version):
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

    from rats import build_info
    assert rats_version == build_info(), "rats version mismatch"

    return h.get_reward(), h.get_penalty()


def run_experiment(agent, env, num_episodes):
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    rats_version = build_info()
    with open(output_dir / 'rats_version.txt', 'w') as f:
        f.write(rats_version)

    results = []
    for i in range(num_episodes):
        results.append(single_run.remote(agent, env, rats_version))

    df = pd.DataFrame(ray.get(results), columns=['reward', 'penalty'])
    df.to_csv(output_dir / 'results.csv', index=False)
