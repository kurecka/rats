#!/usr/bin/env python3

import pandas as pd
from pathlib import Path
import yaml
from sys import argv

from matplotlib import pyplot as plt
import numpy as np


def process_run_results(run_directory):
    run_directory = Path(run_directory)
    results_file = run_directory / 'results.csv'
    config_file = run_directory / '.hydra' / 'config.yaml'
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    agent = config['agent']['class'] + "_" + str(config['agent']['sim_time_limit']) + "_" + str(config['agent']['exploration_constant'])
    risk_thd = config['risk_thd']

    df = pd.read_csv(results_file).agg(['mean', 'std'])
    df = pd.concat([df.reward, df.penalty], keys=['reward', 'penalty'], axis=0).to_frame().T
    df.index = [risk_thd]
    df.index.name = 'risk_thd'

    experiment_desc = {
        'env': config['env']['class'],
        'num_episodes': config['task']['num_episodes'],
    }

    return df, agent, experiment_desc

def process_job_dir(job_dir):
    job_dir = Path(job_dir)

    if not (job_dir / 'sweep').exists():
        return process_run_results(job_dir)
    else:
        data = {}
        for sub_job_dir in (job_dir/'sweep').iterdir():
            if sub_job_dir.is_dir():
                try:
                    df, agent, experiment = process_run_results(sub_job_dir)
                    if agent not in data:
                        data[agent] = df
                    else:
                        data[agent] = pd.concat([data[agent], df])
                except FileNotFoundError:
                    pass
        for df in data.values():
            df.sort_index(inplace=True)
        
        table = pd.concat(data.values(), keys=data.keys(), axis=1)
        with open(job_dir / 'table.txt', 'w') as f:
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
                f.write(str(table))
        
        df = pd.concat(data.values(), keys=data.keys(), axis=0)
        df.reset_index(inplace=True, names=['agent', 'risk_thd'])
        
        # plot rewards with stripes
        df['low_reward'] = df.reward['mean'] - df.reward['std']
        df['high_reward'] = df.reward['mean'] + df.reward['std']
        df['low_penalty'] = df.penalty['mean'] - df.penalty['std']
        df['high_penalty'] = df.penalty['mean'] + df.penalty['std']

        # plot rewards and penalties on different subplots
        fig, ax = plt.subplots(2, 1, sharex=True)
        # set dimensions
        fig.set_figheight(8)
        fig.set_figwidth(16)
        # set ticks
        ax[0].set_xticks(np.linspace(0, 1, 11))
        # add grid
        ax[0].grid()
        ax[1].grid()
        # clip ax[0] to [-0.1, 1.1]
        # ax[0].set_ylim(bottom=-0.1, top=3.1)
        for agent, df_agent in df.groupby('agent'):
            ax[0].plot(df_agent.risk_thd, df_agent.reward['mean'], label=agent)
            # ax[0].fill_between(df_agent.risk_thd, df_agent.low_reward, df_agent.high_reward, alpha=0.25)
            ax[1].plot(df_agent.risk_thd, df_agent.penalty['mean'], label=agent, linestyle='--')
            # ax[1].fill_between(df_agent.risk_thd, df_agent.low_penalty, df_agent.high_penalty, alpha=0.25)
        ax[0].set_ylabel('Expected reward')
        ax[1].set_ylabel('Expected penalty')
        ax[1].set_xlabel('Risk threshold')
        ax[0].legend()
        ax[1].legend()

        # set title
        fig.suptitle(f'Env {experiment["env"]}, {experiment["num_episodes"]} episodes')
        fig.savefig(job_dir / 'results.png')

process_job_dir(argv[1])
