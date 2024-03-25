from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


def visualize_large_hg( filename ):

    # filter path
    paths = filename.split('/')
    plot_name = paths[-1]

    df = pd.read_csv( filename , sep=';')

    #TODO



def visualize_small_hg( filename , time_limits ):
    paths = filename.split('/')
    plot_name = paths[-1]

    df = pd.read_csv( filename, sep = ';' )

    for index, row in df.iterrows():
        lp_reward = row['lp_reward']

        pareto_rewards = [row[f'pareto_{limit}_rew'] for limit in time_limits ]
        pareto_feasible = [row[f'pareto_{limit}_feasible'] for limit in time_limits]
        pareto_penalty = [row[f'pareto_{limit}_penalty'] for limit in time_limits ]


        plt.plot( time_limits , pareto_rewards, label='Pareto rew',
                 marker='none', color = 'k')
        for i, limit in enumerate(time_limits):
            if not pareto_feasible[i]:
                plt.plot(limit, pareto_rewards[i], marker='X', markersize=10,
                         markeredgecolor='r', markerfacecolor='none')  # Red point for infeasible solutions

        plt.plot ( [ time_limits[0], time_limits[-1] ],
                   [ lp_reward for _ in range(2) ],
                   label = "LP reward", marker ='none', linestyle='--',
                  color='c')
        plt.xlabel('Pareto time limit (ms)')
        plt.ylabel('Reward')
        plt.title(f"Reward - {row['Benchmark']}")
        plt.legend()
        plt.grid(True, linewidth='0.4', linestyle='--')
        plt.savefig(f'plots/{row["Benchmark"]}_reward.png')
        plt.close()

        plt.plot( time_limits , pareto_penalty, label='Pareto penalty',
                 marker='none', color = 'k')

        plt.plot ( [ time_limits[0], time_limits[-1] ],
                   [ lp_reward for _ in range(2) ],
                   label = "constraint", marker ='none', linestyle='--',
                  color='c')

        plt.xlabel('Pareto time limit (ms)')
        plt.ylabel('Penalty')
        plt.title(f"Penalty - {row['Benchmark']}")
        plt.legend()
        plt.grid(True, linewidth='0.4', linestyle='--')
        plt.savefig(f'plots/{row["Benchmark"]}_penalty.png')
        plt.close()



# time_limits = [5, 10, 15, 25, 50, 100, 250, 500]
time_limits = [ 5, 10, 15, 25 ]
visualize_small_hg( 'results.csv', time_limits )
