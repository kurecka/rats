from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import multiprocessing
import os

"""
Visualization script,
note that the script assumes the _std entries for rew and penalty are sd's of
data and not the standard error of the mean, so they are normalized by
sqrt(runs) in aggregate_df_ifnormation and process_line

"""


# csv format: agent;time_limit;exp_const;mean_reward;mean_penalty;std_reward;std_penalty;feasible;emp_feasible;runs
# plot the mean reward and penalty of the (agent, exp_const) for each time limit,
# mark infeasible solutions with a red X
def visualize_large_hg( filename ):

    # filter path
    paths = filename.split('/')
    plot_name = paths[-1]

    df = pd.read_csv( filename , sep=';')
    agent_list = ["ParetoUCT", "DualUCT", "RAMCP"]
    exp_const = [1, 5, 20]

    for agent in agent_list:
        for c in exp_const:
            sub_df = df[(df['agent'] == agent) & (df['exp_const'] == c)]

            # plot reward
            plt.plot( sub_df['time_limit'], sub_df['mean_reward'], label='Mean reward',
                     marker='none', color = 'k')
            for row in sub_df.rows():
                if not row['feasible']:
                    plt.plot(row['time_limit'], row['mean_reward'], marker='X', markersize=10,
                             markeredgecolor='r', markerfacecolor='none')
                    
    plt.xlabel('Time limit (ms)')
    plt.ylabel('Reward')
    plt.title(f"{plot_name}")
    plt.legend()
    plt.grid(True, linewidth='0.4', linestyle='--')
    plt.savefig(f'large_HG_eval/plots/{plot_name}.png')
    plt.close()


def aggregate_df_information(dataframe, timesteps, samples):
   
    esat_t = [ 0 for _ in timesteps ]
    sat_t = [ 0 for _ in timesteps ]
    cr_t = [ 0 for _ in timesteps ]

    # mean time per step
    mtps_t = [ 0 for _ in timesteps ]
    i = 0
    for t in timesteps:
        
        # empirical satisfaction of the cost constraint
        esat_t[i] = dataframe[f"timestep_{t}_esat"].sum()

        # admitted satisfaction of the cost constraint, std of mean incorporated
        sat_t[i] = dataframe[f"timestep_{t}_sat"].sum()

        # cumulative reward
        cr_t[i] = 1/400 * dataframe[dataframe[f"timestep_{t}_sat"] == True][f"timestep_{t}_rew"].sum()

        # mean time per step of algorithm
        mtps_t[i] = dataframe[f'timestep_{t}_steptime'].mean()

        i += 1

    return sat_t, esat_t, cr_t, mtps_t


def aggregate_lp_information( df_lp ):
    sat_t = df_lp['feasible'].sum()
    cr_t = 1/400 * df_lp[df_lp[f"sat_timestep_{t}"] == True][f"timestep_{t}_rew"].sum()
    return sat_t, cr_t



# process line into entries of choice
def process_line( row, timesteps ):
    r = [row[f'timestep_{limit}_rew'] for limit in timesteps ]
    p = [row[f'timestep_{limit}_penalty'] for limit in timesteps ]
    feas = [row[f'timestep_{limit}_sat'] for limit in timesteps ]
    std = [row[f'timestep_{limit}_penalty_std'] for limit in timesteps ]

    return r, p, feas, std


def aggregate_small_hg( df_pareto, df_dual, df_ramcp, df_lp, timesteps, samples=100 ):

    columns=["name", "limit", "cr", "sat", "esat", "steptime"]
    out_df = pd.DataFrame( columns=columns )
    
    names = [ "ParetoUCT", "DualUCT", "RAMCP" ]
    name_idx = 0

    for df in [ df_pareto, df_dual, df_ramcp ]:

        name = names[name_idx]
        sat_t, esat_t, cr_t, mtps_t = aggregate_df_information( df, timesteps, samples )

        for i in range(len(timesteps)):
            row = {
                "name": name,
                "limit": timesteps[i],
                "cr": cr_t[i],
                "esat": esat_t[i],
                "sat": sat_t[i],
                "steptime": mtps_t[i]
            }
            out_df = pd.concat([out_df, pd.DataFrame(row, index=[0])], ignore_index=True)
        name_idx += 1

    out_df.to_csv("aggregated_results.csv", sep=';')

# samples is number of repetitions used to obtain the data for sat calculation
def visualize_small_hg( df_pareto, df_dual, df_ramcp, df_lp, timesteps, samples ):


    for i in df_pareto.index:
        lp_rew = df_lp.iloc[i]['lp_reward']
        name = df_pareto.iloc[i]['Benchmark']
        print("Plotting ", name)
        p_rew, p_pen, p_feas, p_std = process_line( df_pareto.iloc[i], timesteps )
        d_rew, d_pen, d_feas, d_std = process_line( df_dual.iloc[i], timesteps )
        r_rew, r_pen, r_feas, r_std = process_line( df_ramcp.iloc[i], timesteps )
        plot_hallway_rewards( name , timesteps, p_rew, p_feas,
                                                                     d_rew, d_feas,
                                                                     r_rew,
                                                                     r_feas,
                                                                     lp_rew )
        plot_hallway_penalty( name, timesteps, 
                              p_pen, p_std, d_pen, d_std, r_pen, r_std,
                              df_pareto.iloc[i]['c'], samples )

# plot summarizing performance of the three algorithms vs a lp baseline
def plot_hallway_rewards( name, timesteps, pareto_rews, pareto_feas,
                                           dual_rews, dual_feas,
                                           ramcp_rews, ramcp_feas,
                                           lp_baseline ):

    plt.figure(figsize=(10, 6))
    plt.axhline(y=lp_baseline, color='gray', linestyle='--', label='LP')

    # Plot the experiment results with distinct colors
    colors = ['blue', 'green', 'red']

    idx = 0
    for title, data, feas in [ ("ParetoUCT", pareto_rews, pareto_feas ),
                               ("DualUCT", dual_rews, dual_feas ),
                               ("RAMCP", ramcp_rews, ramcp_feas ) ]:

         plt.plot(timesteps, data, color=colors[idx], label=f'{title}')

         for i in range(len(data)):
             # mark infeasible rewards
             if not feas[i]:
                plt.plot(timesteps[i], data[i], marker='X', markersize=10,
                         markeredgecolor='r', markerfacecolor='none')


         idx += 1

    # Add labels and legend
    plt.xlabel('Time')
    plt.xticks(timesteps[1:], fontsize=8)
    plt.ylabel('Result')
    plt.title(f"Reward - {name}")
    plt.legend()
    plt.grid(True, linewidth='0.4', linestyle='--')
    plt.savefig(f'plots/{name}_reward.png')
    plt.close()

def plot_hallway_penalty( name, timesteps, pareto_pen, pareto_std,
                                           dual_pen, dual_std,
                                           ramcp_pen, ramcp_std, c , samples ):

    plt.figure(figsize=(10, 6))
    plt.axhline(y=c, color='gray', linestyle='--', label='LP')

    # Plot the experiment results with distinct colors
    colors = ['blue', 'green', 'red']

    idx = 0
    for title, data, std in [ ("ParetoUCT", pareto_pen, pareto_std ),
                         ("DualUCT", dual_pen, dual_std ),
                         ("RAMCP", ramcp_pen, ramcp_std ) ]:

         plt.plot(timesteps, data, color=colors[idx], label=f'{title}')

         ci_upper = np.array( data ) + 1.65 * np.array( std ) / np.sqrt( samples )
         ci_lower = np.array( data ) - 1.65 * np.array( std ) / np.sqrt( samples )

         plt.fill_between(timesteps, ci_upper, ci_lower, alpha=0.05, color=colors[idx])
         plt.plot(timesteps, ci_upper, color=colors[idx], linestyle='--',
                 linewidth=0.3, alpha=0.5)
         plt.plot(timesteps, ci_lower, color=colors[idx], linestyle='--',
                 linewidth=0.3, alpha=0.5)
         idx += 1

    # Add labels and legend
    plt.xlabel('Time')
    plt.xticks(timesteps[1:], fontsize=8)
    plt.ylabel('Penalty')
    plt.title(f"Penalty - {name}")
    plt.legend()
    plt.grid(True, linewidth='0.4', linestyle='--')
    plt.savefig(f'plots/{name}_penalty.png')
    plt.close()


def process_small_hg( timesteps, samples=100 ):

    # get all of the results
    df_pareto = pd.read_csv( "results_ParetoUCT.csv", sep = ';' )
    df_dual = pd.read_csv( "results_DualUCT.csv", sep = ';' )
    df_ramcp = pd.read_csv( "results_RAMCP.csv", sep = ';' )
    df_lp = pd.read_csv( "results.csv", sep = ';');


    # fix data sd to sem (std error of the mean c)
    # can delete later
    for df in [ df_pareto, df_dual, df_ramcp ]:
        for t in timesteps:

            # empirical satisfaction of the cost constraint
            df[f'timestep_{t}_esat']= (df[f"timestep_{t}_penalty"] <= df["c"])

            # admitted satisfaction of the cost constraint, std of mean incorporated
            df[f'timestep_{t}_sat']= df[f"timestep_{t}_penalty"] - 1.65 * df[f"timestep_{t}_penalty_std"] / np.sqrt(samples) <= df["c"]

    
    aggregate_small_hg( df_pareto, df_dual, df_ramcp, df_lp, timesteps, samples )
    visualize_small_hg( df_pareto, df_dual, df_ramcp, df_lp, timesteps, samples )



def visualize_small_hg_old( filename , time_limits ):

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
# time_limits = [5, 10, 25, 50, 100, 250, 500]
# process_small_hg( time_limits, samples=100 )

if __name__ == "__main__":

    # visualize all the large HG results in large_HG_eval/results
    for filename in os.listdir("large_HG_eval/results"):
        visualize_large_hg( f"large_HG_eval/results/{filename}" )
