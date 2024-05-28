#!/bin/bash

#list of instances 1 up to 12
instances=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)
n=1000
t="1,2,5"
c="0.1,5,20"
# slidea=(0 0.2)
# trapa=(0.1 0.7)

slidea=(0.2)
trapa=(0.7)

agents="dual_uct,dual_ramcp,ramcp,pareto_uct"
# agents="dual_ramcp"


for i in "${instances[@]}"
do
    for slide in "${slidea[@]}"
    do
        for trap in "${trapa[@]}"
        do
            exp="FF_$slide_$trap_$n"
            echo "Running instance $i, n $n, t $t, c $c, slide $slide, trap $trap"
            ray job submit --no-wait -- sh -c "cd /work/rats/pyrats && python experiment.py -m +task=indep_runs +agent=$agents ++agent.exploration_constant=${c} ++agent.sim_time_limit=${t} +env=final_${i} ++metadata.tag=final_${i}_${exp} ++env.slide_prob=$slide ++env.trap_prob=$trap ++task.num_episodes=$n ++gamma=0.99 ++risk_thd=0,0.1,0.2,0.3,0.4"
        done
    done
done

# ray job submit --no-wait -- sh -c "cd /work/rats/pyrats && python experiment.py -m +task=indep_runs +agent=dual_uct,dual_ramcp,ramcp,pareto_uct ++agent.exploration_constant=0.1,5 ++agent.sim_time_limit=5 +env=ramcp_ce ++metadata.tag=ramcp_FINAL ++task.num_episodes=300 ++gamma=0.99 ++risk_thd=0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1"
