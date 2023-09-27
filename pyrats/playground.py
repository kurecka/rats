import rats
import rats
import ray
import numpy as np


rats.set_log_level('info')
e = rats.InvestorEnv(2, 20)
a = rats.ParetoUCT(
    rats.EnvironmentHandler(e),
    max_depth=20, num_sim=4, risk_thd=0.2, gamma=1,
    exploration_constant=1, graphviz_depth=7
)


for i in range(30):
    e.reset()
    a.reset()
    while not e.is_over():
        a.play()


e.reset()
a.reset()

for i in range(3):
    a.play()
    with open(f"../logs/tree_{i}.dot", "w") as f:
        f.write(a.get_graphviz())
