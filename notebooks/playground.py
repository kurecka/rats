import rats
# import numpy as np

rats.set_log_level('info')

e = rats.InvestorEnv(2, 20)

a = rats.ParetoUCT(
    rats.EnvironmentHandler(e),
    max_depth=10, num_sim=300, risk_thd=0.8, gamma=1,
    exploration_constant=1, graphviz_depth=-1
)

e.reset()
a.reset()

for i in range(3):
    a.play()
    with open(f"../logs/tree_{i}.dot", "w") as f:
        f.write(a.get_graphviz())


# o = rats.Orchestrator()
# o.load_agent(a)
# o.load_environment(e)
# o.run(200)

# A = np.zeros((21, 21))
# b = np.zeros(21)

# for i in range(21):
#     if i == 0:
#         A[i, i] = 1
#         b[i] = 0
#     elif i == 20:
#         A[i, i] = 1
#         b[i] = 1
#     else:
#         A[i, i] = 1
#         A[i, i+1] = -0.7
#         A[i, i-1] = -0.3
#         b[i] = 0

# print('[', end='')
# for v in 1-np.linalg.solve(A, b):
#     # dont use scientific notation
#     print(f"{v:.6f}", end=",")
# print(']')




