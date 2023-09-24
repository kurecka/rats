import rats

rats.set_log_level('info')

e = rats.InvestorEnv(2, 20)
a = rats.ParetoUCT(
    rats.EnvironmentHandler(e),
    max_depth=8, num_sim=100, risk_thd=0.2, gamma=1,
    exploration_constant=1, graphviz_depth=5
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
# o.run(10)
    
