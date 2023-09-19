import rats

rats.set_log_level('INFO')

e = rats.InvestorEnv(2, 20)

thd = 0.0

agents = [
    rats.ConstantAgent(rats.EnvironmentHandler(e), 0),
    rats.ConstantAgent(rats.EnvironmentHandler(e), 1),
    rats.PrimalUCT(rats.EnvironmentHandler(e), max_depth=10, num_sim=40, risk_thd=thd, gamma=0.9),
    rats.DualUCT(rats.EnvironmentHandler(e), max_depth=10, num_sim=40, risk_thd=thd, gamma=0.9),
    rats.ParetoUCT(rats.EnvironmentHandler(e), max_depth=10, num_sim=40, risk_thd=thd, gamma=0.9),
]

for agent in agents:
    n = 100
    o = rats.Orchestrator()
    o.load_agent(agent)
    o.load_environment(e)
    o.run(n)
    
