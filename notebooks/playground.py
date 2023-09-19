import rats

rats.set_log_level('info')

e = rats.InvestorEnv(2, 20)
a = rats.ParetoUCT(
    rats.EnvironmentHandler(e),
    max_depth=8, num_sim=1000, risk_thd=0.2, gamma=1,
    exploration_constant=1,
)

# e.reset()
# a.reset()
# a.play()

o = rats.Orchestrator()
o.load_agent(a)
o.load_environment(e)
o.run(10)
    
