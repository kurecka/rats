import rats

rats.set_log_level('trace')

e = rats.InvestorEnv(5, 20)
h = rats.EnvironmentHandler(e)
a = rats.DualUCT(h, max_depth=10, risk_thd=0.5, num_sim=1, gamma=0.9)
a.reset()
e.reset()

a.play()