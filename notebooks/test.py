import rats

rats.set_log_level('TRACE')

o = rats.Orchestrator()
e = rats.InvestorEnv(5, 20)
e.reset()
e.play_action(0)

h = rats.EnvironmentHandler(e)
h.play_action(0)
c1 = rats.ConstantAgent(rats.EnvironmentHandler(e), action=0)

# o.load_environment(e)
o.load_agent(c1)


# o.run(10)


# print(f'Orchestrator', flush=True)
# o = rats.Orchestrator()
# print(f'Environment', flush=True)
# e = rats.InvestorEnv(5, 20)
# print(f'Reset env', flush=True)
# e.reset()
# print(f'Play action', flush=True)
# e.play_action(0)

# print(f'Environment handler', flush=True)
# h = rats.EnvironmentHandler(e)
# print(f'Play action', flush=True)
# h.play_action(0)
# print(f'Agent', flush=True)
# c1 = rats.ConstantAgent(rats.EnvironmentHandler(e), action=0)

# print(f'Load agent', flush=True)
# o.load_agent(c1)
# print(f'Load environment', flush=True)
# o.load_environment(e)

# print(f'Run', flush=True)
# o.run(10)