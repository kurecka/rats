import rats
import gymnasium as gym

rats.set_log_level("DEBUG")

e = rats.FrozenLake()
h = rats.EnvironmentHandler(e)
a = rats.PrimalUCT(h, max_depth=10, num_sim=10, risk_thd=0.4, gamma=0.99)

e.reset()
a.reset()
h.reset()

while not e.is_over():
    a.play()
    print(h.get_num_steps(), h.get_current_state(), h.get_reward(), h.get_penalty(), e.is_over())
