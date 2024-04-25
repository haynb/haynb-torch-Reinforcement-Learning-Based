from gym_torcs import TorcsEnv



env = TorcsEnv(vision=False, throttle=True, gear_change=False)

print(env.action_space.shape[0])
# env.action_space.shape[0]