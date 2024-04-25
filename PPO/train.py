import torch
from gym_torcs import TorcsEnv
from PPO import PPO
import rl_utils


state_dim = 29
action_dim = 3
actor_lr = 0.0001
critic_lr = 0.001
num_episodes = 2000
gamma = 0.9
lmbda = 0.9
epochs = 10
eps = 0.2
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
env = TorcsEnv(vision=False, throttle=True, gear_change=False)
torch.manual_seed(0)


if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


agent = PPO(state_dim, action_dim, actor_lr, critic_lr, lmbda, epochs, eps, gamma, device)
return_list = rl_utils.train_on_policy_agent(env, agent, num_episodes)





















