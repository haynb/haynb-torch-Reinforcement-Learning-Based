import torch
from gym_torcs import TorcsEnv
from SAC import SAC
import rl_utils





state_dim = 29
action_dim = 3
actor_lr = 0.0001
critic_lr = 0.001
alpha_lr = 1e-4
num_episodes = 1000
gamma = 0.99
tau = 0.005
buffer_size = 1000000
minimal_size = 64
batch_size = 64
target_entropy = -action_dim
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
replay_buffer = rl_utils.ReplayBuffer(buffer_size)
agent = SAC(state_dim,action_dim,actor_lr,critic_lr,alpha_lr,target_entropy,tau,gamma,device)
env = TorcsEnv(vision=False, throttle=True, gear_change=False)
return_list = rl_utils.train_off_policy_agent(env,agent,num_episodes,replay_buffer,minimal_size,batch_size)

