from tqdm import tqdm
import numpy as np
import torch
import collections
import random
from OU import OU
from tensorboardX import SummaryWriter


OU = OU()
epsilon = 1.0


writer = SummaryWriter(log_dir='logs')


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity) 

    def add(self, state, action, reward, next_state, done): 
        self.buffer.append((state, action, reward, next_state, done)) 

    def sample(self, batch_size): 
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done 

    def size(self): 
        return len(self.buffer)

def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0)) 
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))

def train_on_policy_agent(env, agent, num_episodes):
    return_list = []
    num = 0
    # 接下来，实例化OU类，用于生成噪声
    global OU
    global epsilon
    global writer
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):


                episode_return = 0
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                ob = env.reset(relaunch = True)
                state = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))
                done = False
                num+=1
                reward_sum = 0
                while not done:
                    epsilon -= 1.0 / 100000
                    action = agent.take_action(state)
                    noise = np.zeros(3)
                    noise[0] = max(epsilon, 0) * OU.function(action[0], 0.0, 0.60, 0.30)
                    noise[1] = max(epsilon, 0) * OU.function(action[1], 0.5, 1.00, 0.10)
                    noise[2] = max(epsilon, 0) * OU.function(action[2], -0.1, 1.00, 0.05)
                    if random.random() <= 0.1:
                        print("apply the brake")
                        noise[2] = max(epsilon, 0) * OU.function(action[2], 0.2, 1.00, 0.10)

                    action = action + noise
                    next_state, reward, done, _ = env.step(action)
                    print("epsode--------",num,"Reward-----------",reward)
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    next_state = np.hstack((next_state.angle, next_state.track, next_state.trackPos, next_state.speedX, next_state.speedY, next_state.speedZ, next_state.wheelSpinVel/100.0, next_state.rpm))
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    reward_sum += reward
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward
                return_list.append(episode_return)
                print("epsode--------",num,"reward_sum-----------",reward_sum)
                writer.add_scalar('Reward', reward_sum, num)
                agent.update(transition_dict)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list

def train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _ = env.step(action)
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
                        agent.update(transition_dict)
                return_list.append(episode_return)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list


def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)
                