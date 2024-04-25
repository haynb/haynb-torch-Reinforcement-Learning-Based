from tqdm import tqdm
import numpy as np
import torch
import collections
import random
from OU import OU
from tensorboardX import SummaryWriter


OU = OU()

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



def train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size):
    return_list = []
    num = 0
    global writer
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                ob = env.reset()
                state = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))
                done = False
                num += 1
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _ = env.step(action)
                    print("epsode--------",num,"reward----------------",reward)
                    next_state = np.hstack((next_state.angle, next_state.track, next_state.trackPos, next_state.speedX, next_state.speedY, next_state.speedZ, next_state.wheelSpinVel/100.0, next_state.rpm))
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
                        agent.update(transition_dict)
                agent.save_model()
                return_list.append(episode_return)
                print("epsode--------",num,"reward_sum-----------",episode_return)
                writer.add_scalar('Reward', episode_return, num)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list


