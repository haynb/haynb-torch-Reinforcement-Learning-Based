import torch
from torch.autograd import Variable
import numpy as np
import random
from gym_torcs import TorcsEnv
import argparse
import collections
from tensorboardX import SummaryWriter




from ReplayBuffer import ReplayBuffer             #用于经验回放的缓存器
from ActorNetwork import ActorNetwork             #用于构建Actor和Critic神经网络的类
from CriticNetwork import CriticNetwork           #用于构建Actor和Critic神经网络的类
import OU                                 #是Ornstein-Uhlenbeck噪声生成器,用于探索和策略发现。


#创建一个SummaryWriter对象，用于记录训练信息,保存在log目录中
writer = SummaryWriter(log_dir='logs')


"""定义了一些超参数和模型输入输出的尺寸,包括状态(state)和动作(action)的大小,
学习率(LRA和LRC),缓存器大小(BUFFER_SIZE)和批量大小(BATCH_SIZE),
折扣因子(GAMMA),探索因子(EXPLORE),以及训练指示器(train_indicator)和软更新因子(TAU)。"""



state_size = 29
action_size = 3

"""LRA和LRC是两种不同的学习率，它们的含义如下：


LRA（Learning Rate for Activations）：指的是在反向传播过程中，用于更新神经元激活值的学习率。在深度学习模型中，激活值通常是通过激活函数计算得到的，
例如ReLU函数、Sigmoid函数等。LRA的作用是控制这些激活值在反向传播过程中的更新速度，从而影响模型的训练效果。
LRC（Learning Rate for Connections）：指的是在反向传播过程中，用于更新神经元之间连接权重的学习率。在深度学习模型中，
每个神经元都和其它神经元之间有连接，这些连接权重通常是通过反向传播算法更新的。LRC的作用是控制这些连接权重在反向传播过程中的更新速度，从而影响模型的训练效果。
总的来说，LRA和LRC都是控制深度学习模型中参数更新速度的超参数，但是它们控制的是不同的参数。LRA控制的是神经元的激活值，而LRC控制的是神经元之间的连接权重。
在实际应用中，选择合适的LRA和LRC通常需要进行多次实验来确定最佳的超参数组合。"""
LRA = 0.0001
LRC = 0.001
BUFFER_SIZE = 100000  #to change
BATCH_SIZE = 32


"""折扣因子（Gamma）是强化学习中一个重要的超参数，通常用符号 γ 表示。它用于控制未来奖励的折扣程度，即在计算累计奖励时，考虑到未来奖励的价值。
折扣因子的值通常介于 0 和 1 之间，当 γ 接近 0 时，模型更注重当前奖励，而当 γ 接近 1 时，模型更注重未来奖励。
探索因子（Explore）是强化学习中用于平衡探索（Explore）和利用（Exploit）的超参数。通常用符号 ε 表示，它表示在选择动作时，
有多大的概率选择随机动作而不是当前最优动作。探索因子的值通常在训练过程中逐渐减小，以便让模型在开始时更多地进行探索，逐渐过渡到利用最优动作。
训练指示器（train_indicator）通常是用来表示强化学习算法当前处于训练阶段还是测试阶段的变量。在训练阶段，模型通过与环境交互获取经验，并使用经验进行参数更新；
而在测试阶段，模型不再进行参数更新，而是仅仅根据当前的策略执行动作，获取奖励并评估模型的表现。
软更新因子（TAU）是深度强化学习中用于实现目标网络（Target Network）更新的超参数。目标网络是在更新当前策略网络时使用的辅助网络，
它通常使用之前训练过的策略网络的参数作为初始值，以便更快地学习。软更新因子用于平滑地将当前策略网络的参数更新到目标网络中，以避免参数更新过程中的不稳定性和震荡。
总的来说，折扣因子、探索因子、训练指示器和软更新因子都是强化学习中的重要超参数，它们对于模型的训练和表现都有重要影响。
选择合适的超参数组合通常需要进行多次实验和调优。"""


GAMMA = 0.95
EXPLORE = 100000.
epsilon = 1
train_indicator = 1    # train or not
TAU = 0.001

VISION = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'


"""实例化了OU噪声生成器，并定义了一个函数init_weights()，用于初始化神经网络的权重和偏置。然后创建了Actor和Critic神经网络实例，
并将它们的权重初始化为随机数，其中Actor网络的输入维度为状态大小，输出维度为动作大小，Critic网络的输入维度为状态和动作大小的总和，输出维度为1。"""


OU = OU()

"""在DDPG（Deep Deterministic Policy Gradient）算法中，OU噪声生成器通常用于增加Actor网络输出的随机性，以便更好地探索环境和学习最优策略。"""


"""Actor和Critic神经网络是DDPG算法中的两个重要组成部分。Actor网络负责根据当前状态生成动作，Critic网络负责评估Actor网络生成的动作的优劣性。具体来说，
Actor网络的输入是当前状态，输出是对应的动作；而Critic网络的输入是当前状态和Actor网络生成的动作，输出是对应的动作价值（Q值）。
在DDPG算法中，Actor和Critic网络通过交替训练来不断更新参数，并逐渐优化策略和价值函数，最终学习到最优策略。初始化Actor和Critic网络的权重是非常重要的，
因为初始权重的好坏可能会对模型的收敛速度和效果产生重要影响。函数init_weights()的作用是初始化神经网络的权重和偏置，通常使用正态分布或者均匀分布等随机初始化方法，
以便让模型更好地进行训练和学习。"""

def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.normal_(m.weight, 0, 1e-4)
        m.bias.data.fill_(0.0)

"""调用actor.apply(init_weights)的作用是将init_weights函数应用到actor模型中的所有线性层（即权重参数是torch.nn.Linear类型的层），
以初始化这些层的权重和偏置参数。"""
actor = ActorNetwork(state_size).to(device)
actor.apply(init_weights)                                    #初始化
critic = CriticNetwork(state_size, action_size).to(device)


"""尝试从磁盘加载已经训练好的Actor和Critic网络的权重，如果找到了就加载到对应的网络中，并将网络的模式设置为评估模式（eval()）。"""

#load model
print("loading model")
try:

    actor.load_state_dict(torch.load('actormodel.pth'))
    actor.eval()
    critic.load_state_dict(torch.load('criticmodel.pth'))
    critic.eval()
    print("model load successfully")
except:
    print("cannot find the model")



"""创建一个经验回放缓存器（ReplayBuffer），用于存储智能体的经验；同时创建一个Target Actor网络和Target Critic网络，它们的权重与Actor和Critic网络相同，
并将它们的模式设置为评估模式。定义了Critic网络的损失函数为均方误差（MSE），并使用Adam优化器来更新Actor和Critic网络的参数。
最后创建了一个Torcs仿真环境的实例，用于智能体的训练和评估。"""



#critic.apply(init_weights)
buff = ReplayBuffer(BUFFER_SIZE)

target_actor = ActorNetwork(state_size).to(device)
target_critic = CriticNetwork(state_size, action_size).to(device)
target_actor.load_state_dict(actor.state_dict())
target_actor.eval()
target_critic.load_state_dict(critic.state_dict())
target_critic.eval()


"""此部分定义了损失函数criterion_critic用于计算Critic网络的损失，
使用Adam优化算法分别创建了optimizer_actor和optimizer_critic用于更新Actor和Critic网络的参数。"""


criterion_critic = torch.nn.MSELoss(reduction='sum')

optimizer_actor = torch.optim.Adam(actor.parameters(), lr=LRA)
optimizer_critic = torch.optim.Adam(critic.parameters(), lr=LRC)

#env environment

"""此部分创建了TorcsEnv环境类的实例env，并根据是否支持GPU设置默认张量类型。"""


env = TorcsEnv(vision=VISION, throttle=True, gear_change=False)

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor') 


"""该部分为训练的核心部分。其中主要包含以下几个步骤：

从经验回放缓存器中取出一个批次的经验数据。
用神经网络计算出当前状态下的动作 $a_t$。
使用 Ornstein-Uhlenbeck 噪声生成器产生噪声，为动作加入一定的随机性，提高策略探索能力。
执行动作 $a_t$，并得到环境反馈的奖励 $r_t$ 和下一个状态 $s_{t+1}$。
将 $(s_t, a_t, r_t, s_{t+1})$ 存入经验回放缓存器。
从经验回放缓存器中取出一个批次的经验数据，并分别将其分为状态、动作、奖励、下一个状态和是否终止五部分。
根据 target network 计算 $y_t$ 值，作为训练 critic network 的目标值。
使用 critic network 计算 $q_{values}$，并根据 $y_t$ 和 $q_{values}$ 计算出 critic 的 loss。
对 critic 的梯度进行反向传播，更新参数。
使用 critic network 和 actor network 计算 $q_{values}$ 和 $a_{for_grad}$。
对 $q_{values}$ 求和，并计算出 $a_{for_grad}$ 关于 $q_{values}$ 的梯度。
对 actor 的梯度进行反向传播，更新参数。
对 target network 进行软更新，使得 target network 的参数逐渐向着实际 network 的参数靠拢。"""


l = 0
for i in range(2000):
    reward = []
    if np.mod(i, 10) == 0:
        ob = env.reset(relaunch = True)
    else:
        ob = env.reset()
    """这段代码是将游戏赛车环境中的观测(ob)转化为状态(s_t)。其中，ob包含了游戏赛车环境的各种信息，如车辆的状态、位置、速度、方向、转弯角度、油门、刹车等信息，而s_t是一个数值向量，用于表示车辆状态的状态特征。

具体来说，这段代码使用了np.hstack函数将所有的状态信息连接在一起，并组成了一个长度为29的一维数组。具体的状态信息包括：

ob.angle：车辆的方向角度，以弧度表示。
ob.track：当前车辆所在的赛道的状态，以浮点数数组的形式表示。
ob.trackPos：车辆距离赛道中心线的距离，以浮点数表示。
ob.speedX：车辆在x轴方向上的速度，以浮点数表示。
ob.speedY：车辆在y轴方向上的速度，以浮点数表示。
ob.speedZ：车辆在z轴方向上的速度，以浮点数表示。
ob.wheelSpinVel：车轮旋转的速度，以浮点数数组的形式表示。
ob.rpm：车辆发动机的转速，以浮点数表示。
在深度强化学习中，状态的特征提取是非常重要的一步，它能够有效地将高维的状态空间转化为一个低维的特征向量，并能够帮助智能体更好地学习到环境的规律和特征。"""
    s_t = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))
    
    for j in range(100000):
        loss = 0
        epsilon -= 1.0 / EXPLORE                                                          # 每轮训练结束后逐渐减小随机噪声的强度
        a_t = np.zeros([1, action_size])                                                    #初始化动作向量
        noise_t = np.zeros([1, action_size])                                                 #初始化噪声向量
        #ipdb.set_trace() 
        a_t_original = actor(torch.tensor(s_t.reshape(1, s_t.shape[0]), device=device).float())   # 获取当前状态下的原始动作向量

        if torch.cuda.is_available():
            a_t_original = a_t_original.data.cpu().numpy()
        else:
            a_t_original = a_t_original.data.numpy()
        #print(type(a_t_original[0][0]))

        noise_t[0][0] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][0], 0.0, 0.60, 0.30)   # 计算动作的随机噪声向量
        noise_t[0][1] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][1], 0.5, 1.00, 0.10)
        noise_t[0][2] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][2], -0.1, 1.00, 0.05)

        #stochastic brake
        # 在10%的情况下随机将第三个动作替换为随机噪声,随机刹车
        if random.random() <= 0.1:
            print("apply the brake")
            noise_t[0][2] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][2], 0.2, 1.00, 0.10)
        
        a_t[0][0] = a_t_original[0][0] + noise_t[0][0]                                          # 将第一个动作的随机噪声加到原始动作上
        a_t[0][1] = a_t_original[0][1] + noise_t[0][1]
        a_t[0][2] = a_t_original[0][2] + noise_t[0][2]

        ob, r_t, done, info = env.step(a_t[0])  # 调用环境函数执行新的动作，得到下一个状态和奖励  # 执行动作，获取新的观测、奖励、结束标志和其他信息
         # 根据得到的状态信息生成新的状态向量
        s_t1 = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))

        #add to replay buffer
        # 将这次经历加入经验回放缓存中
        buff.add(s_t, a_t[0], r_t, s_t1, done)

        batch = buff.getBatch(BATCH_SIZE)                 # 从缓存中获取一批经验


        """ # 获取这批经验中的状态，动作，奖励，下一个状态和done标志"""
        states = torch.tensor(np.asarray([e[0] for e in batch]), device=device).float()    #torch.cat(batch[0])
        actions = torch.tensor(np.asarray([e[1] for e in batch]), device=device).float()
        rewards = torch.tensor(np.asarray([e[2] for e in batch]), device=device).float()
        new_states = torch.tensor(np.asarray([e[3] for e in batch]), device=device).float()
        dones = np.asarray([e[4] for e in batch])
        y_t = torch.tensor(np.asarray([e[1] for e in batch]), device=device).float()
        
        #use target network to calculate target_q_value
        # 使用Target网络计算目标Q值
        target_q_values = target_critic(new_states, target_actor(new_states))

        for k in range(len(batch)):
            if dones[k]:
                 # 如果done标记为True，即为最后一个状态，此时无下一状态，直接赋予奖励值
                y_t[k] = rewards[k]
            else:
                 # 如果done标记为False，即为非最后一个状态，计算target Q值
                y_t[k] = rewards[k] + GAMMA * target_q_values[k]

        if(train_indicator):
            # 训练Actor-Critic网络
            #training
            # 计算当前Q值
            q_values = critic(states, actions)
            # 计算损失函数，即MSE
            loss = criterion_critic(y_t, q_values)  

            # 梯度清零
            optimizer_critic.zero_grad()
            # 计算反向传播
            loss.backward(retain_graph=True)                         ##for param in critic.parameters(): param.grad.data.clamp(-1, 1)
            # 优化网络参数
            optimizer_critic.step()
            # 计算Actor的梯度
            a_for_grad = actor(states)
            # 开启梯度记录
            a_for_grad.requires_grad_()    #enables the requires_grad of a_for_grad
            # 计算当前状态下的Q值
            q_values_for_grad = critic(states, a_for_grad)
            # 梯度清零
            critic.zero_grad()
            # 计算Q值的和，即期望的Q值
            q_sum = q_values_for_grad.sum()
            # 反向传播计算梯度
            q_sum.backward(retain_graph=True)
            # 获取Actor的梯度
            grads = torch.autograd.grad(q_sum, a_for_grad) #a_for_grad is not a leaf node  
             # grads是一个tuple，grads[0]才是我们需要的梯度值

            #grads[0] = -grads[0]
            #print(grads)   
            # 更新Actor网络的参数
            act = actor(states)
            actor.zero_grad()
            # 计算Actor的梯度并反向传播
            act.backward(-grads[0])
            # 优化网络参数
            optimizer_actor.step()

            #soft update for target network
            #actor_params = list(actor.parameters())
            #critic_params = list(critic.parameters())
            # 软更新target network
            # 将Actor和Critic的参数按比例更新到target network上
            print("soft updates target network")
            new_actor_state_dict = collections.OrderedDict()
            new_critic_state_dict = collections.OrderedDict()
            for var_name in target_actor.state_dict():
                new_actor_state_dict[var_name] = TAU * actor.state_dict()[var_name] + (1-TAU) * target_actor.state_dict()[var_name]
            target_actor.load_state_dict(new_actor_state_dict)

            for var_name in target_critic.state_dict():
                new_critic_state_dict[var_name] = TAU * critic.state_dict()[var_name] + (1-TAU) * target_critic.state_dict()[var_name]
            target_critic.load_state_dict(new_critic_state_dict)
        
        s_t = s_t1
        print("---Episode ", i , "  Action:", a_t, "  Reward:", r_t, "  Loss:", loss)
        ################################
        writer.add_scalar('train_loss', loss, l)
        
        
        l += 1


        reward.append(r_t)

        if done:


            writer.add_scalar('Reward', sum(reward), i)
            reward.clear()
            break


    """这段代码用于在每个 episode 结束时保存训练好的 actor 和 critic 网络参数。这些参数将被用于在下一次训练中作为 target network 的初始参数。

    此外，代码还调用了 env.end() 方法来终止环境的运行，最后输出训练结束的提示信息。"""


    if np.mod(i, 3) == 0:
        if (train_indicator):
            print("saving model")
            torch.save(actor.state_dict(), 'actormodel.pth')
            torch.save(critic.state_dict(), 'criticmodel.pth')

    
env.end()
print("Finish.")

#for param in critic.parameters(): param.grad.data.clamp(-1, 1)

