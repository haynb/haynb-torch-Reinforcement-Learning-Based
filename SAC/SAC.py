import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F




import SAC_Net



class SAC():
    def __init__(self,state_dim,action_dim,action_lr,critic_lr,alpha_lr,target_entropy,tau,gamma,device):
        self.actor = SAC_Net.ActorNet(state_dim,action_dim).to(device)
        self.critic1 = SAC_Net.QNet(state_dim,action_dim).to(device)
        self.critic2 = SAC_Net.QNet(state_dim,action_dim).to(device)
        print("loading model")
        try:

            self.actor.load_state_dict(torch.load('actormodel.pth'))
            self.critic1.load_state_dict(torch.load('criticmodel1.pth'))
            self.critic2.load_state_dict(torch.load('criticmodel2.pth'))
            print("model load successfully")
        except:
            print("cannot find the model")
        self.target_critic1 = SAC_Net.QNet(state_dim,action_dim).to(device)
        self.target_critic2 = SAC_Net.QNet(state_dim,action_dim).to(device)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),lr=action_lr)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(),lr=critic_lr)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(),lr=critic_lr)
        self.log_alpha = torch.tensor(np.log(0.01),dtype=torch.float32,requires_grad=True,device=device)
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],lr=alpha_lr)
        self.target_entropy = target_entropy
        self.tau = tau
        self.gamma = gamma
        self.device = device



    def take_action(self,state):
        state = torch.FloatTensor(state).to(self.device)
        action,log_prob = self.actor(state)
        return action.detach().cpu().numpy()


    def calc_target(self,next_state,reward,done):
        next_actions, log_prob = self.actor(next_state)
        entropy = -log_prob
        q1 = self.target_critic1(next_state,next_actions)
        q2 = self.target_critic2(next_state,next_actions)
        next_value = torch.min(q1,q2) + self.log_alpha.exp() * entropy
        reward = reward.unsqueeze(1).expand(-1, 3)
        done = done.unsqueeze(1).expand(-1, 3)
        td_target = reward + self.gamma * (1-done) * next_value
        return td_target
    
    def soft_update(self,net,target_net):
        for param,target_param in zip(net.parameters(),target_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1-self.tau) * target_param.data)
    

    def update(self,transition_dict):
        states = torch.FloatTensor(transition_dict['states']).to(self.device)
        actions = torch.FloatTensor(transition_dict['actions']).to(self.device)
        rewards = torch.FloatTensor(transition_dict['rewards']).to(self.device)
        next_states = torch.FloatTensor(transition_dict['next_states']).to(self.device)
        done = torch.FloatTensor(transition_dict['dones']).to(self.device)
        td_target = self.calc_target(next_states,rewards,done)
        critic1_loss = torch.mean(F.mse_loss(self.critic1(states,actions),td_target.detach()))
        critic2_loss = torch.mean(F.mse_loss(self.critic2(states,actions),td_target.detach()))
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        new_actions,log_prob = self.actor(states)
        entropy = -log_prob
        q1 = self.critic1(states,new_actions)
        q2 = self.critic2(states,new_actions)
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy - torch.min(q1,q2))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        alpha_loss = torch.mean((entropy-self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()
        self.soft_update(self.critic1,self.target_critic1)
        self.soft_update(self.critic2,self.target_critic2)
        # return critic1_loss.item(),critic2_loss.item(),actor_loss.item(),alpha_loss.item()
    

    def save_model(self):
        torch.save(self.actor.state_dict(), 'actormodel.pth')
        torch.save(self.critic1.state_dict(), 'criticmodel1.pth')
        torch.save(self.critic2.state_dict(), 'criticmodel2.pth')
        print("model saved")

