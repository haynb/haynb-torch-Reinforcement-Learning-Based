import torch
import PPO_Net
import rl_utils
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR


class PPO:
    def __init__(self,state_dim,action_dim,actor_lr,critic_lr,lmbda,epochs,eps,gamma,devices) -> None:
        self.actor=PPO_Net.ActorNet(state_dim,action_dim).to(devices)
        self.critic=PPO_Net.ValueNet(state_dim).to(devices)
        self.actor_optimizer=torch.optim.Adam(self.actor.parameters(),lr=actor_lr)
        self.critic_optimizer=torch.optim.Adam(self.critic.parameters(),lr=critic_lr)
        self.lmbda=lmbda
        self.epochs=epochs
        self.eps=eps
        self.gamma=gamma
        self.devices=devices
#        self.scheduler = StepLR(self.actor_optimizer, step_size=200, gamma=0.1)
#        self.scheduler = StepLR(self.critic_optimizer, step_size=200, gamma=0.1)



    def take_action(self,state):
        state = torch.FloatTensor(state).to(self.devices)
        mu, std = self.actor(state)
        action_dist = torch.distributions.Normal(mu, std)
        action = action_dist.sample()
        return action.detach().cpu().numpy()



    def update(self,transition_dict):
        states = torch.FloatTensor(transition_dict['states']).to(self.devices)
        actions = torch.FloatTensor(transition_dict['actions']).to(self.devices)
#        print("actions------------------",actions.shape)
        rewards = torch.FloatTensor(transition_dict['rewards']).to(self.devices)
#        print("rewards------------------",rewards.shape)
        next_states = torch.FloatTensor(transition_dict['next_states']).to(self.devices)
        dones = torch.FloatTensor(transition_dict['dones']).to(self.devices)
        td_target = rewards.unsqueeze(1).expand(-1, 3) + self.gamma * self.critic(next_states) * (1 - dones.unsqueeze(1).expand(-1, 3))
#        print("td_target------------------",td_target.shape)
        td_delta = td_target - self.critic(states)
#        print("td_delta------------------",td_delta.shape)
        advantages = rl_utils.compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(self.devices)
#        print("advantages------------------",advantages.shape)
        old_mu, old_std = self.actor(states)
        action_dist = torch.distributions.Normal(old_mu.detach(), old_std.detach())
        # old_log_prob = action_dist.log_prob(actions).sum(dim=1, keepdim=True)
        old_log_prob = action_dist.log_prob(actions)
        for _ in range(self.epochs):
            mu, std = self.actor(states)
            action_dist = torch.distributions.Normal(mu, std)
            log_prob = action_dist.log_prob(actions)
#            print("log_prob------------------",log_prob.shape)
            ratio = torch.exp(log_prob - old_log_prob)
#            print("ratio---------------------",ratio.shape)
#            print("advantages------------------",advantages.shape)
            # advantages = advantages.unsqueeze(-1)


            surr1 = ratio * advantages




            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()



        torch.save(self.actor.state_dict(), 'actormodel.pth')
        torch.save(self.critic.state_dict(), 'criticmodel.pth')
#        self.scheduler.step()
        print("model saved")
        print("actor-lr_____________________",self.actor_optimizer.param_groups[0]['lr'])
        print("critic-lr_____________________",self.critic_optimizer.param_groups[0]['lr'])
        pass