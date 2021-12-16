import torch
import torch.optim as optim
import torch.nn.functional as F
from networks import Actor, Critic
from torch.nn.utils import clip_grad_norm_
from utils import *


class Agent:
    """与环境交互并且学习好的策略"""
    def __init__(self, state_size, action_size, hidden_size, config):
        self.state_size = state_size
        self.action_size = action_size
        self.config = config
        self.device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")

        # Actor网络
        self.actor = Actor(state_size, action_size, hidden_size).to(self.device)
        self.actor_target = Actor(state_size, action_size, hidden_size).to(self.device)

        # Critic网络
        self.critic = Critic(state_size, action_size, hidden_size).to(self.device)
        self.critic_target = Critic(state_size, action_size, hidden_size).to(self.device)

        # optimizer网络
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.lr)

        # OU噪声
        self.noise = OUNoise(action_size, mu=config.ou_mu, theta=config.ou_theta, sigma=config.ou_sigma)
        self.epsilon = config.epsilon

        # ReplayBuffer
        self.buffer = ReplayBuffer(action_size, buffer_size=config.buffer_size, batch_size=config.batch_size)

    def reset(self):
        self.noise.reset()

    def step(self, state, action, reward, next_state, done, timestamp):
        """往buffer中保存经验， 并且使用随机抽样进行学习"""
        self.buffer.add(state, action, reward, next_state, done)

        if len(self.buffer) > self.config.batch_size and timestamp % self.config.learn_every == 0:
            for _ in range(self.config.n_updates):
                experiences = self.buffer.sample()
                self.learn(experiences)

    def get_action(self, state, add_noise=True):
        # 根据当前策略返回给定状态的操作，确定性策略
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)  # 增加一个维度给batch_size
        self.actor.eval()
        action = self.actor(state).detach().cpu().numpy().squeeze(0)
        self.actor.train()
        if add_noise:
            action += self.noise.sample() * self.epsilon
        return np.clip(action, -1, 1)

    def learn(self, experiences):
        """
        使用一个批次的经验轨迹数据来更新值网络和策略网络
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state)) ：这个是基于真实值的标签
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        """
        gamma = self.config.gamma
        states, actions, rewards, next_states, dones = experiences

        # 更新值函数的网络
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        Q_targets = rewards + (gamma * Q_targets_next * (1-dones))
        Q_expected = self.critic(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        clip_grad_norm_(self.critic.parameters(), max_norm=self.config.max_grad_norm)
        self.critic_optimizer.step()

        # 更新策略函数的网络
        actions = self.actor(states)  # 策略梯度
        actor_loss = -self.critic(states, actions).mean()  # actions是预测的，本身是一个参数theta
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 更新目标网络
        self.soft_update(self.critic, self.critic_target, tau=self.config.soft_update_tau)
        self.soft_update(self.actor, self.actor_target, tau=self.config.soft_update_tau)

        # 更新epsilon和噪声
        self.epsilon *= 1
        self.noise.reset()

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

