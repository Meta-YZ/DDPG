import torch
import torch.nn as nn


class Actor(nn.Module):
    """Actor网络or动作（策略）网络"""
    def __init__(self, state_size, action_size, hidden_size):
        super(Actor, self).__init__()
        self.net = nn.Sequential(nn.Linear(state_size, hidden_size), nn.ReLU(),
                                 nn.Linear(hidden_size, hidden_size), nn.ReLU(),
                                 nn.Linear(hidden_size, hidden_size), nn.Hardswish(),
                                 nn.Linear(hidden_size, action_size))

    def forward(self, state):
        return self.net(state).tanh()  # 将动作边界映射到[-1,1]


class Critic(nn.Module):
    """Critic网络（Q值orV值）"""
    def __init__(self, state_size, action_size, hidden_size):
        super(Critic, self).__init__()
        self.net = nn.Sequential(nn.Linear(state_size+action_size, hidden_size), nn.ReLU(),
                                 nn.Linear(hidden_size, hidden_size), nn.ReLU(),
                                 nn.Linear(hidden_size, hidden_size), nn.Hardswish(),
                                 nn.Linear(hidden_size, 1))

    def forward(self, state, action):
        return self.net(torch.cat((state, action), dim=1))  # 把action按列拼接到state屁股后变