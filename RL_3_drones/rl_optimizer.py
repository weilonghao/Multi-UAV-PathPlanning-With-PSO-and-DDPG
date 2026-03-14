import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import heapq

from config import (
    DDPG_ACTOR_LR, DDPG_CRITIC_LR, DDPG_GAMMA, DDPG_TAU,
    DDPG_BATCH_SIZE, DDPG_BUFFER_SIZE
)


class Actor(nn.Module):


    def __init__(self, state_dim, hidden_dim=256):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # 使用LayerNorm替代BN
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),  # 输出二维坐标建议
            nn.Sigmoid()  # 限制输出在[0,1]范围
        )

        # 初始化最后一层权重
        nn.init.uniform_(self.net[-2].weight, -3e-3, 3e-3)
        nn.init.uniform_(self.net[-2].bias, -3e-3, 3e-3)

    def forward(self, x):
        return self.net(x)


class Critic(nn.Module):
    """改进的Critic网络（状态动作联合输入）"""

    def __init__(self, state_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.state_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU()
        )

        self.action_net = nn.Sequential(
            nn.Linear(2, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU()
        )

        self.q_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # 初始化输出层权重
        nn.init.uniform_(self.q_net[-1].weight, -3e-3, 3e-3)
        nn.init.uniform_(self.q_net[-1].bias, -3e-3, 3e-3)

    def forward(self, state, action):
        state_out = self.state_net(state)
        action_out = self.action_net(action)
        return self.q_net(torch.cat([state_out, action_out], dim=-1))


class PrioritizedReplayBuffer:
    """带优先级的经验回放缓冲区"""

    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.max_priority = 1.0

    def push(self, state, action, reward, next_state, done):
        """存储经验并赋予当前最大优先级"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)

        self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.priorities[self.pos] = self.max_priority ** self.alpha
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        """优先采样并计算重要性权重"""
        if len(self.buffer) == 0:
            return None

        priorities = self.priorities[:len(self.buffer)]
        probs = priorities / priorities.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)

        # 计算重要性权重
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()

        batch = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.float32),
            np.array(rewards, dtype=np.float32).reshape(-1, 1),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32).reshape(-1, 1),
            indices,
            np.array(weights, dtype=np.float32).reshape(-1, 1)
        )

    def update_priorities(self, indices, priorities):
        """更新采样经验的优先级"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = (priority + 1e-5) ** self.alpha  # 避免零优先级
        self.max_priority = max(self.max_priority, priorities.max())

    def __len__(self):
        return len(self.buffer)


class DDPGAgent:
    """改进的DDPG智能体"""

    def __init__(self, state_dim, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.state_dim = state_dim
        self.device = device

        # 创建网络
        self.actor = Actor(state_dim).to(device)
        self.actor_target = Actor(state_dim).to(device)
        self.critic = Critic(state_dim).to(device)
        self.critic_target = Critic(state_dim).to(device)

        # 初始化目标网络
        self._soft_update(self.actor, self.actor_target, tau=1.0)
        self._soft_update(self.critic, self.critic_target, tau=1.0)

        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=DDPG_ACTOR_LR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=DDPG_CRITIC_LR)

        # 经验回放（带优先级）
        self.replay_buffer = PrioritizedReplayBuffer(DDPG_BUFFER_SIZE)

        # 训练参数
        self.gamma = DDPG_GAMMA
        self.tau = DDPG_TAU
        self.batch_size = DDPG_BATCH_SIZE
        self.noise_scale = 0.3
        self.noise_decay = 0.9995
        self.min_noise = 0.1

    def select_action(self, state, add_noise=True):
        """选择动作（带探索噪声）"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state).cpu().numpy()[0]
        self.actor.train()

        # 添加OU噪声（更适合连续动作空间）
        if add_noise:
            noise = self.noise_scale * np.random.randn(2) * (1 - self.noise_decay)
            action = np.clip(action + noise, 0.0, 1.0)

        return action

    def update(self):
        """优先经验回放更新网络"""
        if len(self.replay_buffer) < self.batch_size * 10:  # 确保足够样本
            return

        # 优先采样
        batch = self.replay_buffer.sample(self.batch_size)
        if batch is None:
            return

        states, actions, rewards, next_states, dones, indices, weights = batch

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)

        # --- 更新Critic ---
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            y = rewards + (1 - dones) * self.gamma * target_q

        current_q = self.critic(states, actions)
        td_errors = (y - current_q).abs().detach().cpu().numpy()  # 用于更新优先级
        critic_loss = (weights * F.mse_loss(current_q, y, reduction='none')).mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)  # 梯度裁剪
        self.critic_optimizer.step()

        # --- 更新Actor ---
        actor_loss = -self.critic(states, self.actor(states)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        # --- 更新优先级 ---
        self.replay_buffer.update_priorities(indices, td_errors)

        # --- 软更新目标网络 ---
        self._soft_update(self.actor, self.actor_target, self.tau)
        self._soft_update(self.critic, self.critic_target, self.tau)

        # --- 噪声衰减 ---
        self.noise_scale = max(self.min_noise, self.noise_scale * self.noise_decay)

        return critic_loss.item(), actor_loss.item()

    def _soft_update(self, local_model, target_model, tau):
        """软更新目标网络"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def save(self, filename):
        """保存模型"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'noise_scale': self.noise_scale
        }, filename)

    def load(self, filename):
        """加载模型"""
        checkpoint = torch.load(filename)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.noise_scale = checkpoint.get('noise_scale', 0.3)