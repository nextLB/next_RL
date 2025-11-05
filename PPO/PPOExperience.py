"""
    PPO经验回放缓冲区
"""

import torch
import numpy as np
from collections import deque

class PPOExperienceBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.logProbs = []

    def push(self, state, action, reward, done, value, logProb):
        """添加经验到缓冲区"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        self.logProbs.append(logProb)

    def clear(self):
        """清空缓冲区"""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()
        self.logProbs.clear()

    def compute_returns_and_advantages(self, nextValue, gamma, gaeLambda):
        """计算GAE和回报"""
        rewards = np.array(self.rewards)
        values = np.array(self.values + [nextValue])
        dones = np.array(self.dones)

        # 计算TD误差
        deltas = rewards + gamma * values[1:] * (1 - dones) - values[:-1]

        # 计算GAE
        advantages = np.zeros_like(rewards, dtype=np.float32)
        advantage = 0
        for t in reversed(range(len(rewards))):
            advantage = deltas[t] + gamma * gaeLambda * (1 - dones[t]) * advantage
            advantages[t] = advantage

        # 计算回报
        returns = advantages + values[:-1]

        return torch.tensor(returns, dtype=torch.float32), torch.tensor(advantages, dtype=torch.float32)

    def get_batch_data(self):
        """获取批次数据"""
        statesTensor = torch.stack(self.states)
        actionsTensor = torch.tensor(self.actions, dtype=torch.long)
        logProbsTensor = torch.tensor(self.logProbs, dtype=torch.float32)

        return statesTensor, actionsTensor, logProbsTensor

    def __len__(self):
        return len(self.states)



