"""
经验缓冲区相关类
"""
import torch
from collections import namedtuple
from typing import Tuple, Generator
from Config import device

PPOExperience = namedtuple('PPOExperience',
                          ['state', 'action', 'reward', 'value', 'logProb', 'done'])


class PPOBuffer:
    """PPO专用的经验缓冲区 - GPU加速版本"""

    def __init__(self, horizon: int, numActors: int,
                 stateShape: Tuple[int, int, int], device: torch.device = device):
        self.horizon = horizon
        self.numActors = numActors
        self.stateShape = stateShape
        self.device = device

        # 初始化缓冲区 - 直接在GPU上创建
        self.states = torch.zeros((horizon, numActors) + stateShape, device=device)
        self.actions = torch.zeros((horizon, numActors), dtype=torch.long, device=device)
        self.rewards = torch.zeros((horizon, numActors), device=device)
        self.values = torch.zeros((horizon, numActors), device=device)
        self.logProbs = torch.zeros((horizon, numActors), device=device)
        self.dones = torch.zeros((horizon, numActors), dtype=torch.bool, device=device)

        self.advantages = torch.zeros((horizon, numActors), device=device)
        self.returns = torch.zeros((horizon, numActors), device=device)

        self.step = 0

    def push(self, state: torch.Tensor, action: torch.Tensor, reward: torch.Tensor,
             value: torch.Tensor, logProb: torch.Tensor, done: torch.Tensor):
        """添加经验到缓冲区 - 所有张量已经在GPU上"""
        if self.step < self.horizon:
            self.states[self.step] = state
            self.actions[self.step] = action
            self.rewards[self.step] = reward
            self.values[self.step] = value
            self.logProbs[self.step] = logProb
            self.dones[self.step] = done

            self.step += 1

    def computeAdvantagesAndReturns(self, lastValues: torch.Tensor,
                                     gamma: float = 0.99, gaeLambda: float = 0.95):
        """计算优势函数和回报 - 在GPU上执行"""
        advantages = torch.zeros_like(self.rewards)
        lastAdvantage = 0

        for t in reversed(range(self.horizon)):
            if t == self.horizon - 1:
                nextValue = lastValues
                nextNonTerminal = 1.0 - self.dones[t].float()
            else:
                nextValue = self.values[t + 1]
                nextNonTerminal = 1.0 - self.dones[t].float()

            delta = (self.rewards[t] + gamma * nextValue * nextNonTerminal
                    - self.values[t])
            advantages[t] = lastAdvantage = (delta + gamma * gaeLambda
                                            * nextNonTerminal * lastAdvantage)

        self.returns = advantages + self.values

        # 标准化优势函数
        if advantages.std() > 0:  # 避免除零
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        self.advantages = advantages

    def getBatches(self, batchSize: int) -> Generator:
        """生成训练批次 - 数据已经在GPU上，无需转移"""
        # 展平所有数据
        states = self.states.view(-1, *self.stateShape)
        actions = self.actions.view(-1)
        oldLogProbs = self.logProbs.view(-1)
        advantages = self.advantages.view(-1)
        returns = self.returns.view(-1)

        # 随机打乱
        indices = torch.randperm(states.size(0), device=self.device)

        for startIdx in range(0, states.size(0), batchSize):
            endIdx = min(startIdx + batchSize, states.size(0))
            batchIndices = indices[startIdx:endIdx]

            yield (
                states[batchIndices],
                actions[batchIndices],
                oldLogProbs[batchIndices],
                advantages[batchIndices],
                returns[batchIndices]
            )

    def clear(self):
        """清空缓冲区"""
        self.step = 0
        self.advantages.zero_()
        self.returns.zero_()










