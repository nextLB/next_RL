"""
经验缓冲区相关类
"""
import torch
from collections import namedtuple, deque
from typing import Tuple, Generator, List
from Config import device
import numpy as np

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







# LunarLander游戏的缓冲池
class LunarLanderExperienceBuffer:
    """LunarLander PPO专用的经验缓冲区"""
    def __init__(self, bufferSize: int):
        self.bufferSize = bufferSize
        self.buffer = deque(maxlen=self.bufferSize)
        self.onPolicyBuffer = []
        self.offPolicyBuffer = []

    def add_on_policy_experience(self,
                                frame: np.ndarray,
                               state: np.ndarray,
                               action: int,
                               reward: float,
                                nextFrame: np.ndarray,
                               nextState: np.ndarray,
                               done: bool,
                               logProb: float,
                               value: float):
        """添加On-policy经验"""
        experience = {
            'frame': frame,
            'state': state,
            'action': action,
            'reward': reward,
            'nextFrame': nextFrame,
            'nextState': nextState,
            'done': done,
            'logProb': logProb,
            'value': value
        }
        self.onPolicyBuffer.append(experience)

    def add_off_policy_experience(self,
                                  frame: np.ndarray,
                                  state: np.ndarray,
                                  action: int,
                                  reward: float,
                                  nextFrame: np.ndarray,
                                  nextState: np.ndarray,
                                  done: bool):
        """添加Off-policy经验"""
        experience = {
            'frame': frame,
            'state': state,
            'action': action,
            'reward': reward,
            'nextFrame': nextFrame,
            'nextState': nextState,
            'done': done
        }
        if len(self.offPolicyBuffer) < self.bufferSize:
            self.offPolicyBuffer.append(experience)
        else:
            self.offPolicyBuffer.pop(0)
            self.offPolicyBuffer.append(experience)

    def clear_on_policy_buffer(self):
        """清空On-policy缓冲区"""
        self.onPolicyBuffer.clear()

    def compute_advantages_and_returns(self,
                                       values: List[float],
                                       rewards: List[float],
                                       dones: List[bool],
                                       gamma: float = 0.99,
                                       gaeLambda: float = 0.95) -> Tuple[List[float], List[float]]:
        """计算优势函数和回报

        使用广义优势估计(GAE)方法计算优势函数，然后基于优势函数计算回报。
        这是贝尔曼方程的一种推广形式，结合了多步TD误差。

        Args:
            values: 状态价值估计 V(s_t)，长度为T的列表
            rewards: 即时奖励 r_t，长度为T的列表
            dones: 终止标志，表示是否到达终止状态，长度为T的列表
            gamma: 折扣因子，默认0.99
            gaeLambda: GAE权衡参数，平衡偏差和方差，默认0.95

        Returns:
            advantages: 优势函数估计 A(s_t, a_t)，长度为T的列表
            returns: 回报估计 G_t，长度为T的列表

        计算公式:
            1. TD残差: δ_t = r_t + γ * V(s_{t+1}) * (1 - done_t) - V(s_t)
            2. GAE优势: A_t = Σ_{l=0}^{∞} (γλ)^l δ_{t+l}
            3. 回报: G_t = A_t + V(s_t)

        注意: 当done_t为True时，下一状态价值为0(终止状态)
        """
        advantages = []
        returns = []
        gae = 0  # 累计的GAE优势

        # 从后向前计算，便于累计处理
        nextValue = 0  # 下一个状态的价值估计

        for t in reversed(range(len(rewards))):
            # 计算下一状态的价值，如果是终止状态则价值为0
            if t == len(rewards) - 1:
                # 最后一个时间步：如果是终止状态则nextValue=0，否则使用当前状态价值
                nextValue = values[t] * (1 - dones[t])
            else:
                # 非最后一个时间步：使用下一个状态的价值，如果是终止状态则价值为0
                nextValue = values[t + 1] * (1 - dones[t])

            # 计算TD残差 (时序差分误差)
            # δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
            # 当done_t为True时，V(s_{t+1}) = 0
            delta = rewards[t] + gamma * nextValue - values[t]

            # 更新广义优势估计(GAE)
            # GAE: A_t = δ_t + γλ * (1 - done_t) * A_{t+1}
            # 当done_t为True时，后续优势为0
            gae = delta + gamma * gaeLambda * (1 - dones[t]) * gae

            # 将计算的优势插入到列表开头（因为我们是反向遍历）
            advantages.insert(0, gae)

            # 计算回报：G_t = A_t + V(s_t)
            returns.insert(0, gae + values[t])

        return advantages, returns












