"""
A3C智能体网络定义
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple

class ActorCriticNetwork(nn.Module):
    def __init__(self, inputChannels: int, numActions: int):
        super().__init__()
        self.numActions = numActions

        # 修正卷积层结构 - 使用更标准的架构
        self.conv1 = nn.Conv2d(inputChannels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # 计算卷积层输出尺寸
        convOutputSize = self._getConvOutputSize(inputChannels)

        # 增加网络容量
        self.fc = nn.Linear(convOutputSize, 512)

        # 策略头 (Actor)
        self.policyHead = nn.Linear(512, numActions)

        # 价值头 (Critic)
        self.valueHead = nn.Linear(512, 1)

    def _getConvOutputSize(self, inputChannels: int) -> int:
        """计算卷积层输出尺寸"""
        with torch.no_grad():
            x = torch.zeros(1, inputChannels, 84, 84)
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            return x.view(1, -1).size(1)

    def _initializeWeights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        # 卷积层
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))

        # 策略和价值输出
        policyLogits = self.policyHead(x)
        value = self.valueHead(x)

        return policyLogits, value

    def getAction(self, state: torch.Tensor) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """根据状态选择动作"""
        with torch.no_grad():
            policyLogits, value = self.forward(state.unsqueeze(0))
            policy = F.softmax(policyLogits, dim=-1)
            action = policy.multinomial(1).item()
            return action, policyLogits, value

