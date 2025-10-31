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

        # 增强的卷积层结构
        self.conv1 = nn.Conv2d(inputChannels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1)  # 新增层

        # 计算卷积层输出尺寸
        convOutputSize = self._getConvOutputSize(inputChannels)

        # 增强的全连接层
        self.fc1 = nn.Linear(convOutputSize, 512)
        self.fc2 = nn.Linear(512, 256)  # 新增层

        # 策略头 (Actor)
        self.policyHead = nn.Linear(256, numActions)

        # 价值头 (Critic)
        self.valueHead = nn.Linear(256, 1)

        # 添加dropout防止过拟合
        self.dropout = nn.Dropout(0.2)

    def _getConvOutputSize(self, inputChannels: int) -> int:
        with torch.no_grad():
            x = torch.zeros(1, inputChannels, 84, 84)
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            x = F.relu(self.conv4(x))  # 新增层
            return int(np.prod(x.shape[1:]))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 3:
            x = x.unsqueeze(0)
        # 检查输入是否为NaN
        if torch.isnan(x).any():
            # 用零替换NaN
            x = torch.nan_to_num(x, 0.0)

        # 卷积层
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))  # 新增层

        # 展平
        x = x.view(x.size(0), -1)

        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # 添加dropout
        x = F.relu(self.fc2(x))

        # 策略和价值输出
        policyLogits = self.policyHead(x)
        value = self.valueHead(x)

        # 检查输出是否为NaN
        if torch.isnan(policyLogits).any():
            # 用小的随机值替换NaN
            policyLogits = torch.nan_to_num(policyLogits, 0.0)
            # 添加小的噪声避免完全相同的值
            policyLogits = policyLogits + torch.randn_like(policyLogits) * 0.01

        if torch.isnan(value).any():
            value = torch.nan_to_num(value, 0.0)

        return policyLogits, value

    def _initializeWeights(self):
        """更稳定的权重初始化"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain('relu'))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
    def getValue(self, state: torch.Tensor) -> torch.Tensor:
        """获取状态价值"""
        _, value = self.forward(state)
        return value

    def getAction(self, state: torch.Tensor) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """根据状态选择动作"""
        with torch.no_grad():
            policyLogits, value = self.forward(state)

            # 安全检查：如果logits包含NaN，使用均匀分布
            if torch.isnan(policyLogits).any():
                policyLogits = torch.ones_like(policyLogits) / policyLogits.size(-1)


            policy = F.softmax(policyLogits, dim=-1)

            # 使用多项式采样选择动作
            try:
                actionDist = torch.distributions.Categorical(logits=policyLogits)
                action = actionDist.sample().item()
            except Exception as e:
                action = torch.randint(0, self.numActions, (1,)).item()

            return action, policyLogits, value

    def getActionProbabilities(self, state: torch.Tensor) -> torch.Tensor:
        """获取动作概率分布（用于调试）"""
        with torch.no_grad():
            policyLogits, _ = self.forward(state)
            return F.softmax(policyLogits, dim=-1)