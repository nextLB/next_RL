"""
    PPO智能体模型搭建文件
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical



class ResidualBlock(nn.Module):

    def __init__(self, inChannels: int, outChannels: int, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(inChannels, outChannels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(outChannels)
        self.conv2 = nn.Conv2d(outChannels, outChannels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outChannels)

        # 快捷连接 - 确保尺寸匹配
        self.shortcut = nn.Sequential()
        if stride != 1 or inChannels != outChannels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inChannels, outChannels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outChannels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        residual = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # 确保残差连接尺寸匹配
        residual = self.shortcut(residual)
        out += residual
        out = F.relu(out)

        return out


class PPONetwork(nn.Module):
    def __init__(self, inputShape, numActions):
        super(PPONetwork, self).__init__()
        self.conv1 = nn.Conv2d(inputShape[0], 64, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(64)

        # ResNet层
        self.layer1 = self._makeLayer(64, 64, 2, stride=1)
        self.layer2 = self._makeLayer(64, 128, 2, stride=2)
        self.layer3 = self._makeLayer(128, 256, 2, stride=2)
        self.layer4 = self._makeLayer(256, 512, 2, stride=2)

        # 自适应平均池化
        self.adaptiveAvgPool = nn.AdaptiveAvgPool2d((1, 1))

        # Actor和Critic共享的特征提取层
        self.featureSize = 512

        # Actor网络 (策略网络)
        self.actor = nn.Sequential(
            nn.Linear(self.featureSize, 256),
            nn.ReLU(),
            nn.Linear(256, numActions)
        )

        # Critic网络 (价值网络)
        self.critic = nn.Sequential(
            nn.Linear(self.featureSize, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        self._initializeWeights()

    def _makeLayer(self, inChannels: int, outChannels: int, numBlocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (numBlocks - 1)
        layers = []
        for currentStride in strides:
            layers.append(ResidualBlock(inChannels, outChannels, currentStride))
            inChannels = outChannels
        return nn.Sequential(*layers)

    def _initializeWeights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> tuple:
        # 特征提取
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.adaptiveAvgPool(x)
        x = x.view(x.size(0), -1)

        # Actor和Critic输出
        actionLogits = self.actor(x)
        stateValue = self.critic(x)

        return actionLogits, stateValue.squeeze(-1)




class V1_2_PPOAgent:
    def __init__(self, config):
        self.name = "V1.2_PPOAgent"
        self.config = config
        self.network = PPONetwork(self.config.imageShape, self.config.numActions).to(self.config.device)
        # 优化器
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=self.config.learningRate,
            eps=1e-5
        )

        # 训练状态
        self.stepsCompleted = 0
        self.episodesCompleted = 0




