"""
A3C智能体网络定义 - ResNet主干版本
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple


class ResidualBlock(nn.Module):
    """残差块，确保尺寸匹配"""

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


class ResNetActorCriticNetwork(nn.Module):
    def __init__(self, inputChannels: int, numActions: int):
        super().__init__()
        self.numActions = numActions
        self.inChannels = 64

        # ResNet主干网络
        # 初始卷积层 - 适配84x84输入
        self.conv1 = nn.Conv2d(inputChannels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # ResNet层 - 适配84x84输入
        self.layer1 = self._makeLayer(64, 64, 2, stride=1)  # 84x84 -> 84x84
        self.layer2 = self._makeLayer(64, 128, 2, stride=2)  # 84x84 -> 42x42
        self.layer3 = self._makeLayer(128, 256, 2, stride=2)  # 42x42 -> 21x21
        self.layer4 = self._makeLayer(256, 512, 2, stride=2)  # 21x21 -> 11x11

        # 自适应平均池化到固定尺寸
        self.adaptiveAvgPool = nn.AdaptiveAvgPool2d((1, 1))

        # 编码器部分 - 保持原有结构但调整输入维度
        self.encoder = nn.Sequential(
            nn.Linear(512, 512),  # 输入维度调整为512
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # 双流架构 - Actor和Critic的独立处理
        # 策略头 (Actor)
        self.policy_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, numActions)
        )

        # 价值头 (Critic)
        self.value_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )

        # 初始化权重
        self._initializeWeights()

    def _makeLayer(self, inChannels: int, outChannels: int, numBlocks: int, stride: int) -> nn.Sequential:
        """创建ResNet层"""
        strides = [stride] + [1] * (numBlocks - 1)
        layers = []

        for currentStride in strides:
            layers.append(ResidualBlock(inChannels, outChannels, currentStride))
            inChannels = outChannels

        return nn.Sequential(*layers)

    def _initializeWeights(self):
        """改进的权重初始化策略"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 0.01)  # 小的正偏置

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 3:
            x = x.unsqueeze(0)

        # 输入安全检查
        if torch.isnan(x).any():
            x = torch.nan_to_num(x, 0.0)

        batch_size = x.size(0)

        # ResNet特征提取
        # 初始卷积
        x = F.relu(self.bn1(self.conv1(x)))

        # ResNet层
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # 全局池化
        x = self.adaptiveAvgPool(x)
        x = x.view(batch_size, -1)

        # 全连接编码
        encoded = self.encoder(x)

        # 双流输出
        policyLogits = self.policy_stream(encoded)
        value = self.value_stream(encoded)

        # 输出安全检查
        if torch.isnan(policyLogits).any():
            policyLogits = torch.nan_to_num(policyLogits, 0.0)
            policyLogits = policyLogits + torch.randn_like(policyLogits) * 0.01

        if torch.isnan(value).any():
            value = torch.nan_to_num(value, 0.0)

        return policyLogits, value

    def getValue(self, state: torch.Tensor) -> torch.Tensor:
        """获取状态价值"""
        _, value = self.forward(state)
        return value

    def getAction(self, state: torch.Tensor) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """根据状态选择动作"""
        with torch.no_grad():
            policyLogits, value = self.forward(state)

            # 安全检查
            if torch.isnan(policyLogits).any():
                policyLogits = torch.ones_like(policyLogits) / self.numActions

            # 添加探索噪声
            noise = torch.randn_like(policyLogits) * 0.01
            policyLogits = policyLogits + noise

            try:
                # 使用带温度参数的softmax
                temperature = 1.1  # 稍微提高探索性
                scaled_logits = policyLogits / temperature
                actionDist = torch.distributions.Categorical(logits=scaled_logits)
                action = actionDist.sample().item()

                # 确保动作在有效范围内
                action = max(0, min(action, self.numActions - 1))

            except Exception as e:
                # 备用随机选择
                action = torch.randint(0, self.numActions, (1,)).item()

            return action, policyLogits, value

    def getActionProbabilities(self, state: torch.Tensor) -> torch.Tensor:
        """获取动作概率分布"""
        with torch.no_grad():
            policyLogits, _ = self.forward(state)
            return F.softmax(policyLogits, dim=-1)

    def getActionWithLogProb(self, state: torch.Tensor) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        """获取动作及其对数概率（用于PPO等算法）"""
        with torch.no_grad():
            policyLogits, value = self.forward(state)

            if torch.isnan(policyLogits).any():
                policyLogits = torch.ones_like(policyLogits) / self.numActions

            policy = F.softmax(policyLogits, dim=-1)
            dist = torch.distributions.Categorical(probs=policy)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            return action.item(), policyLogits, value, log_prob


# 为了保持向后兼容性，保留原类名
ActorCriticNetwork = ResNetActorCriticNetwork