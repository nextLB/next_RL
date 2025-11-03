"""
    关于DQNAgent的模型搭建文件
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random



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


class ResNetDeepQNetwork(nn.Module):
    def __init__(self, inputShape, numActions):
        super(ResNetDeepQNetwork, self).__init__()
        self.conv1 = nn.Conv2d(inputShape[0], 64, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(64)

        # ResNet层 - 适配84x84输入
        self.layer1 = self._makeLayer(64, 64, 2, stride=1)  # 84x84 -> 84x84
        self.layer2 = self._makeLayer(64, 128, 2, stride=2)  # 84x84 -> 42x42
        self.layer3 = self._makeLayer(128, 256, 2, stride=2)  # 42x42 -> 21x21
        self.layer4 = self._makeLayer(256, 512, 2, stride=2)  # 21x21 -> 11x11

        # 自适应平均池化到固定尺寸
        self.adaptiveAvgPool = nn.AdaptiveAvgPool2d((1, 1))

        # 全连接层
        self.fc = nn.Linear(512, numActions)

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
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 初始卷积
        x = F.relu(self.bn1(self.conv1(x)))

        # ResNet层
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # 全局池化和全连接
        x = self.adaptiveAvgPool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x



class V1_2_DQNAgent:
    def __init__(self, config):
        self.name = "V1.2_DQNAgent"
        self.config = config
        self.policyNetwork = ResNetDeepQNetwork(self.config.imageShape, self.config.numActions).to(self.config.device)
        self.targetNetwork = ResNetDeepQNetwork(self.config.imageShape, self.config.numActions).to(self.config.device)

        self._updateTargetNetwork()
        self.targetNetwork.eval()

        # 优化器
        self.optimizer = optim.Adam(
            self.policyNetwork.parameters(),
            lr=self.config.learningRate,
            eps=1e-4
        )

        # 训练状态
        self.stepsCompleted = 0
        self.episodesCompleted = 0


    def _updateTargetNetwork(self) -> None:
        """更新目标网络参数"""
        self.targetNetwork.load_state_dict(self.policyNetwork.state_dict())


    def selectAction(self, state):
        randomValue = random.random()
        epsilon = self._calculateCurrentEpsilon()

        self.stepsCompleted += 1
        if randomValue > epsilon:
            with torch.no_grad():
                qValues = self.policyNetwork(state)
                return qValues.max(1)[1].item()
        else:
            return random.randrange(self.config.numActions)

    def _calculateCurrentEpsilon(self) -> float:
        """计算当前的epsilon值"""
        return self.config.finalEpsilon + (self.config.initialEpsilon - self.config.finalEpsilon) * \
            np.exp(-1.0 * self.stepsCompleted / self.config.epsilonDecaySteps)

    def getCurrentEpsilon(self) -> float:
        """获取当前epsilon值"""
        return self._calculateCurrentEpsilon()

    def optimizeModel(self, experience):
        # 采样
        states, actions, rewards, nextStates, dones = experience.sample(1)

        # 计算当前Q值
        currentQValues = self.policyNetwork(states).gather(1, actions.unsqueeze(1))

        # 计算目标Q值
        with torch.no_grad():
            nextQValues = self.targetNetwork(nextStates).max(1)[0]
            targetQValues = rewards + (self.config.discountFactor * nextQValues * (1 - dones))

        # 计算损失
        loss = F.smooth_l1_loss(currentQValues.squeeze(), targetQValues)

        # 优化模型
        self.optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.policyNetwork.parameters(), 10.0)
        self.optimizer.step()

        # 定期更新目标网络
        if self.stepsCompleted % self.config.targetUpdateFrequency == 0:
            self._updateTargetNetwork()

        return loss.item()



















