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

    # 选择动作并返回动作、对数概率和状态价值
    def select_action(self, state: torch.Tensor):
        with torch.no_grad():
            actionLogits, stateValue = self.network(state)

            # 添加数值稳定性处理
            actionLogits = actionLogits - actionLogits.max(dim=-1, keepdim=True)[0]

            # 创建分布
            actionDistribution = Categorical(logits=actionLogits)
            action = actionDistribution.sample()
            actionLogProb = actionDistribution.log_prob(action)

        return action.item(), actionLogProb.item(), stateValue.item()

    # 评估动作，返回对数概率、熵和状态价值
    def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor):
        actionLogits, stateValues = self.network(states)

        # 添加数值稳定性处理
        actionLogits = actionLogits - actionLogits.max(dim=-1, keepdim=True)[0]

        actionDistribution = Categorical(logits=actionLogits)

        actionLogProbs = actionDistribution.log_prob(actions)
        distEntropy = actionDistribution.entropy()

        return actionLogProbs, distEntropy, stateValues

    # 计算PPO损失
    def compute_loss(self, states, actions, oldLogProbs, advantages, returns, clipEpsilon):
        newLogProbs, entropy, stateValues = self.evaluate_actions(states, actions)
        # 概率比
        ratio = torch.exp(newLogProbs - oldLogProbs)
        # PPO裁剪目标
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clipEpsilon, 1 + clipEpsilon) * advantages
        actorLoss = -torch.min(surr1, surr2).mean()
        # Critic损失
        criticLoss = F.mse_loss(stateValues, returns)
        # 熵正则化
        entropyLoss = -entropy.mean()
        # 总损失
        totalLoss = (actorLoss + self.config.valueCoefficient * criticLoss + self.config.entropyCoefficient * entropyLoss)

        return totalLoss, actorLoss.item(), criticLoss.item(), entropyLoss.item()

    # 更新网络参数
    def update(self, states, actions, oldLogProbs, advantages, returns):
        states = states.detach()
        actions = actions.detach()
        oldLogProbs = oldLogProbs.detach()
        advantages = advantages.detach()
        returns = returns.detach()

        # 多轮PPO更新
        totalActorLoss = 0
        totalCriticLoss = 0
        totalEntropyLoss = 0
        updateCount = 0

        for _ in range(self.config.ppoEpochs):
            # 随机打乱数据
            indices = torch.randperm(states.size(0))

            for start in range(0, states.size(0), self.config.miniBatchSize):
                end = start + self.config.miniBatchSize
                batchIndices = indices[start:end]

                batchStates = states[batchIndices]
                batchActions = actions[batchIndices]
                batchOldLogProbs = oldLogProbs[batchIndices]
                batchAdvantages = advantages[batchIndices]
                batchReturns = returns[batchIndices]

                # 计算损失并更新
                self.optimizer.zero_grad()
                totalLoss, actorLoss, criticLoss, entropyLoss = self.compute_loss(
                    batchStates, batchActions, batchOldLogProbs,
                    batchAdvantages, batchReturns, self.config.clipEpsilon
                )

                totalLoss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
                self.optimizer.step()

                totalActorLoss += actorLoss
                totalCriticLoss += criticLoss
                totalEntropyLoss += entropyLoss
                updateCount += 1

        # 返回平均损失
        avgActorLoss = totalActorLoss / updateCount if updateCount > 0 else 0
        avgCriticLoss = totalCriticLoss / updateCount if updateCount > 0 else 0
        avgEntropyLoss = totalEntropyLoss / updateCount if updateCount > 0 else 0

        return avgActorLoss, avgCriticLoss, avgEntropyLoss

    def get_training_statistics(self) -> dict:
        return {
            'stepsCompleted': self.stepsCompleted,
            'episodesCompleted': self.episodesCompleted,
        }

    def save_check_point(self, filePath: str) -> None:
        checkpoint = {
            'networkState': self.network.state_dict(),
            'optimizerState': self.optimizer.state_dict(),
            'stepsCompleted': self.stepsCompleted,
            'episodesCompleted': self.episodesCompleted,
            'config': self.config
        }
        torch.save(checkpoint, filePath)

    def load_checkpoint(self, filePath: str) -> None:
        checkpoint = torch.load(filePath, map_location=self.config.device)
        self.network.load_state_dict(checkpoint['networkState'])
        self.optimizer.load_state_dict(checkpoint['optimizerState'])
        self.stepsCompleted = checkpoint['stepsCompleted']
        self.episodesCompleted = checkpoint['episodesCompleted']


