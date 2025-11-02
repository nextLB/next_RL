"""
PPO智能体和网络结构
"""
from pyexpat import features

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Tuple, Dict, Any, Optional
import logging

from torch import Tensor
from torch.distributions import Distribution

from Config import PPOConfig, device, LunarLanderConfig
import numpy as np

logger = logging.getLogger(__name__)


class ResidualBlock(nn.Module):
    """修复的残差块，确保尺寸匹配"""

    def __init__(self, inChannels: int, outChannels: int, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(inChannels, outChannels, kernel_size=3,
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(outChannels)
        self.conv2 = nn.Conv2d(outChannels, outChannels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outChannels)

        # 快捷连接 - 确保尺寸匹配
        self.shortcut = nn.Sequential()
        if stride != 1 or inChannels != outChannels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inChannels, outChannels, kernel_size=1,
                         stride=stride, bias=False),
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
    """PPO网络 - 包含策略网络和价值网络"""

    def __init__(self, inputShape: Tuple[int, int, int], numActions: int):
        super().__init__()

        self.inChannels = 64
        self.numActions = numActions

        # 共享的特征提取层
        self.conv1 = nn.Conv2d(inputShape[0], 64, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # ResNet层 - 适配84x84输入
        self.layer1 = self._makeLayer(64, 64, 2, stride=1)  # 84x84 -> 84x84
        self.layer2 = self._makeLayer(64, 128, 2, stride=2)  # 84x84 -> 42x42
        self.layer3 = self._makeLayer(128, 256, 2, stride=2)  # 42x42 -> 21x21
        self.layer4 = self._makeLayer(256, 512, 2, stride=2)  # 21x21 -> 11x11

        # 自适应平均池化到固定尺寸
        self.adaptiveAvgPool = nn.AdaptiveAvgPool2d((1, 1))

        # 策略头
        self.policyHead = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, numActions)
        )

        # 价值头
        self.valueHead = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        # 初始化权重
        self._initializeWeights()

    def _makeLayer(self, inChannels: int, outChannels: int,
                   numBlocks: int, stride: int) -> nn.Sequential:
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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """前向传播 - 返回动作概率、状态价值和动作log概率"""
        # 共享特征提取
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.adaptiveAvgPool(x)
        x = x.view(x.size(0), -1)

        # 策略头
        policyLogits = self.policyHead(x)
        actionProbs = F.softmax(policyLogits, dim=-1)

        # 价值头
        stateValue = self.valueHead(x)

        return actionProbs, stateValue.squeeze(-1), policyLogits

    def getAction(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor,
                                                     torch.Tensor, torch.Tensor]:
        """选择动作并返回相关信息"""
        with torch.no_grad():
            actionProbs, stateValue, policyLogits = self.forward(state)
            dist = torch.distributions.Categorical(actionProbs)
            action = dist.sample()
            logProb = dist.log_prob(action)

            return action, logProb, stateValue, actionProbs


class PPOAgent:
    """PPO智能体"""

    def __init__(self, stateShape: Tuple[int, int, int], numActions: int, config: PPOConfig):
        self.numActions = numActions
        self.stateShape = stateShape
        self.config = config

        self.network = PPONetwork(stateShape, numActions).to(device)

        # 优化器
        if config.useAdam:
            self.optimizer = optim.Adam(
                self.network.parameters(),
                lr=config.learningRate,
                eps=config.adamEpsilon
            )
        else:
            self.optimizer = optim.RMSprop(
                self.network.parameters(),
                lr=config.learningRate,
                eps=config.adamEpsilon
            )

        # 训练状态
        self.stepsCompleted = 0
        self.episodesCompleted = 0

        logger.info(f"PPO智能体初始化完成: 状态形状={stateShape}, 动作数量={numActions}")

    def computeLoss(self, states: torch.Tensor, actions: torch.Tensor, oldLogProbs: torch.Tensor,
                    advantages: torch.Tensor, returns: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """计算PPO损失"""
        actionProbs, values, policyLogits = self.network(states)

        # 策略损失
        dist = torch.distributions.Categorical(actionProbs)
        logProbs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        ratio = torch.exp(logProbs - oldLogProbs)

        # PPO裁剪目标
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.config.clipEpsilon,
                          1.0 + self.config.clipEpsilon) * advantages
        policyLoss = -torch.min(surr1, surr2).mean()

        # 价值损失
        valueLoss = F.mse_loss(values, returns)

        # 总损失
        totalLoss = (policyLoss
                     + self.config.valueLossCoeff * valueLoss
                     - self.config.entropyCoeff * entropy)

        # 额外统计信息
        clipFraction = torch.mean((torch.abs(ratio - 1.0) > self.config.clipEpsilon).float()).item()
        explainedVariance = 1 - F.mse_loss(values, returns) / returns.var()

        stats = {
            'policyLoss': policyLoss.item(),
            'valueLoss': valueLoss.item(),
            'entropy': entropy.item(),
            'clipFraction': clipFraction,
            'explainedVariance': explainedVariance.item(),
            'approxKl': (oldLogProbs - logProbs).mean().item()
        }

        return totalLoss, stats

    def update(self, buffer) -> Dict[str, float]:
        """使用PPO算法更新网络"""
        totalStats = {
            'policyLoss': 0.0,
            'valueLoss': 0.0,
            'entropy': 0.0,
            'clipFraction': 0.0,
            'explainedVariance': 0.0,
            'approxKl': 0.0
        }

        numUpdates = 0

        for epoch in range(self.config.ppoEpochs):
            for batch in buffer.getBatches(self.config.batchSize):
                states, actions, oldLogProbs, advantages, returns = batch

                self.optimizer.zero_grad()
                loss, stats = self.computeLoss(states, actions, oldLogProbs, advantages, returns)
                loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.config.maxGradNorm)
                self.optimizer.step()

                # 累积统计信息
                for key in totalStats:
                    totalStats[key] += stats[key]
                numUpdates += 1

        # 平均统计信息
        if numUpdates > 0:
            for key in totalStats:
                totalStats[key] /= numUpdates

        return totalStats

    def saveCheckpoint(self, filePath: str) -> None:
        """保存模型检查点"""
        try:
            checkpoint = {
                'networkState': self.network.state_dict(),
                'optimizerState': self.optimizer.state_dict(),
                'stepsCompleted': self.stepsCompleted,
                'episodesCompleted': self.episodesCompleted,
                'config': self.config
            }
            torch.save(checkpoint, filePath)
            logger.info(f"PPO模型已保存到: {filePath}")
        except Exception as e:
            logger.error(f"保存PPO模型失败: {e}")
            raise

    def loadCheckpoint(self, filePath: str) -> None:
        """加载模型检查点"""
        try:
            checkpoint = torch.load(filePath, map_location=device)
            self.network.load_state_dict(checkpoint['networkState'])
            self.optimizer.load_state_dict(checkpoint['optimizerState'])
            self.stepsCompleted = checkpoint['stepsCompleted']
            self.episodesCompleted = checkpoint['episodesCompleted']
            logger.info(f"PPO模型已从 {filePath} 加载")
        except Exception as e:
            logger.error(f"加载PPO模型失败: {e}")
            raise

    def getTrainingStatistics(self) -> Dict[str, Any]:
        """获取训练统计信息"""
        from Memory import MemoryManager
        gpuAllocated, gpuCached = MemoryManager.getGpuMemoryUsage()
        systemMemory = MemoryManager.getSystemMemoryUsage()

        return {
            'stepsCompleted': self.stepsCompleted,
            'episodesCompleted': self.episodesCompleted,
            'gpuMemoryAllocated': gpuAllocated,
            'gpuMemoryCached': gpuCached,
            'systemMemory': systemMemory,
        }

    def clearMemory(self):
        """清理内存"""
        from Memory import MemoryManager
        MemoryManager.clearMemory()






class ActorNetwork(nn.Module):
    """Actor网络 - 策略网络"""

    def __init__(self, inputShape: Tuple[int, int, int], numActions: int):
        super().__init__()
        self.numActions = numActions

        # 特征提取层
        self.convLayers = nn.Sequential(
            nn.Conv2d(inputShape[0], 32, 8, 4, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 0),
            nn.ReLU(),
        )

        # 计算卷积层输出尺寸     使用torch.no_grad()上下文管理器，确保在这个代码块中不会计算梯度
        with torch.no_grad():
            # 创建一个全零的虚拟输入张量
            dummyInput = torch.zeros(1, *inputShape)
            # 将虚拟输入通过卷积层序列，得到卷积层输出
            convOutput = self.convLayers(dummyInput)
            # 计算展平后的特征数量
            self.featureSize = convOutput.view(1, -1).size(1)

        # 策略头
        self.policyHead = nn.Sequential(
            nn.Linear(self.featureSize, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, numActions)
        )

        self._initializeWeights()

    def _initializeWeights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.01)
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播 - 返回动作logits"""
        features = self.convLayers(x)
        features = features.view(x.size(0), -1)
        policyLogits = self.policyHead(features)
        return policyLogits

    def get_action_distribution(self, x: torch.Tensor) -> torch.distributions.Distribution:
        """获取动作分布"""
        policyLogits = self.forward(x)
        # 注意在torch.distributions.Categorical方法的内部会自动进行softmax的, 然后返回那个概率最大的那个对象
        # 例如下面这个例子
        """
            policyLogits = torch.tensor([2.0, 1.0, 0.0])  # 动作的"分数"
            distribution = torch.distributions.Categorical(logits=policyLogits)
            
            # Categorical内部自动进行softmax
            probs = F.softmax(policyLogits, dim=-1)  # [0.665, 0.244, 0.090]
            
            # 现在可以使用分布对象：
            action = distribution.sample()      # 采样一个动作（如：0）
            log_prob = distribution.log_prob(action)  # 计算该动作的对数概率
        """
        return torch.distributions.Categorical(logits=policyLogits)


class CriticNetwork(nn.Module):
    """Critic网络 - 价值网络"""

    def __init__(self, inputShape: Tuple[int, int, int]):
        super().__init__()

        # 特征提取层 (与Actor共享结构)
        self.convLayers = nn.Sequential(
            nn.Conv2d(inputShape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        # 计算卷积层输出尺寸
        with torch.no_grad():
            dummyInput = torch.zeros(1, *inputShape)
            convOutput = self.convLayers(dummyInput)
            self.featureSize = convOutput.view(1, -1).size(1)

        # 价值头
        self.valueHead = nn.Sequential(
            nn.Linear(self.featureSize, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        self._initializeWeights()

    def _initializeWeights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播 - 返回状态价值"""
        features = self.convLayers(x)
        features = features.view(x.size(0), -1)
        stateValues = self.valueHead(features)
        return stateValues.squeeze(-1)






class ActorCriticNetwork(nn.Module):
    """Actor-Critic网络 - 结合策略和价值网络"""

    def __init__(self, inputShape: Tuple[int, int, int], numActions: int):
        super().__init__()
        self.actor = ActorNetwork(inputShape, numActions)
        self.critic = CriticNetwork(inputShape)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播 - 返回动作概率和状态价值"""
        # 注意这里调用的是actor的forward方法，还没有经过softmax的
        policyLogits = self.actor(x)
        actionProbs = F.softmax(policyLogits, dim=-1)
        stateValues = self.critic(x)
        return actionProbs, stateValues
    def get_action(self, state: torch.Tensor) -> tuple[Tensor, Tensor, Any, Distribution]:
        """选择动作并返回相关信息"""
        with torch.no_grad():
            actionProbs = self.actor.get_action_distribution(state)
            stateValues = self.critic(state)
            action = actionProbs.sample()
            logProb = actionProbs.log_prob(action)

        return action, logProb, stateValues, actionProbs






class LunarLanderPPOAgent:
    """PPO智能体 - 结合上一版本的PPOAgent的Actor-Critic的思想，此版本还支持on-policy和off-policy训练"""
    def __init__(self, stateShape: Tuple[int, int, int], numActions: int, config: LunarLanderConfig):
        self.numActions = numActions
        self.statesShape = stateShape
        self.config = config

        # 构建与初始化网络
        self.netWork = ActorCriticNetwork(self.statesShape, self.numActions).to(device)

        # 构建与初始化优化器
        self.actorOptimizer = optim.Adam(
            self.netWork.actor.parameters(),
            lr = self.config.actorLearningRate,
            eps=self.config.adamEpsilon
        )
        self.criticOptimizer = optim.Adam(
            self.netWork.critic.parameters(),
            lr = config.criticLearningRate,
            eps=self.config.adamEpsilon
        )

        # 训练状态各变量的初始化
        self.stepCompleted = 0
        self.episodesCompleted = 0
        self.currentEpoch = 0

        logger.info(f"PPO智能体初始化完成: 状态形状={stateShape}, 动作数量={numActions}")


    def compute_actor_loss(self, states: torch.Tensor, actions: torch.Tensor,  oldLogProbs: torch.Tensor, advantages: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """计算Actor损失 (策略损失)"""
        policyLogits = self.netWork.actor(states)
        dist = torch.distributions.Categorical(logits=policyLogits)

        # 新策略的log概率
        newLogProbs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        # 重要性采样比率
        ratio = torch.exp(newLogProbs - oldLogProbs)

        # PPO裁剪目标
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.config.clipEpsilon, 1.0 + self.config.clipEpsilon) * advantages
        policyLoss = -torch.min(surr1, surr2).mean()

        # 熵正则化
        totalPolicyLoss = policyLoss - self.config.entropyCoeff * entropy

        # 统计信息
        clipFraction = torch.mean((torch.abs(ratio - 1.0) > self.config.clipEpsilon).float()).item()
        approxKl = (oldLogProbs - newLogProbs).mean().item()

        stats = {
            'policyLoss': policyLoss.item(),
            'entropy': entropy.item(),
            'clipFraction': clipFraction,
            'approxKl': approxKl,
            'ratioMean': ratio.mean().item()
        }

        return totalPolicyLoss, stats

    def compute_critic_loss(self, states: torch.Tensor, returns: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """计算Critic损失 (价值损失)"""
        values = self.netWork.critic(states)
        valueLoss = F.mse_loss(values, returns)

        # 进行价值裁剪
        valuesClipped = returns + torch.clamp(values - returns, -self.config.clipEpsilon, self.config.clipEpsilon)
        valueLossClipped = F.mse_loss(valuesClipped, returns)
        valueLoss = torch.max(valueLoss, valueLossClipped)

        # 价值正则化
        valueReg = torch.mean(values ** 2)
        valueLoss += self.config.valueRegCoeff * valueReg

        # 统计信息
        explainedVariance = 1 - F.mse_loss(values, returns) / returns.var()

        stats = {
            'valueLoss': valueLoss.item(),
            'explainedVariance': explainedVariance.item(),
            'valueMean': values.mean().item(),
            'returnMean': returns.mean().item()
        }

        return valueLoss, stats

    def on_policy_update(self):
        """On-policy更新"""


    def off_policy_update(self):
        """Off-policy更新"""

    def update(self, onPolicyBuffer, offPolicyBuffer):
        """组合更新 - 支持on-policy和off-policy"""




