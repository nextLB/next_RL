"""
PPO智能体和网络结构
"""
from numpy import floating

from Experience import LunarLanderExperienceBuffer

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Tuple, Dict, Any, Optional
import logging

from torch import Tensor
from torch.distributions import Distribution
import numpy as np
from Config import PPOConfig, device, LunarLanderConfig


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
            nn.Conv2d(inputShape[0], 32, 3, 4, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 1, 0),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, 1, 0),
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, 1, 0),
            nn.ReLU(),
            nn.Conv2d(512, 1024, 3, 1, 0),
            nn.ReLU(),
            nn.Conv2d(1024, 512, 3, 1, 0),
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
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
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
            nn.Conv2d(inputShape[0], 32, 3, 4, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 1, 0),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, 1, 0),
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, 1, 0),
            nn.ReLU(),
            nn.Conv2d(512, 1024, 3, 1, 0),
            nn.ReLU(),
            nn.Conv2d(1024, 512, 3, 1, 0),
            nn.ReLU(),
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
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
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
        self.targetNetwork = ActorCriticNetwork(self.statesShape, self.numActions).to(device)
        self.targetNetwork.load_state_dict(self.netWork.state_dict())

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

        # 价值裁剪
        valuesClipped = returns + torch.clamp(values - returns, -self.config.clipEpsilon, self.config.clipEpsilon)
        valueLoss = F.mse_loss(values, returns)
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

    def on_policy_update(self, experienceBuffer) -> dict[Any, Any] | dict[Any, floating[Any]]:
        """On-policy更新 - 使用当前策略收集的数据"""
        if len(experienceBuffer.onPolicyBuffer) == 0:
            return {}

        # 准备数据
        frames = torch.stack([torch.from_numpy(exp['frame']).float().to(device)
                              for exp in experienceBuffer.onPolicyBuffer]).unsqueeze(1)
        actions = torch.tensor([exp['action'] for exp in experienceBuffer.onPolicyBuffer],
                               dtype=torch.long, device=device)
        oldLogProbs = torch.tensor([exp['logProb'] for exp in experienceBuffer.onPolicyBuffer],
                                   dtype=torch.float, device=device)
        advantages = torch.tensor([exp['advantage'] for exp in experienceBuffer.onPolicyBuffer],
                                  dtype=torch.float, device=device)
        returns = torch.tensor([exp['return'] for exp in experienceBuffer.onPolicyBuffer],
                               dtype=torch.float, device=device)

        # 标准化优势函数
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        totalStats = {}

        # 多轮更新
        for epoch in range(self.config.onPolicyEpochs):
            # 随机打乱数据
            indices = torch.randperm(len(frames))

            # 小批量更新
            for start in range(0, len(frames), self.config.miniUpdateSize):
                end = start + self.config.miniUpdateSize
                batchIndices = indices[start:end]

                batchFrames = frames[batchIndices]
                batchActions = actions[batchIndices]
                batchOldLogProbs = oldLogProbs[batchIndices]
                batchAdvantages = advantages[batchIndices]
                batchReturns = returns[batchIndices]

                # 计算Actor损失
                actorLoss, actorStats = self.compute_actor_loss(
                    batchFrames, batchActions, batchOldLogProbs, batchAdvantages
                )

                # 计算Critic损失
                criticLoss, criticStats = self.compute_critic_loss(batchFrames, batchReturns)

                # 总损失
                totalLoss = actorLoss + self.config.valueLossCoeff * criticLoss

                # 反向传播
                self.actorOptimizer.zero_grad()
                self.criticOptimizer.zero_grad()
                totalLoss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.netWork.actor.parameters(), self.config.maxGradNorm)
                torch.nn.utils.clip_grad_norm_(self.netWork.critic.parameters(), self.config.maxGradNorm)

                # 更新参数
                self.actorOptimizer.step()
                self.criticOptimizer.step()

                # 收集统计信息
                for key, value in {**actorStats, **criticStats}.items():
                    if key not in totalStats:
                        totalStats[key] = []
                    totalStats[key].append(value)

        # 计算平均统计信息
        avgStats = {key: np.mean(values) for key, values in totalStats.items()}
        avgStats['on_policy_update_epochs'] = self.config.onPolicyEpochs

        # 清空on-policy缓冲区
        experienceBuffer.clear_on_policy_buffer()

        return avgStats

    def off_policy_update(self, experienceBuffer) -> Dict[str, float]:
        """Off-policy更新 - 使用经验回放缓冲区的数据"""
        if len(experienceBuffer.offPolicyBuffer) < self.config.offPolicyEpochs:
            return {}

        # 随机采样
        indices = np.random.choice(len(experienceBuffer.offPolicyBuffer),
                                   self.config.offPolicyEpochs, replace=False)

        batch = [experienceBuffer.offPolicyBuffer[i] for i in indices]


        # 准备数据
        frames = torch.stack([torch.from_numpy(exp['frame']).float().to(device)
                              for exp in batch]).unsqueeze(1)
        actions = torch.tensor([exp['action'] for exp in batch],
                               dtype=torch.long, device=device)
        rewards = torch.tensor([exp['reward'] for exp in batch],
                               dtype=torch.float, device=device)
        nextFrames = torch.stack([torch.from_numpy(exp['nextFrame']).float().to(device)
                                  for exp in batch]).unsqueeze(1)
        dones = torch.tensor([exp['done'] for exp in batch],
                             dtype=torch.bool, device=device)


        # 计算目标Q值
        with torch.no_grad():
            nextValues = self.targetNetwork.critic(nextFrames)
            targetReturns = rewards + self.config.gamma * nextValues * (~dones).float()

        # 计算当前值
        currentValues = self.netWork.critic(frames)

        # TD误差
        tdErrors = targetReturns - currentValues

        # 计算Critic损失
        criticLoss = F.mse_loss(currentValues, targetReturns)

        # 计算Actor损失（使用TD误差作为优势函数的近似）
        policyLogits = self.netWork.actor(frames)
        dist = torch.distributions.Categorical(logits=policyLogits)
        logProbs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        # 策略梯度损失
        policyLoss = -(logProbs * tdErrors.detach()).mean()

        # 总损失（包含熵正则化）
        totalLoss = (policyLoss +
                     self.config.valueLossCoeff * criticLoss -
                     self.config.entropyCoeff * entropy)

        # 反向传播
        self.actorOptimizer.zero_grad()
        self.criticOptimizer.zero_grad()
        totalLoss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.netWork.actor.parameters(), self.config.maxGradNorm)
        torch.nn.utils.clip_grad_norm_(self.netWork.critic.parameters(), self.config.maxGradNorm)

        # 更新参数
        self.actorOptimizer.step()
        self.criticOptimizer.step()

        # 目标网络软更新
        self.soft_update_target_network()

        # 收集统计信息
        stats = {
            'off_policy_policy_loss': policyLoss.item(),
            'off_policy_value_loss': criticLoss.item(),
            'off_policy_entropy': entropy.item(),
            'off_policy_td_error_mean': tdErrors.mean().item(),
            'off_policy_batch_size': self.config.offPolicyEpochs
        }

        return stats

    def soft_update_target_network(self):
        """软更新目标网络"""
        for targetParam, param in zip(self.targetNetwork.parameters(), self.netWork.parameters()):
            targetParam.data.copy_(
                self.config.tau * param.data + (1.0 - self.config.tau) * targetParam.data
            )

    def update(self, experienceBuffer) -> Dict[str, float]:
        """组合更新 - 支持on-policy和off-policy"""
        stats = {}

        # On-policy更新
        if len(experienceBuffer.onPolicyBuffer) >= self.config.miniUpdateSize:
            onPolicyStats = self.on_policy_update(experienceBuffer)
            stats.update(onPolicyStats)

        # Off-policy更新（每隔一定步数执行一次）
        if (self.stepCompleted % self.config.offPolicyUpdateFreq == 0 and
                len(experienceBuffer.offPolicyBuffer) >= self.config.offPolicyEpochs):
            offPolicyStats = self.off_policy_update(experienceBuffer)
            stats.update(offPolicyStats)

        self.stepCompleted += 1

        return stats



