"""
PPO智能体和网络结构
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Tuple, Dict, Any
import logging
from Config import PPOConfig, device

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

