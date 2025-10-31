"""
DQN智能体模块
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from Config import TrainingConfig
from Experience import ExperienceReplayBuffer
from Memory import MemoryManager
import logging


logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ResidualBlock(nn.Module):
    """修复的残差块，确保尺寸匹配"""

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
    """ResNet DQN网络"""

    def __init__(self, inputShape: tuple, numActions: int):
        super().__init__()

        self.inChannels = 64

        # 初始卷积层 - 适配84x84输入
        self.conv1 = nn.Conv2d(inputShape[0], 64, kernel_size=3, stride=1, padding=1, bias=False)
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


class DeepQNAgent:
    """DQN智能体 - GPU加速版本"""

    def __init__(self, stateShape: tuple, numActions: int, config: TrainingConfig):
        self.numActions = numActions
        self.stateShape = stateShape
        self.config = config

        self.policyNetwork = ResNetDeepQNetwork(stateShape, numActions).to(device)
        self.targetNetwork = ResNetDeepQNetwork(stateShape, numActions).to(device)
        logger.info("使用完整的ResNet架构")

        self._updateTargetNetwork()
        self.targetNetwork.eval()

        # 优化器
        self.optimizer = optim.Adam(
            self.policyNetwork.parameters(),
            lr=config.learningRate,
            eps=1e-4
        )

        # 经验回放缓冲区 - GPU加速版本
        self.memory = ExperienceReplayBuffer(capacity=config.replayBufferCapacity)

        # 训练状态
        self.stepsCompleted = 0
        self.episodesCompleted = 0

        logger.info(f"GPU加速DQN智能体初始化完成: 状态形状={stateShape}, 动作数量={numActions}")

    def selectAction(self, state: torch.Tensor, training: bool = True) -> int:
        """根据当前状态选择动作"""
        try:
            randomValue = random.random()

            if training:
                epsilon = self._calculateCurrentEpsilon()
            else:
                epsilon = 0.01

            self.stepsCompleted += 1

            if randomValue > epsilon:
                with torch.no_grad():
                    # 确保输入尺寸正确并在GPU上
                    if len(state.shape) == 3:
                        state = state.unsqueeze(0)  # 添加batch维度

                    stateDevice = state.to(device) if not state.is_cuda else state
                    qValues = self.policyNetwork(stateDevice)
                    return qValues.max(1)[1].item()
            else:
                return random.randrange(self.numActions)
        except Exception as e:
            logger.error(f"选择动作失败: {e}")
            return random.randrange(self.numActions)

    def optimizeModel(self) -> float:
        """执行一次模型优化 - GPU加速版本"""
        if (len(self.memory) < self.config.batchSize or
                self.stepsCompleted < self.config.learningStartSteps or
                self.stepsCompleted % self.config.learningUpdateFrequency != 0):
            return 0.0

        from Log import memoryMonitor

        with memoryMonitor("模型优化"):
            try:
                # 采样 - 状态已经在GPU上
                states, actions, rewards, nextStates, dones = self.memory.sample(self.config.batchSize)

                # 确保输入尺寸正确
                if len(states.shape) == 3:
                    states = states.unsqueeze(1)
                if len(nextStates.shape) == 3:
                    nextStates = nextStates.unsqueeze(1)

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
                    logger.debug(f"更新目标网络，步骤: {self.stepsCompleted}")

                return loss.item()

            except torch.cuda.OutOfMemoryError as e:
                logger.error(f"优化模型时GPU显存不足: {e}")
                # 紧急回退到CPU内存
                self.memory.clearGpuMemory()
                return 0.0

            except Exception as e:
                logger.error(f"模型优化失败: {e}")
                return 0.0

    def _calculateCurrentEpsilon(self) -> float:
        """计算当前的epsilon值"""
        return self.config.finalEpsilon + (self.config.initialEpsilon - self.config.finalEpsilon) * \
            np.exp(-1.0 * self.stepsCompleted / self.config.epsilonDecaySteps)

    def _updateTargetNetwork(self) -> None:
        """更新目标网络参数"""
        self.targetNetwork.load_state_dict(self.policyNetwork.state_dict())

    def getCurrentEpsilon(self) -> float:
        """获取当前epsilon值"""
        return self._calculateCurrentEpsilon()

    def saveCheckpoint(self, filePath: str) -> None:
        """保存模型检查点"""
        try:
            checkpoint = {
                'policyNetworkState': self.policyNetwork.state_dict(),
                'targetNetworkState': self.targetNetwork.state_dict(),
                'optimizerState': self.optimizer.state_dict(),
                'stepsCompleted': self.stepsCompleted,
                'episodesCompleted': self.episodesCompleted,
                'config': self.config
            }
            torch.save(checkpoint, filePath)
            logger.info(f"模型已保存到: {filePath}")
        except Exception as e:
            logger.error(f"保存模型失败: {e}")
            raise

    def loadCheckpoint(self, filePath: str) -> None:
        """加载模型检查点"""
        try:
            checkpoint = torch.load(filePath, map_location=device)
            self.policyNetwork.load_state_dict(checkpoint['policyNetworkState'])
            self.targetNetwork.load_state_dict(checkpoint['targetNetworkState'])
            self.optimizer.load_state_dict(checkpoint['optimizerState'])
            self.stepsCompleted = checkpoint['stepsCompleted']
            self.episodesCompleted = checkpoint['episodesCompleted']
            logger.info(f"模型已从 {filePath} 加载")
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            raise

    def getTrainingStatistics(self) -> dict:
        """获取训练统计信息"""
        gpuAllocated, gpuCached = MemoryManager.getGpuMemoryUsage()
        systemMemory = MemoryManager.getSystemMemoryUsage()

        return {
            'stepsCompleted': self.stepsCompleted,
            'episodesCompleted': self.episodesCompleted,
            'currentEpsilon': self.getCurrentEpsilon(),
            'memoryUsage': self.memory.getUsagePercentage(),
            'gpuMemoryAllocated': gpuAllocated,
            'gpuMemoryCached': gpuCached,
            'systemMemory': systemMemory,
        }

    def clearMemory(self):
        """清理内存"""
        MemoryManager.clearMemory()

