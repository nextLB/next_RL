"""
    依照本目录的paper目录下的DQNNaturePaper篇文章进行训练的，与原论文在架构方面有一些改动的
"""

# V1.0  2025.10.28      --- by next, 初步实现了使用Renet深度学习模型架构的DQN强化学习模型的搭建与训练等


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque, namedtuple
import gymnasium as gym
import matplotlib.pyplot as plt
from PIL import Image
import logging
from typing import Tuple, List, Dict, Any, Optional
import gc
import psutil
import os
import time
from dataclasses import dataclass
from contextlib import contextmanager
import json
from datetime import datetime


# 配置类
@dataclass
class TrainingConfig:
    """训练配置参数"""
    environmentName: str = "PongNoFrameskip-v4"
    learningRate: float = 0.00025
    discountFactor: float = 0.99
    batchSize: int = 32
    replayBufferCapacity: int = 3000
    targetUpdateFrequency: int = 1000
    learningStartSteps: int = 1000
    learningUpdateFrequency: int = 4
    initialEpsilon: float = 1.0
    finalEpsilon: float = 0.1
    epsilonDecaySteps: int = 50000
    frameSkip: int = 4
    screenSize: int = 84
    useSimpleResNet: bool = True
    trainingEpisodes: int = 1000
    targetAverageReward: float = 15.0
    saveImages: bool = True
    imageSaveDir: str = "recordedPongEpisodes"
    numEpisodesToRecord: int = 3


# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./log/dqn_resnet_gpu_memory.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# 上下文管理器用于内存监控
@contextmanager
def memoryMonitor(operationName: str):
    """监控内存使用的上下文管理器"""
    startTime = time.time()
    gpuMemoryBefore = MemoryManager.getGpuMemoryUsage()
    systemMemoryBefore = MemoryManager.getSystemMemoryUsage()

    try:
        yield
    finally:
        endTime = time.time()
        gpuMemoryAfter = MemoryManager.getGpuMemoryUsage()
        systemMemoryAfter = MemoryManager.getSystemMemoryUsage()

        logger.debug(
            f"{operationName} - "
            f"耗时: {endTime - startTime:.3f}s | "
            f"GPU内存变化: {gpuMemoryAfter[0] - gpuMemoryBefore[0]:+.2f}GB | "
            f"系统内存变化: {systemMemoryAfter - systemMemoryBefore:+.2f}GB"
        )


# 经验回放缓冲区
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'nextState', 'done'])


class ExperienceReplayBuffer:
    """GPU加速的经验回放缓冲区"""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
        self.gpuMemoryWarningIssued = False

    def push(self, state: torch.Tensor, action: int, reward: float, nextState: torch.Tensor, done: bool) -> None:
        """添加经验到缓冲区"""
        with memoryMonitor("经验回放缓冲区添加"):
            try:
                # 确保状态在GPU上以加速训练
                stateGpu = state.to(device) if not state.is_cuda else state
                nextStateGpu = nextState.to(device) if not nextState.is_cuda else nextState

                self.buffer.append(Experience(stateGpu, action, reward, nextStateGpu, done))

                # 内存使用监控
                if len(self.buffer) % 1000 == 0:
                    self._checkMemoryUsage()

            except torch.cuda.OutOfMemoryError as e:
                logger.warning(f"GPU显存不足，回退到CPU存储: {e}")
                # 回退到CPU存储
                stateCpu = state.cpu() if state.is_cuda else state
                nextStateCpu = nextState.cpu() if nextState.is_cuda else nextState
                self.buffer.append(Experience(stateCpu, action, reward, nextStateCpu, done))

            except Exception as e:
                logger.error(f"添加经验到缓冲区失败: {e}")

    def sample(self, batchSize: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """从缓冲区中随机采样一批经验"""
        if len(self.buffer) < batchSize:
            raise ValueError(f"缓冲区中经验数量不足: {len(self.buffer)} < {batchSize}")

        with memoryMonitor("经验回放缓冲区采样"):
            try:
                experiences = random.sample(self.buffer, batchSize)

                # 使用列表推导式提高效率
                states = torch.stack([exp.state for exp in experiences])
                actions = torch.tensor([exp.action for exp in experiences], dtype=torch.long, device=device)
                rewards = torch.tensor([exp.reward for exp in experiences], dtype=torch.float32, device=device)
                nextStates = torch.stack([exp.nextState for exp in experiences])
                dones = torch.tensor([exp.done for exp in experiences], dtype=torch.float32, device=device)

                return states, actions, rewards, nextStates, dones

            except torch.cuda.OutOfMemoryError as e:
                logger.error(f"采样时GPU显存不足: {e}")
                self._emergencyMemoryCleanup()
                raise

            except Exception as e:
                logger.error(f"采样经验失败: {e}")
                raise

    def _checkMemoryUsage(self):
        """检查GPU内存使用情况"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024 ** 3
            if allocated > 4.0 and not self.gpuMemoryWarningIssued:
                logger.warning(f"GPU内存使用较高: {allocated:.2f} GB")
                self.gpuMemoryWarningIssued = True

    def _emergencyMemoryCleanup(self):
        """紧急内存清理"""
        logger.warning("执行紧急内存清理...")
        MemoryManager.clearMemory()

    def __len__(self) -> int:
        return len(self.buffer)

    def getUsagePercentage(self) -> float:
        """返回缓冲区的使用百分比"""
        return len(self.buffer) / self.capacity * 100

    def clearGpuMemory(self):
        """清理GPU内存中的状态张量"""
        logger.info("清理GPU内存中的经验缓冲区...")
        # 使用列表推导式提高效率
        cpuBuffer = deque(
            [
                Experience(
                    exp.state.cpu() if exp.state.is_cuda else exp.state,
                    exp.action,
                    exp.reward,
                    exp.nextState.cpu() if exp.nextState.is_cuda else exp.nextState,
                    exp.done
                )
                for exp in self.buffer
            ],
            maxlen=self.capacity
        )

        self.buffer = cpuBuffer
        MemoryManager.clearMemory()
        logger.info("GPU内存清理完成")


class AtariEnvironmentPreprocessor:
    """Atari环境预处理包装器"""

    def __init__(self, environment, frameSkip: int = 4, screenSize: int = 84):
        self.environment = environment
        self.frameSkip = frameSkip
        self.screenSize = screenSize
        self.frameBuffer = deque(maxlen=4)

    def reset(self) -> Tuple[torch.Tensor, dict]:
        """重置环境并返回预处理后的初始状态"""
        try:
            state, info = self.environment.reset()
            processedState = self._preprocessFrame(state)

            # 用相同的帧填充初始缓冲区
            self.frameBuffer.extend([processedState] * 4)

            stateTensor = torch.tensor(np.stack(self.frameBuffer), dtype=torch.float32)
            return stateTensor, info
        except Exception as e:
            logger.error(f"重置环境失败: {e}")
            raise

    def step(self, action: int) -> Tuple[torch.Tensor, float, bool, dict]:
        """执行动作并返回预处理后的结果"""
        try:
            totalReward = 0.0
            terminated = False
            truncated = False
            info = {}

            # 使用帧跳过提高效率
            for _ in range(self.frameSkip):
                nextState, reward, terminated, truncated, stepInfo = self.environment.step(action)
                totalReward += reward
                info.update(stepInfo)

                if terminated or truncated:
                    break

            done = terminated or truncated
            processedNextState = self._preprocessFrame(nextState)
            self.frameBuffer.append(processedNextState)

            nextStateTensor = torch.tensor(np.stack(self.frameBuffer), dtype=torch.float32)
            return nextStateTensor, totalReward, done, info
        except Exception as e:
            logger.error(f"执行动作失败: {e}")
            raise

    def _preprocessFrame(self, frame: np.ndarray) -> np.ndarray:
        """预处理帧：灰度化、调整大小、归一化"""
        try:
            # 转换为灰度图
            if len(frame.shape) == 3:
                frame = np.mean(frame, axis=2)  # 使用numpy提高效率

            # 调整大小
            img = Image.fromarray(frame.astype(np.uint8))
            img = img.resize((self.screenSize, self.screenSize), Image.BILINEAR)
            frame = np.array(img)

            # 归一化到 [0, 1]
            frame = frame.astype(np.float32) / 255.0

            return frame
        except Exception as e:
            logger.error(f"预处理帧失败: {e}")
            raise

    @property
    def actionSpace(self):
        return self.environment.action_space

    @property
    def observationSpace(self):
        return self.environment.observation_space

    def close(self) -> None:
        """关闭环境"""
        self.environment.close()


class EnvironmentRecorder:
    """环境记录器，用于保存原始环境数据为PNG图片"""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.environment = gym.make(config.environmentName, render_mode='rgb_array')
        self.setupSaveDirectory()
        self.episodeMetadata = []

    def setupSaveDirectory(self):
        """创建保存目录和子目录"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.saveDir = f"{self.config.imageSaveDir}_{timestamp}"
        os.makedirs(self.saveDir, exist_ok=True)

        # 创建episodes子目录
        self.episodesDir = os.path.join(self.saveDir, "episodes")
        os.makedirs(self.episodesDir, exist_ok=True)

        logger.info(f"环境数据将保存到: {self.saveDir}")

    def preprocessFrame(self, frame):
        """预处理帧 - 这里保存原始帧，不进行灰度化等处理"""
        return frame

    def saveFrameWithInfo(self, frame, episode, step, action, reward, terminated, truncated, info):
        """保存帧图像和相关信息"""
        try:
            # 预处理帧
            processedFrame = self.preprocessFrame(frame)

            # 确保数据格式正确
            if processedFrame.dtype != np.uint8:
                processedFrame = np.clip(processedFrame * 255, 0, 255).astype(np.uint8)

            # 创建PIL图像
            img = Image.fromarray(processedFrame)

            # 构建文件名和路径
            episodeDir = os.path.join(self.episodesDir, f"episode_{episode:04d}")
            os.makedirs(episodeDir, exist_ok=True)

            filename = f"step_{step:04d}_action_{action}_reward_{reward:.1f}.png"
            filepath = os.path.join(episodeDir, filename)

            # 保存图像
            img.save(filepath)

            # 保存帧信息到元数据
            frameInfo = {
                'episode': episode,
                'step': step,
                'action': int(action),
                'reward': float(reward),
                'terminated': bool(terminated),
                'truncated': bool(truncated),
                'filename': filename,
                'timestamp': datetime.now().isoformat()
            }
            if info:
                frameInfo.update(info)  # 添加环境返回的info

            return frameInfo
        except Exception as e:
            logger.error(f"保存帧信息失败: {e}")
            return None

    def recordEpisode(self, episodeNum, maxSteps=1000):
        """记录一个完整的episode"""
        try:
            state, info = self.environment.reset()
            episodeFrames = []
            totalReward = 0

            episodeDir = os.path.join(self.episodesDir, f"episode_{episodeNum:04d}")
            os.makedirs(episodeDir, exist_ok=True)

            for step in range(maxSteps):
                # 随机动作
                action = self.environment.action_space.sample()

                nextState, reward, terminated, truncated, info = self.environment.step(action)

                # 保存当前帧和相关信息
                frameInfo = self.saveFrameWithInfo(
                    state, episodeNum, step, action, reward, terminated, truncated, info
                )
                if frameInfo:
                    episodeFrames.append(frameInfo)

                # 更新状态
                state = nextState
                totalReward += reward

                # 检查是否结束
                if terminated or truncated:
                    break

            # 保存episode的元数据
            episodeMetadata = {
                'episodeNumber': episodeNum,
                'totalReward': totalReward,
                'totalSteps': step + 1,
                'frames': episodeFrames,
                'environment': self.config.environmentName,
                'timestamp': datetime.now().isoformat()
            }

            metadataFile = os.path.join(episodeDir, "metadata.json")
            with open(metadataFile, 'w') as f:
                json.dump(episodeMetadata, f, indent=2)

            self.episodeMetadata.append(episodeMetadata)
            logger.info(f"Episode {episodeNum}: {step + 1} 步, 总奖励: {totalReward}")

            return totalReward
        except Exception as e:
            logger.error(f"记录episode失败: {e}")
            return 0.0

    def saveSummary(self):
        """保存所有episode的摘要信息"""
        try:
            summary = {
                'totalEpisodes': len(self.episodeMetadata),
                'environment': self.config.environmentName,
                'config': self.config.__dict__,
                'episodes': self.episodeMetadata,
                'recordedAt': datetime.now().isoformat()
            }

            summaryFile = os.path.join(self.saveDir, "recordingSummary.json")
            with open(summaryFile, 'w') as f:
                json.dump(summary, f, indent=2)

            logger.info(f"摘要已保存到: {summaryFile}")
        except Exception as e:
            logger.error(f"保存摘要失败: {e}")

    def close(self):
        """关闭环境并保存摘要"""
        self.saveSummary()
        self.environment.close()


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

    def __init__(self, inputShape: Tuple[int, int, int], numActions: int):
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


class MemoryManager:
    """内存管理器，用于监控和优化内存使用"""

    @staticmethod
    def getGpuMemoryUsage() -> Tuple[float, float]:
        """获取GPU内存使用情况"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024 ** 3  # GB
            cached = torch.cuda.memory_reserved() / 1024 ** 3  # GB
            return allocated, cached
        return 0.0, 0.0

    @staticmethod
    def getSystemMemoryUsage() -> float:
        """获取系统内存使用情况"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 ** 3  # GB

    @staticmethod
    def clearMemory():
        """清理内存"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


class DeepQNAgent:
    """DQN智能体 - GPU加速版本"""

    def __init__(self, stateShape: Tuple[int, int, int], numActions: int, config: TrainingConfig):
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

    def getTrainingStatistics(self) -> Dict[str, Any]:
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


class DQNTrainer:
    """DQN训练器 - GPU加速版本"""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.environment = None
        self.preprocessedEnvironment = None
        self.agent = None

    def initializeEnvironment(self) -> None:
        """初始化环境"""
        try:
            # 使用NoFrameskip版本
            self.environment = gym.make(self.config.environmentName, render_mode='rgb_array')
            self.preprocessedEnvironment = AtariEnvironmentPreprocessor(
                self.environment,
                frameSkip=self.config.frameSkip,
                screenSize=self.config.screenSize
            )
            logger.info(f"环境初始化完成: {self.config.environmentName}")
        except Exception as e:
            logger.error(f"环境初始化失败: {e}")
            # 回退到普通版本
            try:
                envName = self.config.environmentName.replace("NoFrameskip", "")
                self.environment = gym.make(envName, render_mode='rgb_array')
                self.preprocessedEnvironment = AtariEnvironmentPreprocessor(
                    self.environment,
                    frameSkip=self.config.frameSkip,
                    screenSize=self.config.screenSize
                )
                logger.info(f"使用回退环境: {envName}")
            except Exception as e2:
                logger.error(f"回退环境也失败: {e2}")
                raise

    def recordInitialEpisodes(self) -> None:
        """在训练之前记录原始环境数据为PNG图片"""
        if not self.config.saveImages:
            logger.info("图像保存功能已禁用，跳过环境记录")
            return

        logger.info("开始记录原始环境数据为PNG图片...")
        recorder = EnvironmentRecorder(self.config)

        try:
            # 记录指定数量的episode
            for episode in range(self.config.numEpisodesToRecord):
                logger.info(f"记录第 {episode + 1}/{self.config.numEpisodesToRecord} 个episode...")
                recorder.recordEpisode(episode, maxSteps=500)

            logger.info(f"环境记录完成！所有帧已保存到: {recorder.saveDir}")
        except Exception as e:
            logger.error(f"环境记录过程中发生错误: {e}")
        finally:
            recorder.close()

    def train(self) -> Tuple[DeepQNAgent, List[float], List[float]]:
        """训练DQN智能体 - GPU加速版本"""
        # 首先记录原始环境数据
        self.recordInitialEpisodes()

        if self.preprocessedEnvironment is None:
            self.initializeEnvironment()

        numActions = self.preprocessedEnvironment.actionSpace.n
        stateShape = (4, self.config.screenSize, self.config.screenSize)

        self.agent = DeepQNAgent(stateShape, numActions, self.config)

        # 训练统计
        episodeRewards = []
        episodeLosses = []
        movingAverageRewards = []
        epsilonHistory = []

        bestAverageReward = -float('inf')

        logger.info(
            f"开始训练 {self.config.environmentName}, 使用GPU加速ResNet架构, 目标回合数: {self.config.trainingEpisodes}")

        for episode in range(self.config.trainingEpisodes):
            try:
                # 定期清理内存
                if episode % 10 == 0:
                    MemoryManager.clearMemory()

                state, _ = self.preprocessedEnvironment.reset()
                totalReward = 0.0
                stepsInEpisode = 0
                totalLoss = 0.0
                lossCount = 0

                while True:
                    # 确保状态张量格式正确并移动到GPU
                    if not isinstance(state, torch.Tensor):
                        state = torch.tensor(state, dtype=torch.float32)
                    state = state.to(device)  # 立即移动到GPU

                    action = self.agent.selectAction(state, training=True)
                    nextState, reward, done, _ = self.preprocessedEnvironment.step(action)

                    # 确保下一个状态张量格式正确并移动到GPU
                    if not isinstance(nextState, torch.Tensor):
                        nextState = torch.tensor(nextState, dtype=torch.float32)
                    nextState = nextState.to(device)  # 立即移动到GPU

                    self.agent.memory.push(state, action, reward, nextState, done)

                    loss = self.agent.optimizeModel()
                    if loss > 0:
                        totalLoss += loss
                        lossCount += 1

                    state = nextState
                    totalReward += reward
                    stepsInEpisode += 1

                    if done:
                        break

                self.agent.episodesCompleted += 1

                # 记录统计信息
                averageLoss = totalLoss / lossCount if lossCount > 0 else 0.0
                episodeRewards.append(totalReward)
                episodeLosses.append(averageLoss)
                epsilonHistory.append(self.agent.getCurrentEpsilon())

                # 计算移动平均奖励
                if len(episodeRewards) >= 50:
                    movingAverage = np.mean(episodeRewards[-50:])
                else:
                    movingAverage = np.mean(episodeRewards)
                movingAverageRewards.append(movingAverage)

                # 更新最佳模型
                if movingAverage > bestAverageReward and len(episodeRewards) >= 20:
                    bestAverageReward = movingAverage
                    self.agent.saveCheckpoint(
                        f"./DQN_V1_0_models/gpu_resnet_dqn_best_{self.config.environmentName.replace('/', '_')}.pth"
                    )

                # 定期日志输出
                if episode % 5 == 0:
                    stats = self.agent.getTrainingStatistics()
                    logger.info(
                        f"回合 {episode:4d} | "
                        f"奖励: {totalReward:7.2f} | "
                        f"步数: {stepsInEpisode:4d} | "
                        f"移动平均: {movingAverage:7.2f} | "
                        f"平均损失: {averageLoss:7.4f} | "
                        f"Epsilon: {stats['currentEpsilon']:.3f} | "
                        f"经验回放: {stats['memoryUsage']:5.1f}% | "
                        f"GPU内存: {stats['gpuMemoryAllocated']:.2f}GB"
                    )

                    # 定期保存检查点
                    if episode % 50 == 0 and episode > 0:
                        self.agent.saveCheckpoint(
                            f"./DQN_V1_0_models/gpu_resnet_dqn_checkpoint_{self.config.environmentName.replace('/', '_')}_episode_{episode}.pth"
                        )

                # 检查停止条件
                if (len(movingAverageRewards) >= 30 and
                        movingAverageRewards[-1] >= self.config.targetAverageReward):
                    logger.info(f"达到目标性能! 在回合 {episode}")
                    self.agent.saveCheckpoint(
                        f"./DQN_V1_0_models/gpu_resnet_dqn_final_{self.config.environmentName.replace('/', '_')}.pth"
                    )
                    break

            except Exception as e:
                logger.error(f"回合 {episode} 训练过程中发生错误: {e}")
                MemoryManager.clearMemory()
                continue

        self._plotTrainingResults(episodeRewards, movingAverageRewards, episodeLosses, epsilonHistory)
        return self.agent, episodeRewards, movingAverageRewards

    def _plotTrainingResults(self, episodeRewards: List[float], movingAverageRewards: List[float],
                             episodeLosses: List[float], epsilonHistory: List[float]) -> None:
        """绘制训练结果图表"""
        try:
            plt.figure(figsize=(15, 10))

            # 奖励曲线
            plt.subplot(2, 2, 1)
            plt.plot(episodeRewards, alpha=0.6, label='每回合奖励')
            plt.plot(movingAverageRewards, 'r-', linewidth=2, label='移动平均奖励 (50回合)')
            plt.title(f'{self.config.environmentName} - GPU加速ResNet DQN回合奖励')
            plt.xlabel('回合')
            plt.ylabel('奖励')
            plt.legend()
            plt.grid(True)

            # 损失曲线
            plt.subplot(2, 2, 2)
            plt.plot(episodeLosses)
            plt.title(f'{self.config.environmentName} - GPU加速ResNet DQN训练损失')
            plt.xlabel('回合')
            plt.ylabel('损失')
            plt.grid(True)

            # Epsilon衰减
            plt.subplot(2, 2, 3)
            plt.plot(epsilonHistory)
            plt.title(f'{self.config.environmentName} - GPU加速ResNet DQN Epsilon衰减')
            plt.xlabel('回合')
            plt.ylabel('Epsilon')
            plt.grid(True)

            # 奖励分布
            plt.subplot(2, 2, 4)
            plt.hist(episodeRewards, bins=50, alpha=0.7)
            plt.title(f'{self.config.environmentName} - GPU加速ResNet DQN奖励分布')
            plt.xlabel('奖励')
            plt.ylabel('频率')
            plt.grid(True)

            plt.tight_layout()
            plotFileName = f'gpu_resnet_training_results_{self.config.environmentName.replace("/", "_")}.png'
            plt.savefig(plotFileName, dpi=150, bbox_inches='tight')
            plt.close()
            logger.info(f"训练图表已保存到: {plotFileName}")
        except Exception as e:
            logger.error(f"绘制训练结果失败: {e}")

    def close(self) -> None:
        """关闭环境"""
        if self.preprocessedEnvironment:
            self.preprocessedEnvironment.close()
        MemoryManager.clearMemory()


def main():
    """主函数"""
    try:
        # 设置内存优化
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True

        # 训练配置
        config = TrainingConfig()

        logger.info("使用GPU加速的经验回放缓冲区 - 注意监控GPU显存使用!")

        # 创建训练器并开始训练
        trainer = DQNTrainer(config)
        trainedAgent, rewards, movingAverages = trainer.train()

        # 输出训练结果
        logger.info("GPU加速ResNet DQN训练完成!")
        if movingAverages:
            logger.info(f"最终移动平均奖励: {movingAverages[-1]:.2f}")
            logger.info(f"最大移动平均奖励: {max(movingAverages):.2f}")

        trainer.close()

    except KeyboardInterrupt:
        logger.info("训练被用户中断")
    except Exception as e:
        logger.error(f"训练过程中发生错误: {e}")
    finally:
        # 最终清理
        MemoryManager.clearMemory()
        logger.info("程序执行完毕")


if __name__ == "__main__":
    main()

