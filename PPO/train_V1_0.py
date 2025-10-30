"""
PPO (Proximal Policy Optimization) 算法实现 - GPU加速版本
基于原DQN代码框架，按照PPO论文完整复现
V1.1 2025.10.30 - PPO实现 + GPU经验缓冲区加速

整理优化：
1. 增强代码可读性和结构
2. 改进错误处理和鲁棒性
3. 优化内存管理和GPU使用
4. 添加详细的类型注解和文档
5. 统一使用驼峰命名法
"""

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
from typing import Tuple, List, Dict, Any, Optional, Generator
import gc
import psutil
import os
import time
from dataclasses import dataclass
from contextlib import contextmanager
import json
from datetime import datetime


# =============================== 配置类 ===============================
@dataclass
class PPOConfig:
    """PPO训练配置参数"""
    """
        "BreakoutNoFrameskip-v4",
        "SpaceInvadersNoFrameskip-v4", 
        "SeaquestNoFrameskip-v4"
    """
    # environmentName: str = "PongNoFrameskip-v4"
    # environmentName: str = "BreakoutNoFrameskip-v4"
    environmentName: str = "SpaceInvadersNoFrameskip-v4"
    # environmentName: str = "SeaquestNoFrameskip-v4"
    learningRate: float = 0.00025
    clipEpsilon: float = 0.1
    discountFactor: float = 0.99
    gaeLambda: float = 0.95
    valueLossCoeff: float = 0.5
    entropyCoeff: float = 0.01
    ppoEpochs: int = 3
    batchSize: int = 32
    horizon: int = 128  # 每个actor的时间步数
    numActors: int = 8  # 并行actor数量
    trainingTimesteps: int = 10000000
    targetAverageReward: float = 15.0
    frameSkip: int = 4
    screenSize: int = 84
    saveImages: bool = True
    imageSaveDir: str = "recordedPongPPO"
    numEpisodesToRecord: int = 3
    useAdam: bool = True
    adamEpsilon: float = 1e-5
    maxGradNorm: float = 0.5
    bufferOnGpu: bool = True  # 经验缓冲区是否放在GPU上
    logInterval: int = 5  # 日志输出间隔
    saveInterval: int = 50  # 模型保存间隔


# =============================== 全局设置 ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./log/ppoResNetGpuMemory.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# =============================== 工具类 ===============================
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


# =============================== 网络结构 ===============================
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


# =============================== 经验缓冲区 ===============================
PPOExperience = namedtuple('PPOExperience',
                          ['state', 'action', 'reward', 'value', 'logProb', 'done'])


class PPOBuffer:
    """PPO专用的经验缓冲区 - GPU加速版本"""

    def __init__(self, horizon: int, numActors: int,
                 stateShape: Tuple[int, int, int], device: torch.device = device):
        self.horizon = horizon
        self.numActors = numActors
        self.stateShape = stateShape
        self.device = device

        # 初始化缓冲区 - 直接在GPU上创建
        self.states = torch.zeros((horizon, numActors) + stateShape, device=device)
        self.actions = torch.zeros((horizon, numActors), dtype=torch.long, device=device)
        self.rewards = torch.zeros((horizon, numActors), device=device)
        self.values = torch.zeros((horizon, numActors), device=device)
        self.logProbs = torch.zeros((horizon, numActors), device=device)
        self.dones = torch.zeros((horizon, numActors), dtype=torch.bool, device=device)

        self.advantages = torch.zeros((horizon, numActors), device=device)
        self.returns = torch.zeros((horizon, numActors), device=device)

        self.step = 0

    def push(self, state: torch.Tensor, action: torch.Tensor, reward: torch.Tensor,
             value: torch.Tensor, logProb: torch.Tensor, done: torch.Tensor):
        """添加经验到缓冲区 - 所有张量已经在GPU上"""
        if self.step < self.horizon:
            self.states[self.step] = state
            self.actions[self.step] = action
            self.rewards[self.step] = reward
            self.values[self.step] = value
            self.logProbs[self.step] = logProb
            self.dones[self.step] = done

            self.step += 1

    def computeAdvantagesAndReturns(self, lastValues: torch.Tensor,
                                     gamma: float = 0.99, gaeLambda: float = 0.95):
        """计算优势函数和回报 - 在GPU上执行"""
        advantages = torch.zeros_like(self.rewards)
        lastAdvantage = 0

        for t in reversed(range(self.horizon)):
            if t == self.horizon - 1:
                nextValue = lastValues
                nextNonTerminal = 1.0 - self.dones[t].float()
            else:
                nextValue = self.values[t + 1]
                nextNonTerminal = 1.0 - self.dones[t].float()

            delta = (self.rewards[t] + gamma * nextValue * nextNonTerminal
                    - self.values[t])
            advantages[t] = lastAdvantage = (delta + gamma * gaeLambda
                                            * nextNonTerminal * lastAdvantage)

        self.returns = advantages + self.values

        # 标准化优势函数
        if advantages.std() > 0:  # 避免除零
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        self.advantages = advantages

    def getBatches(self, batchSize: int) -> Generator:
        """生成训练批次 - 数据已经在GPU上，无需转移"""
        # 展平所有数据
        states = self.states.view(-1, *self.stateShape)
        actions = self.actions.view(-1)
        oldLogProbs = self.logProbs.view(-1)
        advantages = self.advantages.view(-1)
        returns = self.returns.view(-1)

        # 随机打乱
        indices = torch.randperm(states.size(0), device=self.device)

        for startIdx in range(0, states.size(0), batchSize):
            endIdx = min(startIdx + batchSize, states.size(0))
            batchIndices = indices[startIdx:endIdx]

            yield (
                states[batchIndices],
                actions[batchIndices],
                oldLogProbs[batchIndices],
                advantages[batchIndices],
                returns[batchIndices]
            )

    def clear(self):
        """清空缓冲区"""
        self.step = 0
        self.advantages.zero_()
        self.returns.zero_()


# =============================== 环境处理 ===============================
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

    def __init__(self, config: PPOConfig):
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

    def saveFrameWithInfo(self, frame, episode, step, action, reward,
                           terminated, truncated, info):
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


# =============================== PPO智能体 ===============================
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

    def update(self, buffer: PPOBuffer) -> Dict[str, float]:
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
        MemoryManager.clearMemory()


# =============================== PPO训练器 ===============================
class PPOTrainer:
    """PPO训练器"""

    def __init__(self, config: PPOConfig):
        self.config = config
        self.environment = None
        self.preprocessedEnvironment = None
        self.agent = None
        self.buffer = None

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

    def collectExperience(self) -> Tuple[float, int, torch.Tensor]:
        """收集经验数据 - 确保所有数据都在GPU上"""
        totalReward = 0
        episodeCount = 0

        # 初始化状态 - 确保在GPU上
        states = []
        for _ in range(self.config.numActors):
            state, _ = self.preprocessedEnvironment.reset()
            states.append(state)
        states = torch.stack(states).to(device)

        # 收集经验
        for step in range(self.config.horizon):
            with torch.no_grad():
                actions, logProbs, values, _ = self.agent.network.getAction(states)

            # 执行动作
            nextStates = []
            rewards = []
            dones = []
            infos = []

            for i in range(self.config.numActors):
                nextState, reward, done, info = self.preprocessedEnvironment.step(actions[i].item())
                nextStates.append(nextState)
                rewards.append(reward)
                dones.append(done)
                infos.append(info)

                totalReward += reward
                if done:
                    episodeCount += 1

            # 转换为张量并确保在GPU上
            nextStates = torch.stack(nextStates).to(device)
            rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
            dones = torch.tensor(dones, dtype=torch.bool, device=device)

            # 存储经验 - 所有数据已经在GPU上
            self.buffer.push(states, actions, rewards, values, logProbs, dones)

            # 更新状态
            states = nextStates

            # 如果环境结束，重置
            for i in range(self.config.numActors):
                if dones[i]:
                    state, _ = self.preprocessedEnvironment.reset()
                    states[i] = state.to(device)

        # 计算最后一个状态的价值
        with torch.no_grad():
            _, lastValues, _ = self.agent.network(states)

        return totalReward, episodeCount, lastValues

    def train(self) -> Tuple[PPOAgent, List[float], List[float]]:
        """训练PPO智能体"""
        # 首先记录原始环境数据
        self.recordInitialEpisodes()

        if self.preprocessedEnvironment is None:
            self.initializeEnvironment()

        numActions = self.preprocessedEnvironment.actionSpace.n
        stateShape = (4, self.config.screenSize, self.config.screenSize)

        self.agent = PPOAgent(stateShape, numActions, self.config)
        self.buffer = PPOBuffer(self.config.horizon, self.config.numActors, stateShape, device)

        # 训练统计
        episodeRewards = []
        movingAverageRewards = []
        trainingStatsHistory = []

        bestAverageReward = -float('inf')
        totalTimesteps = 0
        updateCount = 0

        logger.info(
            f"开始训练 {self.config.environmentName}, 使用PPO算法, 目标时间步数: {self.config.trainingTimesteps}")

        while totalTimesteps < self.config.trainingTimesteps:
            try:
                # 收集经验
                totalReward, episodeCount, lastValues = self.collectExperience()
                totalTimesteps += self.config.horizon * self.config.numActors
                self.agent.stepsCompleted = totalTimesteps
                self.agent.episodesCompleted += episodeCount

                # 计算优势函数和回报 - 在GPU上执行
                self.buffer.computeAdvantagesAndReturns(
                    lastValues,
                    self.config.discountFactor,
                    self.config.gaeLambda
                )

                # 更新网络
                stats = self.agent.update(self.buffer)
                trainingStatsHistory.append(stats)
                updateCount += 1

                # 记录奖励
                if episodeCount > 0:
                    episodeRewards.append(totalReward / episodeCount)
                else:
                    episodeRewards.append(totalReward)

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
                        f"./PPO_V1_0_models/ppoResNetBest_{self.config.environmentName.replace('/', '_')}.pth"
                    )

                # 定期日志输出
                if updateCount % self.config.logInterval == 0:
                    trainingStats = self.agent.getTrainingStatistics()
                    logger.info(
                        f"时间步 {totalTimesteps:8d} | "
                        f"平均奖励: {episodeRewards[-1]:7.2f} | "
                        f"移动平均: {movingAverage:7.2f} | "
                        f"策略损失: {stats['policyLoss']:7.4f} | "
                        f"价值损失: {stats['valueLoss']:7.4f} | "
                        f"熵: {stats['entropy']:7.4f} | "
                        f"裁剪比例: {stats['clipFraction']:7.4f}"
                    )

                # 定期保存检查点
                if updateCount % self.config.saveInterval == 0 and updateCount > 0:
                    self.agent.saveCheckpoint(
                        f"./PPO_V1_0_models/ppoResNetCheckpoint_{self.config.environmentName.replace('/', '_')}_step_{totalTimesteps}.pth"
                    )

                # 检查停止条件
                if (len(movingAverageRewards) >= 30 and
                        movingAverageRewards[-1] >= self.config.targetAverageReward):
                    logger.info(f"达到目标性能! 在时间步 {totalTimesteps}")
                    self.agent.saveCheckpoint(
                        f"./PPO_V1_0_models/ppoResNetFinal_{self.config.environmentName.replace('/', '_')}.pth"
                    )
                    break

                # 清空缓冲区
                self.buffer.clear()

            except Exception as e:
                logger.error(f"训练过程中发生错误: {e}")
                MemoryManager.clearMemory()
                continue

        self._plotTrainingResults(episodeRewards, movingAverageRewards, trainingStatsHistory)
        return self.agent, episodeRewards, movingAverageRewards

    def _plotTrainingResults(self, episodeRewards: List[float],
                             movingAverageRewards: List[float],
                             trainingStats: List[Dict[str, float]]) -> None:
        """绘制训练结果图表"""
        try:
            plt.figure(figsize=(15, 12))

            # 奖励曲线
            plt.subplot(3, 2, 1)
            plt.plot(episodeRewards, alpha=0.6, label='每回合奖励')
            plt.plot(movingAverageRewards, 'r-', linewidth=2, label='移动平均奖励 (50回合)')
            plt.title(f'{self.config.environmentName} - PPO回合奖励')
            plt.xlabel('更新次数')
            plt.ylabel('奖励')
            plt.legend()
            plt.grid(True)

            # 策略损失
            plt.subplot(3, 2, 2)
            policyLosses = [stats['policyLoss'] for stats in trainingStats]
            plt.plot(policyLosses)
            plt.title(f'{self.config.environmentName} - PPO策略损失')
            plt.xlabel('更新次数')
            plt.ylabel('策略损失')
            plt.grid(True)

            # 价值损失
            plt.subplot(3, 2, 3)
            valueLosses = [stats['valueLoss'] for stats in trainingStats]
            plt.plot(valueLosses)
            plt.title(f'{self.config.environmentName} - PPO价值损失')
            plt.xlabel('更新次数')
            plt.ylabel('价值损失')
            plt.grid(True)

            # 熵
            plt.subplot(3, 2, 4)
            entropies = [stats['entropy'] for stats in trainingStats]
            plt.plot(entropies)
            plt.title(f'{self.config.environmentName} - PPO熵')
            plt.xlabel('更新次数')
            plt.ylabel('熵')
            plt.grid(True)

            # 裁剪比例
            plt.subplot(3, 2, 5)
            clipFractions = [stats['clipFraction'] for stats in trainingStats]
            plt.plot(clipFractions)
            plt.title(f'{self.config.environmentName} - PPO裁剪比例')
            plt.xlabel('更新次数')
            plt.ylabel('裁剪比例')
            plt.grid(True)

            # KL散度
            plt.subplot(3, 2, 6)
            approxKls = [stats['approxKl'] for stats in trainingStats]
            plt.plot(approxKls)
            plt.title(f'{self.config.environmentName} - PPO近似KL散度')
            plt.xlabel('更新次数')
            plt.ylabel('KL散度')
            plt.grid(True)

            plt.tight_layout()
            plotFileName = f'ppoTrainingResults_{self.config.environmentName.replace("/", "_")}.png'
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


# =============================== 主函数 ===============================
def main():
    """主函数"""
    try:
        # 设置内存优化
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True

        # 创建必要的目录
        os.makedirs('./log', exist_ok=True)
        os.makedirs('./PPO_V1_0_models', exist_ok=True)

        # PPO训练配置 - 按照论文中的超参数
        config = PPOConfig()

        logger.info("开始PPO训练!")

        # 创建训练器并开始训练
        trainer = PPOTrainer(config)
        trainedAgent, rewards, movingAverages = trainer.train()

        # 输出训练结果
        logger.info("PPO训练完成!")
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

