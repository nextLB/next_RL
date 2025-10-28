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
from typing import Tuple, List, Dict, Any
import gc
import psutil
import os

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dqn_resnet_fixed_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 经验回放缓冲区
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'nextState', 'done'])


class ExperienceReplayBuffer:
    """经验回放缓冲区，用于存储和采样训练经验"""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity

    def push(self, state: torch.Tensor, action: int, reward: float, nextState: torch.Tensor, done: bool) -> None:
        """添加经验到缓冲区"""
        try:
            # 确保状态在CPU上以节省GPU显存
            stateCpu = state.cpu() if state.is_cuda else state
            nextStateCpu = nextState.cpu() if nextState.is_cuda else nextState

            self.buffer.append(Experience(stateCpu, action, reward, nextStateCpu, done))
        except Exception as e:
            logger.error(f"添加经验到缓冲区失败: {e}")

    def sample(self, batchSize: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """从缓冲区中随机采样一批经验"""
        if len(self.buffer) < batchSize:
            raise ValueError(f"缓冲区中经验数量不足: {len(self.buffer)} < {batchSize}")

        try:
            experiences = random.sample(self.buffer, batchSize)

            # 将状态转移到设备
            states = torch.stack([e.state for e in experiences]).to(device)
            actions = torch.tensor([e.action for e in experiences], dtype=torch.long).to(device)
            rewards = torch.tensor([e.reward for e in experiences], dtype=torch.float32).to(device)
            nextStates = torch.stack([e.nextState for e in experiences]).to(device)
            dones = torch.tensor([e.done for e in experiences], dtype=torch.float32).to(device)

            return states, actions, rewards, nextStates, dones
        except Exception as e:
            logger.error(f"采样经验失败: {e}")
            raise

    def __len__(self) -> int:
        return len(self.buffer)

    def getUsagePercentage(self) -> float:
        """返回缓冲区的使用百分比"""
        return len(self.buffer) / self.capacity * 100


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
            for _ in range(4):
                self.frameBuffer.append(processedState)

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
                frame = frame.mean(axis=2)

            # 调整大小
            img = Image.fromarray(frame)
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


class ResidualBlock(nn.Module):
    """修复的残差块，确保尺寸匹配"""

    def __init__(self, inChannels: int, outChannels: int, stride: int = 1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(inChannels, outChannels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(outChannels)
        self.conv2 = nn.Conv2d(outChannels, outChannels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outChannels)

        # 修复的快捷连接 - 确保尺寸匹配
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


class FixedResNetDeepQNetwork(nn.Module):
    """修复的ResNet DQN网络，确保所有尺寸匹配"""

    def __init__(self, inputShape: Tuple[int, int, int], numActions: int):
        super(FixedResNetDeepQNetwork, self).__init__()

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

        for stride in strides:
            layers.append(ResidualBlock(inChannels, outChannels, stride))
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


class SimpleResNetDQN(nn.Module):
    """简化的ResNet DQN，更稳定的架构"""

    def __init__(self, inputShape: Tuple[int, int, int], numActions: int):
        super(SimpleResNetDQN, self).__init__()

        # 初始卷积
        self.conv1 = nn.Conv2d(inputShape[0], 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # 残差块1
        self.resBlock1 = self._createResBlock(32, 32)
        # 残差块2 - 降采样
        self.resBlock2 = self._createResBlock(32, 64, stride=2)
        # 残差块3
        self.resBlock3 = self._createResBlock(64, 64)
        # 残差块4 - 降采样
        self.resBlock4 = self._createResBlock(64, 128, stride=2)

        # 全局平均池化
        self.globalPool = nn.AdaptiveAvgPool2d((1, 1))

        # 全连接层
        self.fc = nn.Linear(128, numActions)

    def _createResBlock(self, inChannels: int, outChannels: int, stride: int = 1) -> nn.Module:
        """创建残差块"""
        return nn.Sequential(
            nn.Conv2d(inChannels, outChannels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outChannels),
            nn.ReLU(inplace=True),
            nn.Conv2d(outChannels, outChannels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outChannels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 初始卷积
        x = F.relu(self.bn1(self.conv1(x)))

        # 残差块
        x = self.resBlock1(x) + x  # 残差连接
        x = self.resBlock2(x)
        x = self.resBlock3(x) + x  # 残差连接
        x = self.resBlock4(x)

        # 全局池化和输出
        x = self.globalPool(x)
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
    """修复的DQN智能体"""

    def __init__(self, stateShape: Tuple[int, int, int], numActions: int, learningRate: float = 0.00025,
                 useSimpleResNet: bool = True):
        self.numActions = numActions
        self.stateShape = stateShape

        # 初始化网络 - 使用修复的架构
        if useSimpleResNet:
            self.policyNetwork = SimpleResNetDQN(stateShape, numActions).to(device)
            self.targetNetwork = SimpleResNetDQN(stateShape, numActions).to(device)
            logger.info("使用简化的ResNet架构")
        else:
            self.policyNetwork = FixedResNetDeepQNetwork(stateShape, numActions).to(device)
            self.targetNetwork = FixedResNetDeepQNetwork(stateShape, numActions).to(device)
            logger.info("使用完整的ResNet架构")

        self._updateTargetNetwork()
        self.targetNetwork.eval()

        # 优化器
        self.optimizer = optim.Adam(
            self.policyNetwork.parameters(),
            lr=learningRate,
            eps=1e-4
        )

        # 经验回放缓冲区
        self.memory = ExperienceReplayBuffer(capacity=50000)  # 减小容量

        # 训练状态
        self.stepsCompleted = 0
        self.episodesCompleted = 0

        # 超参数
        self.batchSize = 32
        self.discountFactor = 0.99
        self.initialEpsilon = 1.0
        self.finalEpsilon = 0.1
        self.epsilonDecaySteps = 50000
        self.targetUpdateFrequency = 1000
        self.learningStartSteps = 1000  # 减少初始学习步数
        self.learningUpdateFrequency = 4

        # 内存管理
        self.memoryManager = MemoryManager()

        logger.info(f"修复的DQN智能体初始化完成: 状态形状={stateShape}, 动作数量={numActions}")

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
                    # 确保输入尺寸正确
                    if len(state.shape) == 3:
                        state = state.unsqueeze(0)  # 添加batch维度

                    stateDevice = state.to(device) if not state.is_cuda else state
                    qValues = self.policyNetwork(stateDevice)
                    return qValues.max(1)[1].item()
            else:
                return random.randrange(self.numActions)
        except Exception as e:
            logger.error(f"选择动作失败: {e}")
            # 记录状态信息用于调试
            logger.error(f"状态形状: {state.shape if hasattr(state, 'shape') else 'No shape'}")
            return random.randrange(self.numActions)

    def optimizeModel(self) -> float:
        """执行一次模型优化"""
        if (len(self.memory) < self.batchSize or
                self.stepsCompleted < self.learningStartSteps or
                self.stepsCompleted % self.learningUpdateFrequency != 0):
            return 0.0

        try:
            states, actions, rewards, nextStates, dones = self.memory.sample(self.batchSize)

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
                targetQValues = rewards + (self.discountFactor * nextQValues * (1 - dones))

            # 计算损失
            loss = F.smooth_l1_loss(currentQValues.squeeze(), targetQValues)

            # 优化模型
            self.optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.policyNetwork.parameters(), 10.0)
            self.optimizer.step()

            # 定期更新目标网络
            if self.stepsCompleted % self.targetUpdateFrequency == 0:
                self._updateTargetNetwork()
                logger.debug(f"更新目标网络，步骤: {self.stepsCompleted}")

            return loss.item()

        except Exception as e:
            logger.error(f"模型优化失败: {e}")
            return 0.0

    def _calculateCurrentEpsilon(self) -> float:
        """计算当前的epsilon值"""
        return self.finalEpsilon + (self.initialEpsilon - self.finalEpsilon) * \
            np.exp(-1.0 * self.stepsCompleted / self.epsilonDecaySteps)

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
        gpuAllocated, gpuCached = self.memoryManager.getGpuMemoryUsage()
        systemMemory = self.memoryManager.getSystemMemoryUsage()

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
        self.memoryManager.clearMemory()


class DQNTrainer:
    """修复的DQN训练器"""

    def __init__(self, environmentName: str = "PongNoFrameskip-v4", useSimpleResNet: bool = True):
        self.environmentName = environmentName
        self.useSimpleResNet = useSimpleResNet
        self.environment = None
        self.preprocessedEnvironment = None
        self.agent = None
        self.memoryManager = MemoryManager()

    def initializeEnvironment(self) -> None:
        """初始化环境"""
        try:
            # 使用NoFrameskip版本
            self.environment = gym.make(self.environmentName, render_mode='rgb_array')
            self.preprocessedEnvironment = AtariEnvironmentPreprocessor(self.environment)
            logger.info(f"环境初始化完成: {self.environmentName}")
        except Exception as e:
            logger.error(f"环境初始化失败: {e}")
            # 回退到普通版本
            try:
                envName = self.environmentName.replace("NoFrameskip", "")
                self.environment = gym.make(envName, render_mode='rgb_array')
                self.preprocessedEnvironment = AtariEnvironmentPreprocessor(self.environment)
                logger.info(f"使用回退环境: {envName}")
            except Exception as e2:
                logger.error(f"回退环境也失败: {e2}")
                raise

    def train(self, numEpisodes: int = 1000, targetAverageReward: float = 15.0) -> Tuple[
        DeepQNAgent, List[float], List[float]]:
        """训练DQN智能体"""
        if self.preprocessedEnvironment is None:
            self.initializeEnvironment()

        numActions = self.preprocessedEnvironment.actionSpace.n
        stateShape = (4, 84, 84)

        self.agent = DeepQNAgent(stateShape, numActions, useSimpleResNet=self.useSimpleResNet)

        # 训练统计
        episodeRewards = []
        episodeLosses = []
        movingAverageRewards = []
        epsilonHistory = []

        bestAverageReward = -float('inf')

        logger.info(f"开始训练 {self.environmentName}, 使用修复的ResNet架构, 目标回合数: {numEpisodes}")

        for episode in range(numEpisodes):
            try:
                # 定期清理内存
                if episode % 10 == 0:
                    self.memoryManager.clearMemory()

                state, _ = self.preprocessedEnvironment.reset()
                totalReward = 0.0
                stepsInEpisode = 0
                totalLoss = 0.0
                lossCount = 0

                while True:
                    # 确保状态张量格式正确
                    if not isinstance(state, torch.Tensor):
                        state = torch.tensor(state, dtype=torch.float32)

                    action = self.agent.selectAction(state, training=True)
                    nextState, reward, done, _ = self.preprocessedEnvironment.step(action)

                    # 确保下一个状态张量格式正确
                    if not isinstance(nextState, torch.Tensor):
                        nextState = torch.tensor(nextState, dtype=torch.float32)

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
                        f"fixed_resnet_dqn_best_{self.environmentName.replace('/', '_')}.pth"
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
                        f"经验回放: {stats['memoryUsage']:5.1f}%"
                    )

                    # 定期保存检查点
                    if episode % 50 == 0 and episode > 0:
                        self.agent.saveCheckpoint(
                            f"fixed_resnet_dqn_checkpoint_{self.environmentName.replace('/', '_')}_episode_{episode}.pth"
                        )

                # 检查停止条件
                if (len(movingAverageRewards) >= 30 and
                        movingAverageRewards[-1] >= targetAverageReward):
                    logger.info(f"达到目标性能! 在回合 {episode}")
                    self.agent.saveCheckpoint(
                        f"fixed_resnet_dqn_final_{self.environmentName.replace('/', '_')}.pth"
                    )
                    break

            except Exception as e:
                logger.error(f"回合 {episode} 训练过程中发生错误: {e}")
                self.memoryManager.clearMemory()
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
            plt.title(f'{self.environmentName} - 修复的ResNet DQN回合奖励')
            plt.xlabel('回合')
            plt.ylabel('奖励')
            plt.legend()
            plt.grid(True)

            # 损失曲线
            plt.subplot(2, 2, 2)
            plt.plot(episodeLosses)
            plt.title(f'{self.environmentName} - 修复的ResNet DQN训练损失')
            plt.xlabel('回合')
            plt.ylabel('损失')
            plt.grid(True)

            # Epsilon衰减
            plt.subplot(2, 2, 3)
            plt.plot(epsilonHistory)
            plt.title(f'{self.environmentName} - 修复的ResNet DQN Epsilon衰减')
            plt.xlabel('回合')
            plt.ylabel('Epsilon')
            plt.grid(True)

            # 奖励分布
            plt.subplot(2, 2, 4)
            plt.hist(episodeRewards, bins=50, alpha=0.7)
            plt.title(f'{self.environmentName} - 修复的ResNet DQN奖励分布')
            plt.xlabel('奖励')
            plt.ylabel('频率')
            plt.grid(True)

            plt.tight_layout()
            plotFileName = f'fixed_resnet_training_results_{self.environmentName.replace("/", "_")}.png'
            plt.savefig(plotFileName, dpi=150, bbox_inches='tight')
            plt.close()
            logger.info(f"训练图表已保存到: {plotFileName}")
        except Exception as e:
            logger.error(f"绘制训练结果失败: {e}")

    def close(self) -> None:
        """关闭环境"""
        if self.preprocessedEnvironment:
            self.preprocessedEnvironment.close()
        self.memoryManager.clearMemory()


def main():
    """主函数"""
    try:
        # 设置内存优化
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True

        # 训练参数 - 使用更稳定的设置
        environmentName = "PongNoFrameskip-v4"
        trainingEpisodes = 1000
        useSimpleResNet = True  # 使用简化的ResNet架构

        # 创建训练器并开始训练
        trainer = DQNTrainer(environmentName, useSimpleResNet=useSimpleResNet)
        trainedAgent, rewards, movingAverages = trainer.train(trainingEpisodes)

        # 输出训练结果
        logger.info("修复的ResNet DQN训练完成!")
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
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("程序执行完毕")


if __name__ == "__main__":
    main()