"""
    依照本目录的paper目录下的DQNNaturePaper篇文章进行训练的，与原论文在架构方面有一些改动的
"""

# V1.0  2025.10.28      --- by next, 初步实现了使用Renet深度学习模型架构的DQN强化学习模型的搭建与训练等


import torch
import torch.nn as nn
import logging
from collections import deque, namedtuple
from dataclasses import dataclass
import torch.nn.functional as F
from typing import Tuple, List, Dict, Any
import gc
import psutil
import os
import time
from contextlib import contextmanager



# 配置类
@dataclass
class TrainingConfig:
    """训练配置参数"""
    environmentName: str = "PongNoFrameskip-v4"
    learningRate: float = 0.00025
    discountFactor: float = 0.99
    batchSize: int = 32
    replayBufferCapacity: int = 30000
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

############################################################################
# 关于环境 environmentName 常用的还有如下这些:
# A. OpenAI Gym 经典控制环境
    "CartPole-v1"           # 平衡杆
    "Acrobot-v1"            # 杂技机器人
    "MountainCar-v0"        # 爬山车
    "Pendulum-v1"           # 倒立摆
# B. OpenAI Gym Atari 游戏环境
    # 基础版本
    "Pong-v4"               # 乒乓球
    "Breakout-v4"           # 打砖块
    "SpaceInvaders-v4"      # 太空侵略者
    "MsPacman-v4"           # 吃豆人女士

    # NoFrameskip版本（更常用）
    "PongNoFrameskip-v4"
    "BreakoutNoFrameskip-v4"
    "SpaceInvadersNoFrameskip-v4"

# C. MuJoCo 连续控制环境
    "HalfCheetah-v3"        # 猎豹
    "Hopper-v3"             # 单脚跳
    "Walker2d-v3"           # 双足行走
    "Ant-v3"                # 蚂蚁
    "Humanoid-v3"           # 人形机器人
# D. Box2D 环境
    "LunarLander-v2"        # 月球着陆器
    "BipedalWalker-v3"      # 双足行走者
    "CarRacing-v0"          # 赛车
# E. 其他流行环境
    # DeepMind Control Suite
    "cheetah-run"
    "walker-walk"
    "humanoid-walk"

    # Unity ML-Agents
    "3DBall"                # 3D平衡球
    "Pyramids"              # 金字塔

    # 自定义环境
    "MyCustomEnv-v0"
############################################################################



# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./log/resnet_dqn_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

############################################################################
# 经验回放缓冲区
# 经验回放缓冲区就是智能体的“记忆库”，它做的就是同样的事情：存储过去的经历，并从中反复学习。
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'nextState', 'done'])
# 这代表一条经验，它包含了从环境中获得的一个完整的“快照”：
#     state: 智能体在执行动作之前所处的状态。
#     action: 智能体在那个状态下所执行的动作。
#     reward: 执行动作后，环境反馈给智能体的奖励（或惩罚）。
#     nextState: 执行动作后，智能体进入的新状态。
#     done: 一个布尔值（True/False），表示这个nextState是否是终止状态（例如，游戏结束、任务完成）。
############################################################################


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



class ResidualBlock(nn.Module):
    """resnet架构的残差块"""

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



class ExperienceReplayBuffer:
    """GPU加速的经验回放缓冲区"""




def main():
    print(f"使用设备: {device}")



if __name__ == '__main__':
    main()


