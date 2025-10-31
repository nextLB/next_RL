"""
PPO训练配置参数
"""
from dataclasses import dataclass
import torch


@dataclass
class PPOConfig:
    """PPO训练配置参数"""
    environmentName: str = "SpaceInvadersNoFrameskip-v4"
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


# 全局设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



