"""
PPO训练配置参数
"""
from dataclasses import dataclass
import torch


@dataclass
class PPOConfig:
    """PPO训练配置参数"""
    environmentName: str = "PongNoFrameskip-v4"
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






# 月球着陆器的配置
@dataclass
class LunarLanderConfig:
    environmentName: str = "LunarLander-v2"
    saveVideosPath: str = "./record/LunarLander"
    saveVideosEpisode: int = 5
    saveVideosSteps: int = 300
    screenSize: int = 240
    # 策略网络的学习率
    actorLearningRate: float = 0.00001
    # 价值网络的学习率
    criticLearningRate: float = 0.00001
    # 优化器的衰减系数
    adamEpsilon: float = 1e-5
    clipEpsilon: float = 0.1
    entropyCoeff: float = 0.01
    valueRegCoeff: float = 0.01
    # 经验缓冲区的大小设定
    bufferSize: int = 10000
    # 训练的总轮数设定
    trainingEpisode: int = 1000
    # 经验缓冲区中最小的经验数量，依照这个判断是否进行学习
    miniUpdateSize: int = 500
    gamma: float = 0.99
    gaeLambda: float = 0.95
    # 策略内更新每次所需要的轮数
    onPolicyEpochs: int = 20
    # 策略外更新每次所需要的轮数
    offPolicyEpochs: int = 20
    # 策略外的更新频率
    offPolicyUpdateFreq: int = 4
    # 在策略外的目标网络的更新次数
    targetUpdateInterval: int = 300
    tau: float = 0.05
    maxGradNorm: float = 0.5
    valueLossCoeff: float = 0.5





