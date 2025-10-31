"""
配置参数类
"""

from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """训练配置参数"""
    environmentName: str = "PongNoFrameskip-v4"
    learningRate: float = 0.00025
    discountFactor: float = 0.99
    batchSize: int = 16
    replayBufferCapacity: int = 10000
    targetUpdateFrequency: int = 3000
    learningStartSteps: int = 10000
    learningUpdateFrequency: int = 4
    initialEpsilon: float = 1.0
    finalEpsilon: float = 0.01
    epsilonDecaySteps: int = 100000
    frameSkip: int = 4
    screenSize: int = 84
    useSimpleResNet: bool = True
    trainingEpisodes: int = 1000
    targetAverageReward: float = 15.0
    saveImages: bool = True
    imageSaveDir: str = "recordedPongEpisodes"
    numEpisodesToRecord: int = 3