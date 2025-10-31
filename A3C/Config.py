"""
Configuration classes for A3C and DQN training
"""

from dataclasses import dataclass
from typing import Tuple
import torch


@dataclass
class A3CConfig:
    """A3C training configuration parameters"""
    environmentName: str = "PongNoFrameskip-v4"
    learningRate: float = 0.0001
    discountFactor: float = 0.99
    entropyCoeff: float = 0.01
    valueLossCoeff: float = 0.5
    maxGradNorm: float = 40.0
    nStep: int = 40
    numProcesses: int = 32
    trainingTimesteps: int = 1000000
    frameSkip: int = 4
    screenSize: int = 84
    useLSTM: bool = False
    logInterval: int = 10
    saveInterval: int = 10000
    maxEpisodeLength: int = 10000
    modelSavePath: str = "./A3CModels"
    bestModelPath: str = "./A3CModels/best_model.pth"
    checkpointInterval: int = 5000
    rewardClip: bool = True
    gradientClip: bool = True


@dataclass
class TrainingConfig:
    """DQN training configuration parameters"""
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


# Global device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
