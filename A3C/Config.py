"""
A3C训练配置参数
"""

from dataclasses import dataclass

@dataclass
class A3CConfig:
    environmentName: str = "PongNoFrameskip-v4"
    learningRate: float = 0.00025
    discountFactor: float = 0.99
    numProcesses: int = 2
    tMax: int = 30
    entropyCoefficient: float = 0.001
    valueLossCoefficient: float = 0.5
    maxGradientNorm: float = 10.0
    frameSkip: int = 4
    screenSize: int = 84
    trainingSteps: int = 50000000
    saveModelFrequency: int = 100000
    saveCheckpointFrequency: int = 50000
    logInterval: int = 1000
    bestModelThreshold: float = 0.0