"""
A3C训练配置参数
"""

from dataclasses import dataclass

@dataclass
class A3CConfig:
    environmentName: str = "PongNoFrameskip-v4"
    learningRate: float = 0.0001  # 降低学习率
    discountFactor: float = 0.99
    numProcesses: int = 8
    tMax: int = 20
    entropyCoefficient: float = 0.01
    valueLossCoefficient: float = 0.5
    maxGradientNorm: float = 40.0
    frameSkip: int = 4
    screenSize: int = 84
    trainingSteps: int = 10000000  # 增加训练步数
    saveModelFrequency: int = 100000
    logInterval: int = 1000
