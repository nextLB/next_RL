"""
A3C训练配置参数
"""

from dataclasses import dataclass

@dataclass
class A3CConfig:
    environmentName: str = "PongNoFrameskip-v4"
    learningRate: float = 0.0007  # 恢复到原始学习率
    discountFactor: float = 0.99
    numProcesses: int = 8
    tMax: int = 20
    entropyCoefficient: float = 0.01
    valueLossCoefficient: float = 0.5
    maxGradientNorm: float = 50.0
    frameSkip: int = 4
    screenSize: int = 84
    trainingSteps: int = 1000000  # 恢复到合理步数
    saveModelFrequency: int = 10000
    logInterval: int = 1000