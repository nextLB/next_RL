"""
A3C训练配置参数
"""

from dataclasses import dataclass

@dataclass
class A3CConfig:
    environmentName: str = "PongNoFrameskip-v4"
    learningRate: float = 0.0001  # 降低学习率
    discountFactor: float = 0.99
    numProcesses: int = 4
    tMax: int = 20
    entropyCoefficient: float = 0.001  # 调整熵系数
    valueLossCoefficient: float = 0.5
    maxGradientNorm: float = 10.0  # 调整梯度裁剪
    frameSkip: int = 4
    screenSize: int = 84
    trainingSteps: int = 10000000
    saveModelFrequency: int = 500  # 减少保存频率
    saveCheckpointFrequency: int = 500000
    logInterval: int = 1000  # 更频繁的日志
    bestModelThreshold: float = 0.0  # 调整为正值