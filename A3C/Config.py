"""
A3C训练配置参数
"""

from dataclasses import dataclass

@dataclass
class A3CConfig:
    environmentName: str = "PongNoFrameskip-v4"
    learningRate: float = 0.0007
    discountFactor: float = 0.99
    numProcesses: int = 4
    tMax: int = 20
    entropyCoefficient: float = 0.01
    valueLossCoefficient: float = 0.5
    maxGradientNorm: float = 50.0
    frameSkip: int = 4
    screenSize: int = 84
    trainingSteps: int = 9999999
    saveModelFrequency: int = 10000
    saveCheckpointFrequency: int = 500000  # 新增：检查点保存频率
    logInterval: int = 3000
    bestModelThreshold: float = -18.0  # 新增：最佳模型阈值