"""
    关于强化学习算法DQN的配置文件
"""

from dataclasses import dataclass

# 配置类
@dataclass
class Config:
    # 环境配置
    environmentName: str = "PongNoFrameskip-v4"
    # 使用的是强化学习环境的名称，PongNoFrameskip-v4是Atari乒乓球游戏的无跳帧版本
    # 还可以使用"BreakoutNoFrameskip-v4" "SpaceInvadersNoFrameskip-v4"   "SeaquestNoFrameskip-v4"等

    # 优化器参数学习率的配置
    learningRate: float = 0.00025
    # 神经网络优化器的学习率，控制参数更新的步长大小

    # 强化学习核心参数
    discountFactor: float = 0.99
    # 折扣因子γ，权衡当前奖励与未来奖励的重要性，0.99表示更重视长期回报

    # 训练批次参数
    batchSize: int = 32
    # 每次从经验回放缓冲区中采样的样本数量

    # 经验回放配置
    replayBufferCapacity: int = 50000
    # 经验回放缓冲区的最大容量，存储过去的(state, action, reward, next_state, done)元组

    # 目标网络配置
    targetUpdateFrequency: int = 1000
    # 目标网络更新的频率（每隔多少步更新一次），用于稳定训练

    # 学习启动配置
    learningStartSteps: int = 1000
    # 在开始训练前先收集多少步的经验数据填充回放缓冲区

    learningUpdateFrequency: int = 4
    # 学习更新的频率（每隔多少步进行一次梯度更新）

    # ε-贪婪策略参数（探索-利用权衡）
    initialEpsilon: float = 1.0
    # 初始探索率，1.0表示完全随机探索

    finalEpsilon: float = 0.01
    # 最终探索率，0.1表示10%的概率进行随机探索

    epsilonDecaySteps: int = 50000
    # ε从初始值衰减到最终值所需的步数

    # 环境预处理参数
    frameSkip: int = 4
    # 跳帧数量，每4帧执行一次动作，中间帧重复相同动作，减少计算量

    screenSize: int = 84
    # 预处理后输入神经网络的图像尺寸（84x84像素）

    # 训练周期配置
    trainingEpisodes: int = 1000
    # 总训练回合数，每个回合从环境开始到结束

    targetAverageReward: float = 15.0
    # 目标平均奖励值，用于判断训练是否成功

    # 记录和可视化配置
    saveImages: bool = True
    # 是否保存游戏过程的图像，用于后续分析或可视化

    imageSaveDir: str = "recordedPongEpisodes"
    # 保存游戏图像的目录名称

    numEpisodesToRecord: int = 10
    # 预处理时保存游戏图像的游戏回合数量




