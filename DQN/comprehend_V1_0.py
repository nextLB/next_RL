"""
    理解V1.0.py程序的程序文件
"""


import torch
import logging

from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """训练配置参数 - 用于深度强化学习算法（如DQN）的配置"""

    # 环境配置
    environmentName: str = "PongNoFrameskip-v4"
    # 使用的强化学习环境名称，PongNoFrameskip-v4是Atari乒乓球游戏的无跳帧版本

    # 优化器参数
    learningRate: float = 0.00025
    # 神经网络优化器的学习率，控制参数更新的步长大小

    # 强化学习核心参数
    discountFactor: float = 0.99
    # 折扣因子γ，权衡当前奖励与未来奖励的重要性，0.99表示更重视长期回报

    # 训练批次参数
    batchSize: int = 32
    # 每次从经验回放缓冲区中采样的样本数量

    # 经验回放配置
    replayBufferCapacity: int = 30000
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

    finalEpsilon: float = 0.1
    # 最终探索率，0.1表示10%的概率进行随机探索

    epsilonDecaySteps: int = 50000
    # ε从初始值衰减到最终值所需的步数

    # 环境预处理参数
    frameSkip: int = 4
    # 跳帧数量，每4帧执行一次动作，中间帧重复相同动作，减少计算量

    screenSize: int = 84
    # 预处理后输入神经网络的图像尺寸（84x84像素）

    # 网络架构选择
    useSimpleResNet: bool = True
    # 是否使用简化的ResNet架构，True使用ResNet，False可能使用普通CNN

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
    # 记录图像的游戏回合数量


# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./log/dqn_resnet_gpu_memory.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DQNTrainer:
    """DQN训练器 - GPU加速版本"""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.environment = None
        self.preprocessedEnvironment = None
        self.agent = None



def main():
    """主函数"""
    try:
        # 设置内存优化
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
        # 训练配置
        config = TrainingConfig()

        logger.info("使用GPU加速的经验回放缓冲区 - 注意监控GPU显存使用!")

        # 创建训练器并开始训练
        trainer = DQNTrainer(config)


    except KeyboardInterrupt:
        logger.info("训练被用户中断")
    except Exception as e:
        logger.error(f"训练过程中发生错误: {e}")
    finally:
        logger.info("程序执行完毕")


if __name__ == "__main__":
    main()




