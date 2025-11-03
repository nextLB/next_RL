"""
    主训练程序
"""


# V1.0  2025.10.28      --- by next, 初步实现了DQN、PPO、A3C模型架构的训练和推理，但是效果不是很好，有很多地方都还是需要改进和完善的
# V1.1  2025.10.31      --- by next, 针对于V1.0的算法模型训练等进行了整理和归类，便于以后的扩展等
# V1.2  2025.11.3       --- by next, 针对于V1.0和V1.1中所有代码进行重新的手动构建和整理，同时实现了SAC与DDPG这两个新的强化学习的算法模型的逻辑


from dataclasses import dataclass
from game_environment.PongNoFrameskip_v4_environment import PNFSV4Environment
import torch
import logging
import os
from DQN.DQNTrainer import V1_2_DQNTrainer
from DQN.DQNAgent import V1_2_DQNAgent
from DQN.DQNExperience import ExperienceBuffer
from typing import Tuple


def setupLogging():
    """配置日志"""
    logDir = './log'
    if not os.path.exists(logDir):
        os.makedirs(logDir)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(logDir, 'DQNTrain.log'), mode='w'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)



@dataclass
class TrainingConfig:
    """训练配置参数"""
    version: str = "V1.2"
    environmentName: str = "PongNoFrameskip-v4"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    imageShape: Tuple[int, int, int] = (1, 120, 120)
    numActions: int = 0
    learningRate: float = 0.00025
    trainingEpisodes: int = 1000
    initialEpsilon: float = 1.0
    finalEpsilon: float = 0.01
    epsilonDecaySteps: int = 100000
    replayBufferCapacity: int = 10000
    discountFactor: float = 0.99
    targetUpdateFrequency: int = 300



# 进行DQN模型的训练
def Train_DQN():
    # 创建日志类
    logger = setupLogging()

    # 创建配置类
    config = TrainingConfig()

    # 设置内存优化
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # 初始化环境类
    if config.environmentName == "PongNoFrameskip-v4":
        environment = PNFSV4Environment(config)
        config.numActions = environment.actionSpace.n
    else:
        environment = PNFSV4Environment(config)
        config.numActions = environment.actionSpace.n

    # 按照版本号进行后续的流程
    if config.version == "V1.2":
        # 初始化Agent
        DQNAgent = V1_2_DQNAgent(config)

        # 初始化经验池
        Experience = ExperienceBuffer(config.replayBufferCapacity)

        # 初始化训练类
        DQNTrainer = V1_2_DQNTrainer(environment, DQNAgent,  Experience, config)

        # 开始训练
        DQNTrainer.train()







def main():
    # 进行DQN模型的训练
    Train_DQN()



if __name__ == '__main__':
    main()




