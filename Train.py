"""
    主训练程序
"""


# V1.0  2025.10.28      --- by next, 初步实现了DQN、PPO、A3C模型架构的训练和推理，但是效果不是很好，有很多地方都还是需要改进和完善的
# V1.1  2025.10.31      --- by next, 针对于V1.0的算法模型训练等进行了整理和归类，便于以后的扩展等
# V1.2  2025.11.3       --- by next, 针对于V1.0和V1.1中所有代码进行重新的手动构建和整理，同时实现了SAC与DDPG这两个新的强化学习的算法模型的逻辑


from dataclasses import dataclass
from game_environment.PongNoFrameskip_v4_environment import PNFSV4Environment
from game_environment.SpaceInvadersNoFrameskip_v4_environment import SINFSV4Environment
import torch
import logging
import os
from DQN.DQNTrainer import V1_2_DQNTrainer
from DQN.DQNAgent import V1_2_DQNAgent
from DQN.DQNExperience import ExperienceBuffer
from PPO.PPOTrainer import V1_2_PPOTrainer
from PPO.PPOExperience import PPOExperienceBuffer
from PPO.PPOAgent import V1_2_PPOAgent
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
            logging.FileHandler(os.path.join(logDir, 'PPOTrain.log'), mode='w'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)



@dataclass
class TrainingConfig:
    """训练配置参数"""
    version: str = "V1.2"
    environmentName: str = "PongNoFrameskip-v4"
    # environmentName: str = "SpaceInvadersNoFrameskip-v4"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    imageShape: Tuple[int, int, int] = (1, 120, 120)
    numActions: int = 0
    learningRate: float = 0.0001
    trainingEpisodes: int = 1000

    # 下面这些参数主要是DQN的
    initialEpsilon: float = 1.0
    finalEpsilon: float = 0.1
    epsilonDecaySteps: int = 50000
    replayBufferCapacity: int = 20000
    discountFactor: float = 0.99
    targetUpdateFrequency: int = 1000000
    tau: int = 0.01  # 软更新参数
    max_grad_norm: float = 20.0

    # 下面这些参数是PPO算法中特有的
    gamma: float = 0.99
    gaeLambda: float = 0.95
    clipEpsilon: float = 0.2
    valueCoefficient: float = 0.2
    entropyCoefficient: float = 0.01
    ppoEpochs: int = 4
    miniBatchSize: int = 32


# 进行DQN模型的训练
def Train_DQN():
    os.makedirs('./RL_models/DQN_models/', exist_ok=True)
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

    else:
        # 初始化Agent
        DQNAgent = V1_2_DQNAgent(config)

        # 初始化经验池
        Experience = ExperienceBuffer(config.replayBufferCapacity)

        # 初始化训练类
        DQNTrainer = V1_2_DQNTrainer(environment, DQNAgent,  Experience, config)

        # 开始训练
        DQNTrainer.train()





# 进行PPO模型的训练
def Train_PPO():
    os.makedirs('./RL_models/PPO_models/', exist_ok=True)
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
    elif config.environmentName == "SpaceInvadersNoFrameskip-v4":
        environment = SINFSV4Environment(config)
        config.numActions = environment.actionSpace.n
    else:
        environment = PNFSV4Environment(config)
        config.numActions = environment.actionSpace.n

    # 按照版本号进行后续的流程
    if config.version == "V1.2":
        # 初始化Agent
        PPOAgent = V1_2_PPOAgent(config)

        # 初始化经验池
        Experience = PPOExperienceBuffer()

        # 初始化训练类
        PPOTrainer = V1_2_PPOTrainer(environment, PPOAgent,  Experience, config)

        # 开始训练
        PPOTrainer.train()

    else:
        # 初始化Agent
        PPOAgent = V1_2_PPOAgent(config)

        # 初始化经验池
        Experience = PPOExperienceBuffer()

        # 初始化训练类
        PPOTrainer = V1_2_PPOTrainer(environment, PPOAgent,  Experience, config)

        # 开始训练
        PPOTrainer.train()








def main():
    # 进行DQN模型的训练
    Train_DQN()
    # # 进行PPO模型的训练
    # Train_PPO()



if __name__ == '__main__':
    main()




