"""
主训练程序
"""


# V1.0  2025.10.28      --- by next, 初步实现了DQN、PPO、A3C模型架构的训练和推理，但是效果不是很好，有很多地方都还是需要改进和完善的
# V1.1  2025.10.31      --- by next, 针对于V1.0的算法模型训练等进行了整理和归类，便于以后的扩展等
# V1.2  2025.11.3       --- by next, 针对于V1.0和V1.1中所有代码进行重新的手动构建和整理，同时实现了SAC与DDPG这两个新的强化学习的算法模型的逻辑


from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """训练配置参数"""
    environmentName: str = "PongNoFrameskip-v4"



def main():
    # 进行DQN模型的训练



if __name__ == '__main__':
    main()




