"""
主训练程序
"""


# V1.0  2025.10.28      --- by next, 初步实现了使用Renet深度学习模型架构的DQN强化学习模型的搭建与训练等
# V1.1  2025.10.31      --- by next, 针对于V1.0的训练进行了整理和归类，便于以后的扩展等

import torch
from Config import TrainingConfig
from DQNTrainer import DQNTrainer
from Log import setupLogging
from Memory import MemoryManager



def DQN_PongNoFrameskip_v4_main():
    """主函数"""
    try:
        # 设置日志
        logger = setupLogging()

        # 设置内存优化
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True

        # 训练配置
        config = TrainingConfig()

        logger.info("使用GPU加速的经验回放缓冲区 - 注意监控GPU显存使用!")

        # 创建训练器并开始训练
        trainer = DQNTrainer(config)
        trainedAgent, rewards, movingAverages = trainer.train()

        # 输出训练结果
        logger.info("GPU加速ResNet DQN训练完成!")
        if movingAverages:
            logger.info(f"最终移动平均奖励: {movingAverages[-1]:.2f}")
            logger.info(f"最大移动平均奖励: {max(movingAverages):.2f}")

        trainer.close()

    except KeyboardInterrupt:
        logger.info("训练被用户中断")
    except Exception as e:
        logger.error(f"训练过程中发生错误: {e}")
    finally:
        # 最终清理
        MemoryManager.clearMemory()
        logger.info("程序执行完毕")


if __name__ == "__main__":
    DQN_PongNoFrameskip_v4_main()