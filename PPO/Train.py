"""
PPO训练主程序
"""

# V1.0  2025.10.28      --- by next, 初步实现了使用Renet深度学习模型架构的PPO强化学习模型的搭建与训练等
# V1.1  2025.10.31      --- by next, 针对于V1.0的训练进行了整理和归类，便于以后的扩展等
# V1.1  2025.11.1       --- by next,



import torch
import os
from Log import logger
from Config import PPOConfig, device, LunarLanderConfig
from PPOTrainer import PPOTrainer
from Memory import MemoryManager
from Environment import LunarLanderEnvironment



def PPO_PongNoFrameskip_v4_main():
    """主函数"""
    try:
        # 设置内存优化
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True

        # 创建必要的目录
        os.makedirs('./log', exist_ok=True)
        os.makedirs('./PPO_models', exist_ok=True)

        # PPO训练配置 - 按照论文中的超参数
        config = PPOConfig()

        logger.info("开始PPO训练!")
        logger.info(f"使用设备: {device}")

        # 创建训练器并开始训练
        trainer = PPOTrainer(config)
        trainedAgent, rewards, movingAverages = trainer.train()

        # 输出训练结果
        logger.info("PPO训练完成!")
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


# 月球着陆游戏训练主程序
def PPO_LunarLander_main():
    config = LunarLanderConfig()
    environment = LunarLanderEnvironment(config)
    environment.save_episode_images()

if __name__ == "__main__":
    # PPO_PongNoFrameskip_v4_main()
    PPO_LunarLander_main()

