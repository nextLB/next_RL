"""
A3C训练主程序入口
"""

from Config import A3CConfig
from A3CTrainer import trainA3C, testTrainedModel
from Log import logger
import os

def A3CPongNoFrameskipV4Main():
    """主函数"""
    try:
        # 训练配置
        config = A3CConfig()

        logger.info("开始A3C多进程训练")
        logger.info(f"配置参数: {config}")

        # 开始训练
        trainA3C(config)

        logger.info("A3C训练完成!")

        # 测试最终模型
        modelPath = f"./A3CModels/a3c_model_{config.environmentName.replace('/', '_')}_step_{config.trainingSteps}.pth"
        if os.path.exists(modelPath):
            testTrainedModel(config, modelPath)

    except KeyboardInterrupt:
        logger.info("训练被用户中断")
    except Exception as e:
        logger.error(f"训练过程中发生错误: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        logger.info("程序执行完毕")

if __name__ == "__main__":
    A3CPongNoFrameskipV4Main()