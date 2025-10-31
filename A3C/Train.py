"""
A3C训练主程序入口
"""

from Config import A3CConfig
from A3CTrainer import trainA3C, testTrainedModel
from Log import logger
import os
import argparse

def parseArguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='A3C Training')
    parser.add_argument('--resume', type=str, help='从检查点恢复训练')
    parser.add_argument('--test', type=str, help='测试训练好的模型')
    parser.add_argument('--episodes', type=int, default=5, help='测试回合数')
    return parser.parse_args()

def A3CPongNoFrameskipV4Main():
    """主函数"""
    args = parseArguments()

    try:
        # 训练配置
        config = A3CConfig()

        if args.test:
            # 测试模式
            logger.info(f"开始测试模型: {args.test}")
            testTrainedModel(config, args.test, args.episodes)
            return

        logger.info("开始A3C多进程训练")
        logger.info(f"配置参数: {config}")

        # 开始训练
        trainA3C(config, args.resume)

        logger.info("A3C训练完成!")

        # 测试最终模型
        modelPath = f"./A3CModels/a3c_model_{config.environmentName.replace('/', '_')}_step_{config.trainingSteps}.pth"
        if os.path.exists(modelPath):
            testTrainedModel(config, modelPath, numEpisodes=3)

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