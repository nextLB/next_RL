"""
日志配置
"""
import logging
import os


def setupLogging():
    """配置日志系统"""
    # 创建日志目录
    os.makedirs('./log', exist_ok=True)

    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('./log/train.log'),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info("日志系统初始化完成")

    return logger


# 创建全局logger实例
logger = setupLogging()