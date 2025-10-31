"""
日志配置模块
"""

import logging
import time
from contextlib import contextmanager
import os
os.makedirs('./log', exist_ok=True)


def setupLogging():
    """配置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('./log/train.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


@contextmanager
def memoryMonitor(operationName: str):
    """监控内存使用的上下文管理器"""
    startTime = time.time()

    try:
        yield
    finally:
        endTime = time.time()
        logger = logging.getLogger(__name__)
        logger.debug(f"{operationName} - 耗时: {endTime - startTime:.3f}s")


