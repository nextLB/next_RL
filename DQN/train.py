"""
    依照本目录的paper目录下的DQNNaturePaper篇文章进行训练的，与原论文在架构方面有一些改动的
"""

# V1.0  2025.10.28      --- by next, 初步实现了使用Renet深度学习模型架构的DQN强化学习模型的搭建与训练等
# V1.1 2025.10.31       --- by next, 手动整理与归纳了V1.0版本的训练代码，兼顾了可读性、鲁棒性、可移植性与可扩展性等

import torch
import Config
import logging
import os
from memory_utils import MemoryManager
import time
from contextlib import contextmanager
from collections import deque, namedtuple

# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ #
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ #
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ #
# 全局存储路径配置
LOG_ROOT_PATH = './log'
DQN_MODELS_ROOT_PATH = './DQN_models'

# 创建一系列数据存储的文件夹
os.makedirs(LOG_ROOT_PATH, exist_ok=True)
os.makedirs(DQN_MODELS_ROOT_PATH, exist_ok=True)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_ROOT_PATH, 'train.log'), mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 经验回放缓冲区
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'nextState', 'done'])

# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ #
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ #
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ #




# 上下文管理器用于内存监控
@contextmanager
def memoryMonitor(operationName: str):
    """监控内存使用的上下文管理器"""
    startTime = time.time()
    gpuMemoryBefore = MemoryManager.getGpuMemoryUsage()
    systemMemoryBefore = MemoryManager.getSystemMemoryUsage()

    try:
        yield
    finally:
        endTime = time.time()
        gpuMemoryAfter = MemoryManager.getGpuMemoryUsage()
        systemMemoryAfter = MemoryManager.getSystemMemoryUsage()

        logger.debug(
            f"{operationName} - "
            f"耗时: {endTime - startTime:.3f}s | "
            f"GPU内存变化: {gpuMemoryAfter[0] - gpuMemoryBefore[0]:+.2f}GB | "
            f"系统内存变化: {systemMemoryAfter - systemMemoryBefore:+.2f}GB"
        )






def main():
    try:
        # 设置内存优化
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
        # 训练配置
        config = Config.Config()
        
        logger.info("使用GPU加速的经验回放缓冲区 - 注意监控GPU显存使用!")



    except KeyboardInterrupt:
        logger.info("训练被用户中断")
    except Exception as e:
        logger.error(f"训练过程中发生错误: {e}")
    finally:
        logger.info("程序执行完毕")



if __name__ == '__main__':
    main()

