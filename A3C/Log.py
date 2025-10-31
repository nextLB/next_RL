"""
日志和监控模块
"""

import logging
import os
import queue
import numpy as np
import torch.multiprocessing as mp
from Config import A3CConfig

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./log/a3c_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def trainingMonitor(trainingQueue: mp.Queue, config: A3CConfig):
    """训练监控进程"""
    episodeRewards = []
    try:
        while True:
            try:
                data = trainingQueue.get(timeout=10)
                if data is None:  # 终止信号
                    break

                episodeRewards.append(data['reward'])

                if len(episodeRewards) % 5 == 0:
                    avgReward = np.mean(episodeRewards[-5:])
                    logger.info(f"进程 {data['processId']} - 回合 {data['episode']}: 奖励 = {data['reward']}, 平均奖励 (最近5回合): {avgReward:.2f}, 总步数: {data['step']}")

            except queue.Empty:
                continue

    except Exception as e:
        logger.error(f"监控进程错误: {e}")