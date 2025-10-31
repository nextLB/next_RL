"""
日志和监控模块
"""

import logging
import os
import queue
import numpy as np
import torch.multiprocessing as mp
from Config import A3CConfig

# 创建日志目录
os.makedirs('./log', exist_ok=True)

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

def trainingMonitor(trainingQueue: mp.Queue, config: A3CConfig, bestReward: mp.Value = None):
    """训练监控进程"""
    episodeRewards = {}
    bestModelSaved = False

    try:
        while True:
            try:
                data = trainingQueue.get(timeout=30)
                if data is None:  # 终止信号
                    break

                processId = data['processId']
                if processId not in episodeRewards:
                    episodeRewards[processId] = []

                episodeRewards[processId].append(data['reward'])

                # 保存最佳模型
                if bestReward is not None and data.get('isBest', False):
                    logger.info(f"🚀 发现新的最佳模型! 奖励: {data['reward']:.2f}")
                    # 这里可以添加保存最佳模型的逻辑
                    # 注意：在监控进程中无法直接访问共享模型，需要通过队列通知主进程

                if len(episodeRewards[processId]) % 5 == 0:
                    recentRewards = episodeRewards[processId][-5:]
                    avgReward = np.mean(recentRewards)
                    logger.info(f"进程 {processId} - 回合 {data['episode']}: 奖励 = {data['reward']:.2f}, 平均奖励 (最近5回合): {avgReward:.2f}, 总步数: {data['step']}")

                    # 记录训练进展
                    if avgReward > -20.0:
                        logger.info(f"🎯 进程 {processId} 开始学习! 平均奖励: {avgReward:.2f}")

            except queue.Empty:
                continue

    except Exception as e:
        logger.error(f"监控进程错误: {e}")