"""
    PPO模型训练器
"""

import torch
import numpy as np
import logging

class V1_2_PPOTrainer:
    def __init__(self, environment, agent, experienceBuffer, config):
        self.name = 'V1.2_PPOTrainer'
        self.environment = environment
        self.agent = agent
        self.experienceBuffer = experienceBuffer
        self.config = config
        self.logger = logging.getLogger(__name__)

    def train(self):
        # 训练统计
        episodeRewards = []
        movingAverageRewards = []
        bestAverageReward = -float('inf')

        print(episodeRewards, movingAverageRewards, bestAverageReward)


