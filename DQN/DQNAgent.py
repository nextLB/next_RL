"""
    关于DQNAgent的模型搭建文件
"""

import torch
import torch.nn as nn



class ResNetDeepQNetwork(nn.Module):
    def __init__(self, inputShape, numActions):
        super(ResNetDeepQNetwork, self).__init__()
        self.conv1 = nn.Conv2d(inputShape[0], 16, 3, 1, 1)




class V1_2_DQNAgent:
    def __init__(self, config):
        self.name = "V1.2_DQNAgent"
        self.config = config
        self.policyNetwork = ResNetDeepQNetwork(self.config.imageShape, self.config.numActions)
        self.targetNetwork = ResNetDeepQNetwork(self.config.imageShape, self.config.numActions)

