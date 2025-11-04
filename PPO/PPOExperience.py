"""
    PPO经验回放缓冲区
"""

import torch
import numpy as np
from collections import deque

class PPOExperienceBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.logProbs = []





