"""
A3C Agent and Neural Network
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Dict, Any
import os
import logging
from threading import Lock
from Config import A3CConfig, device


class A3CNetwork(nn.Module):
    """A3C Neural Network"""

    def __init__(self, inputShape: Tuple[int, int, int], numActions: int):
        super().__init__()

        self.convLayers = nn.Sequential(
            nn.Conv2d(inputShape[0], 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7))
        )

        # Calculate feature size
        with torch.no_grad():
            sampleInput = torch.zeros(1, *inputShape)
            convOutput = self.convLayers(sampleInput)
            self.featureSize = convOutput.view(1, -1).size(1)

        # Policy and value heads
        self.policyHead = nn.Linear(self.featureSize, numActions)
        self.valueHead = nn.Linear(self.featureSize, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batchSize = x.size(0)
        features = self.convLayers(x).view(batchSize, -1)

        policyLogits = self.policyHead(features)
        actionProbs = F.softmax(policyLogits, dim=-1)
        stateValue = self.valueHead(features).squeeze(-1)

        return actionProbs, stateValue

    def getAction(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select action"""
        with torch.no_grad():
            actionProbs, stateValue = self.forward(state)
            actionDistribution = torch.distributions.Categorical(actionProbs)
            action = actionDistribution.sample()
            logProbability = actionDistribution.log_prob(action)

            return action, logProbability, stateValue


class A3CAgent:
    """A3C Agent"""

    def __init__(self, stateShape: Tuple[int, int, int], numActions: int, config: A3CConfig):
        self.globalNetwork = A3CNetwork(stateShape, numActions).to(device)
        self.optimizer = optim.RMSprop(self.globalNetwork.parameters(),
                                     lr=config.learningRate,
                                     alpha=0.99,
                                     eps=1e-5)

        # Share network
        self.globalNetwork.share_memory()

        self.globalStep = 0
        self.episodesCompleted = 0
        self.bestReward = -float('inf')
        self.lock = Lock()
        self.config = config

        # Training statistics
        self.trainingStats = {
            'episodeRewards': [],
            'episodeLengths': [],
            'valueLosses': [],
            'policyLosses': [],
            'entropies': []
        }

        logging.info("A3C Agent initialized")

    def updateModel(self, gradients: List[Tuple[torch.Tensor, torch.Tensor]]):
        """Update global model"""
        with self.lock:
            self.optimizer.zero_grad()

            # Apply gradients
            for param, grad in gradients:
                if grad is not None:
                    if param.grad is None:
                        param.grad = grad.clone()
                    else:
                        param.grad += grad

            # Gradient clipping
            if self.config.gradientClip:
                torch.nn.utils.clip_grad_norm_(self.globalNetwork.parameters(), self.config.maxGradNorm)

            # Optimization step
            self.optimizer.step()

    def incrementCounters(self, steps: int = 1, episodes: int = 0):
        """Update counters"""
        with self.lock:
            self.globalStep += steps
            self.episodesCompleted += episodes

    def saveModel(self, filePath: str):
        """Save model"""
        with self.lock:
            torch.save({
                'globalStep': self.globalStep,
                'episodesCompleted': self.episodesCompleted,
                'modelStateDict': self.globalNetwork.state_dict(),
                'optimizerStateDict': self.optimizer.state_dict(),
                'bestReward': self.bestReward,
                'trainingStats': self.trainingStats
            }, filePath)
            logging.info(f"Model saved to: {filePath}")

    def loadModel(self, filePath: str):
        """Load model"""
        with self.lock:
            if os.path.exists(filePath):
                try:
                    checkpoint = torch.load(filePath, map_location=device)

                    # Load model state
                    modelState = checkpoint['modelStateDict']

                    # Handle possible key mismatches
                    currentModelState = self.globalNetwork.state_dict()

                    # Only load matching keys
                    filteredState = {k: v for k, v in modelState.items()
                                   if k in currentModelState and currentModelState[k].shape == v.shape}

                    # Load matching parameters
                    currentModelState.update(filteredState)
                    self.globalNetwork.load_state_dict(currentModelState)

                    # Load other states
                    self.optimizer.load_state_dict(checkpoint['optimizerStateDict'])
                    self.globalStep = checkpoint['globalStep']
                    self.episodesCompleted = checkpoint['episodesCompleted']
                    self.bestReward = checkpoint['bestReward']
                    self.trainingStats = checkpoint.get('trainingStats', self.trainingStats)

                    logging.info(f"Model successfully loaded from {filePath}")
                    logging.info(f"Loaded {len(filteredState)}/{len(modelState)} parameters")
                    return True

                except Exception as e:
                    logging.warning(f"Error loading model {filePath}: {e}")
                    logging.info("Starting with fresh model...")
                    return False
            else:
                logging.info(f"No existing model found at {filePath}, starting fresh")
                return False

    def updateBestReward(self, reward: float):
        """Update best reward"""
        with self.lock:
            if reward > self.bestReward:
                oldReward = self.bestReward
                self.bestReward = reward
                logging.info(f"New best reward: {oldReward:.1f} -> {reward:.1f}")
                return True
        return False

    def updateTrainingStats(self, statsUpdate: Dict[str, float]):
        """Update training statistics"""
        with self.lock:
            for key, value in statsUpdate.items():
                if key in self.trainingStats:
                    self.trainingStats[key].append(value)


