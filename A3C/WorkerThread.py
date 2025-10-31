"""
Worker thread implementation for A3C
"""

import threading
import torch
import numpy as np
from typing import List, Tuple
import logging
import time
from Config import device
from A3CAgent import A3CAgent, A3CNetwork
from Environment import EnvironmentWrapper
from Config import A3CConfig
import torch.nn.functional as F
from collections import deque


class A3CWorker(threading.Thread):
    """A3C Worker Thread"""

    def __init__(self, workerId: int, globalAgent: A3CAgent, config: A3CConfig):
        super().__init__()
        self.workerId = workerId
        self.globalAgent = globalAgent
        self.config = config

        # Local network
        self.localNetwork = A3CNetwork((2, 84, 84), 4).to(device)

        # Environment
        self.envWrapper = None

        # Worker statistics
        self.episodeRewards = deque(maxlen=100)
        self.episodeLengths = deque(maxlen=100)

        logging.info(f"Worker thread {workerId} created")

    def run(self):
        """Main worker loop"""
        try:
            # Initialize environment
            self.envWrapper = EnvironmentWrapper(
                self.config.environmentName,
                self.config.frameSkip,
                self.config.screenSize
            )
            logging.info(f"Worker thread {self.workerId} environment initialized")

            episodeCount = 0

            while self.globalAgent.globalStep < self.config.trainingTimesteps:
                # Synchronize networks
                self.localNetwork.load_state_dict(self.globalAgent.globalNetwork.state_dict())

                # Reset environment
                state = self.envWrapper.reset().to(device)

                episodeReward = 0
                episodeLength = 0
                done = False

                # Store trajectory
                states, actions, rewards, logProbs, values = [], [], [], [], []

                while not done and episodeLength < self.config.maxEpisodeLength:
                    # Select action
                    action, logProb, value = self.localNetwork.getAction(state.unsqueeze(0))

                    # Execute action
                    nextState, reward, done, info = self.envWrapper.step(action.item())
                    nextState = nextState.to(device)

                    # Reward clipping
                    if self.config.rewardClip:
                        reward = np.clip(reward, -1, 1)

                    # Store data
                    states.append(state)
                    actions.append(action)
                    rewards.append(reward)
                    logProbs.append(logProb)
                    values.append(value)

                    # Update state
                    state = nextState
                    episodeReward += reward
                    episodeLength += 1

                    # n-step update or episode end
                    if len(states) >= self.config.nStep or done:
                        policyLoss, valueLoss, entropy = self.updateWithTrajectory(
                            states, actions, rewards, logProbs, values, done, nextState
                        )

                        # Update training statistics
                        if policyLoss is not None:
                            statsUpdate = {
                                'valueLosses': valueLoss.item(),
                                'policyLosses': policyLoss.item(),
                                'entropies': entropy.item()
                            }
                            self.globalAgent.updateTrainingStats(statsUpdate)

                        states, actions, rewards, logProbs, values = [], [], [], [], []

                # Episode completed
                episodeCount += 1
                self.globalAgent.incrementCounters(episodes=1)

                # Update worker statistics
                self.episodeRewards.append(episodeReward)
                self.episodeLengths.append(episodeLength)

                averageReward = np.mean(self.episodeRewards) if self.episodeRewards else 0
                averageLength = np.mean(self.episodeLengths) if self.episodeLengths else 0

                # Check if best reward
                if self.globalAgent.updateBestReward(episodeReward):
                    self.globalAgent.saveModel(self.config.bestModelPath)

                # Log episode results
                if episodeCount % self.config.logInterval == 0:
                    logging.info(f"Worker {self.workerId} - Episode {episodeCount}: "
                               f"Reward={episodeReward:.1f}, Avg Reward={averageReward:.1f}, "
                               f"Length={episodeLength}, Steps={self.globalAgent.globalStep}")

        except Exception as e:
            logging.error(f"Worker thread {self.workerId} error: {e}")
            import traceback
            logging.error(traceback.format_exc())
        finally:
            if self.envWrapper:
                self.envWrapper.close()

    def updateWithTrajectory(self, states, actions, rewards, logProbs, values, done, nextState):
        """Trajectory update"""
        try:
            # Prepare data tensors
            statesTensor = torch.stack(states).to(device)
            actionsTensor = torch.stack(actions).to(device)
            logProbsTensor = torch.stack(logProbs).to(device)
            valuesTensor = torch.stack(values).to(device)

            # Calculate returns
            with torch.no_grad():
                if not done:
                    _, nextValue = self.localNetwork(nextState.unsqueeze(0))
                    returns = self.computeReturns(rewards, nextValue.item(), done)
                else:
                    returns = self.computeReturns(rewards, 0.0, done)

            returnsTensor = torch.tensor(returns, dtype=torch.float32, device=device)

            # Calculate advantages
            advantages = returnsTensor - valuesTensor.detach()

            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Calculate losses - Fixed shape mismatch
            policyLoss = -(logProbsTensor * advantages).mean()

            # Fix: Ensure tensors have same shape for MSE loss
            valueLoss = F.mse_loss(valuesTensor, returnsTensor.unsqueeze(1))

            # Calculate entropy
            actionProbs, _ = self.localNetwork(statesTensor)
            actionDistribution = torch.distributions.Categorical(actionProbs)
            entropy = actionDistribution.entropy().mean()

            # Total loss
            totalLoss = (policyLoss +
                        self.config.valueLossCoeff * valueLoss -
                        self.config.entropyCoeff * entropy)

            # Calculate gradients
            self.localNetwork.zero_grad()
            totalLoss.backward()

            # Collect gradients
            gradients = []
            for localParam, globalParam in zip(
                self.localNetwork.parameters(),
                self.globalAgent.globalNetwork.parameters()
            ):
                if localParam.grad is not None:
                    gradients.append((globalParam, localParam.grad.clone()))

            # Update global model
            self.globalAgent.updateModel(gradients)
            self.globalAgent.incrementCounters(steps=len(states))

            return policyLoss, valueLoss, entropy

        except Exception as e:
            logging.error(f"Worker {self.workerId} trajectory update failed: {e}")
            return None, None, None

    def computeReturns(self, rewards: List[float], lastValue: float, done: bool) -> List[float]:
        """Compute n-step returns"""
        returns = []
        R = lastValue if not done else 0.0

        for r in reversed(rewards):
            R = r + self.config.discountFactor * R
            returns.insert(0, R)

        return returns


