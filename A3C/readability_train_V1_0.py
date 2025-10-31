"""
A3C Training Module - Optimized Version
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import gymnasium as gym
from PIL import Image
import logging
from typing import Tuple, List, Dict, Any, Optional
import os
import time
import threading
from threading import Lock
import random


# =============================== Configuration Class ===============================
class A3CConfig:
    """A3C training configuration parameters"""
    def __init__(self):
        self.environmentName = "BreakoutNoFrameskip-v4"
        self.learningRate = 0.0007
        self.discountFactor = 0.99
        self.entropyCoeff = 0.01
        self.valueLossCoeff = 0.5
        self.maxGradNorm = 40.0
        self.nStep = 5
        self.numProcesses = 4  # Increased for better performance
        self.trainingTimesteps = 1000000  # Increased training steps
        self.frameSkip = 4
        self.screenSize = 84
        self.useLSTM = False
        self.logInterval = 10  # More reasonable logging interval
        self.saveInterval = 10000  # Save every 10k steps
        self.maxEpisodeLength = 10000
        self.modelSavePath = "./A3CModels"
        self.bestModelPath = "./A3CModels/best_model.pth"
        self.checkpointInterval = 50000  # Checkpoint every 50k steps


# =============================== Global Settings ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("a3c_training.log")
    ]
)
logger = logging.getLogger("A3C_Training")


# =============================== Neural Network ===============================
class A3CNetwork(nn.Module):
    """A3C Neural Network with improved architecture"""

    def __init__(self, inputShape: Tuple[int, int, int], numActions: int):
        super().__init__()

        # Enhanced convolutional layers with batch normalization
        self.convLayers = nn.Sequential(
            nn.Conv2d(inputShape[0], 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((6, 6))  # Changed to fixed size for more stable features
        )

        # Calculate feature size after conv layers
        with torch.no_grad():
            sampleInput = torch.zeros(1, *inputShape)
            convOutput = self.convLayers(sampleInput)
            self.featureSize = convOutput.view(1, -1).size(1)

        # Policy and value heads
        self.policyHead = nn.Sequential(
            nn.Linear(self.featureSize, 512),
            nn.ReLU(),
            nn.Linear(512, numActions)
        )

        self.valueHead = nn.Sequential(
            nn.Linear(self.featureSize, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batchSize = x.size(0)
        features = self.convLayers(x).view(batchSize, -1)

        policyLogits = self.policyHead(features)
        actionProbs = F.softmax(policyLogits, dim=-1)
        stateValue = self.valueHead(features).squeeze(-1)

        return actionProbs, stateValue

    def getAction(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select action with exploration"""
        with torch.no_grad():
            actionProbs, stateValue = self.forward(state)
            actionDistribution = torch.distributions.Categorical(actionProbs)
            action = actionDistribution.sample()
            logProbability = actionDistribution.log_prob(action)

            return action, logProbability, stateValue


# =============================== Environment Wrapper ===============================
class EnvironmentWrapper:
    """Enhanced environment wrapper with better preprocessing"""

    def __init__(self, envName: str, frameSkip: int = 4, screenSize: int = 84):
        self.env = gym.make(envName, render_mode='rgb_array')
        self.frameSkip = frameSkip
        self.screenSize = screenSize
        self.frameBuffer = deque(maxlen=4)
        self.lives = 0

    def reset(self) -> torch.Tensor:
        state, _ = self.env.reset()
        processedState = self.preprocessFrame(state)

        # Initialize frame buffer
        self.frameBuffer.clear()
        for _ in range(4):
            self.frameBuffer.append(processedState)

        # Get initial lives for Breakout
        self.lives = self.env.unwrapped.ale.lives()

        return torch.tensor(np.stack(self.frameBuffer), dtype=torch.float32)

    def step(self, action: int) -> Tuple[torch.Tensor, float, bool, Dict]:
        totalReward = 0.0
        done = False
        info = {}

        for _ in range(self.frameSkip):
            nextState, reward, terminated, truncated, stepInfo = self.env.step(action)
            totalReward += reward

            # Check for life loss in Breakout
            currentLives = self.env.unwrapped.ale.lives()
            if currentLives < self.lives:
                reward = -1.0  # Penalize life loss
                self.lives = currentLives

            info.update(stepInfo)
            if terminated or truncated:
                done = True
                break

        processedNextState = self.preprocessFrame(nextState)
        self.frameBuffer.append(processedNextState)

        nextStateTensor = torch.tensor(np.stack(self.frameBuffer), dtype=torch.float32)
        return nextStateTensor, totalReward, done, info

    def preprocessFrame(self, frame: np.ndarray) -> np.ndarray:
        """Enhanced frame preprocessing"""
        if len(frame.shape) == 3:
            # Convert to grayscale and crop irrelevant parts
            frame = np.mean(frame, axis=2)
            frame = frame[34:194, :]  # Crop score and borders for Breakout

        img = Image.fromarray(frame.astype(np.uint8))
        img = img.resize((self.screenSize, self.screenSize), Image.BILINEAR)
        frame = np.array(img).astype(np.float32) / 255.0

        return frame

    @property
    def actionSpace(self):
        return self.env.action_space

    def close(self):
        self.env.close()


# =============================== A3C Agent ===============================
class A3CAgent:
    """Enhanced A3C Agent with better training management"""

    def __init__(self, stateShape: Tuple[int, int, int], numActions: int, config: A3CConfig):
        self.globalNetwork = A3CNetwork(stateShape, numActions).to(device)
        self.optimizer = optim.RMSprop(self.globalNetwork.parameters(),
                                     lr=config.learningRate,
                                     alpha=0.99,
                                     eps=1e-5)

        # Shared network
        self.globalNetwork.share_memory()

        self.globalStep = 0
        self.episodesCompleted = 0
        self.bestReward = -float('inf')
        self.lock = Lock()
        self.config = config

        # Training statistics
        self.trainingStats = {
            'episode_rewards': [],
            'episode_lengths': [],
            'value_losses': [],
            'policy_losses': [],
            'entropies': []
        }

        logger.info("A3C Agent initialized with RMSprop optimizer")

    def updateModel(self, gradients: List[Tuple[torch.Tensor, torch.Tensor]]):
        """Update global model with gradients"""
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
            torch.nn.utils.clip_grad_norm_(self.globalNetwork.parameters(), self.config.maxGradNorm)

            # Optimization step
            self.optimizer.step()

    def incrementCounters(self, steps: int = 1, episodes: int = 0):
        """Update counters with thread safety"""
        with self.lock:
            self.globalStep += steps
            self.episodesCompleted += episodes

    def saveModel(self, filePath: str):
        """Save model with training state"""
        with self.lock:
            torch.save({
                'globalStep': self.globalStep,
                'episodesCompleted': self.episodesCompleted,
                'modelStateDict': self.globalNetwork.state_dict(),
                'optimizerStateDict': self.optimizer.state_dict(),
                'bestReward': self.bestReward,
                'trainingStats': self.trainingStats,
                'config': self.config.__dict__
            }, filePath)
            logger.info(f"Model saved to: {filePath}")

    def loadModel(self, filePath: str):
        """Load model and training state"""
        with self.lock:
            if os.path.exists(filePath):
                checkpoint = torch.load(filePath, map_location=device)
                self.globalNetwork.load_state_dict(checkpoint['modelStateDict'])
                self.optimizer.load_state_dict(checkpoint['optimizerStateDict'])
                self.globalStep = checkpoint['globalStep']
                self.episodesCompleted = checkpoint['episodesCompleted']
                self.bestReward = checkpoint['bestReward']
                self.trainingStats = checkpoint.get('trainingStats', self.trainingStats)
                logger.info(f"Model loaded from {filePath}")
                return True
            else:
                logger.warning(f"Model file {filePath} does not exist")
                return False

    def updateBestReward(self, reward: float):
        """Update best reward with thread safety"""
        with self.lock:
            if reward > self.bestReward:
                self.bestReward = reward
                return True
        return False

    def updateTrainingStats(self, statsUpdate: Dict[str, float]):
        """Update training statistics"""
        with self.lock:
            for key, value in statsUpdate.items():
                if key in self.trainingStats:
                    self.trainingStats[key].append(value)


# =============================== Worker Thread ===============================
class A3CWorker(threading.Thread):
    """Enhanced A3C Worker Thread with better training logic"""

    def __init__(self, workerId: int, globalAgent: A3CAgent, config: A3CConfig):
        super().__init__()
        self.workerId = workerId
        self.globalAgent = globalAgent
        self.config = config

        # Local network
        self.localNetwork = A3CNetwork((4, 84, 84), 4).to(device)

        # Environment
        self.envWrapper = None

        # Worker statistics
        self.episodeRewards = deque(maxlen=100)
        self.episodeLengths = deque(maxlen=100)

        logger.info(f"Worker thread {workerId} created")

    def run(self):
        """Main worker loop with enhanced training logic"""
        try:
            # Initialize environment
            self.envWrapper = EnvironmentWrapper(
                self.config.environmentName,
                self.config.frameSkip,
                self.config.screenSize
            )
            logger.info(f"Worker thread {self.workerId} environment initialized")

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
                    nextState, reward, done, _ = self.envWrapper.step(action.item())
                    nextState = nextState.to(device)

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
                                'value_losses': valueLoss.item(),
                                'policy_losses': policyLoss.item(),
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

                # Check for new best reward
                if self.globalAgent.updateBestReward(episodeReward):
                    self.globalAgent.saveModel(self.config.bestModelPath)
                    logger.info(f"Worker {self.workerId} - New best reward: {episodeReward:.1f}")

                # Log episode results
                if episodeCount % self.config.logInterval == 0:
                    logger.info(f"Worker {self.workerId} - Episode {episodeCount}: "
                               f"Reward={episodeReward:.1f}, Avg Reward={averageReward:.1f}, "
                               f"Length={episodeLength}, Avg Length={averageLength:.1f}")

        except Exception as e:
            logger.error(f"Worker thread {self.workerId} error: {e}")
            import traceback
            logger.error(traceback.format_exc())
        finally:
            if self.envWrapper:
                self.envWrapper.close()

    def updateWithTrajectory(self, states, actions, rewards, logProbs, values, done, nextState):
        """Enhanced trajectory update with proper loss calculation"""
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

            # Normalize advantages for stability
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Calculate losses
            policyLoss = -(logProbsTensor * advantages).mean()
            valueLoss = F.mse_loss(valuesTensor, returnsTensor)

            # Calculate entropy
            actionProbs, _ = self.localNetwork(statesTensor)
            actionDistribution = torch.distributions.Categorical(actionProbs)
            entropy = actionDistribution.entropy().mean()

            # Total loss
            totalLoss = (policyLoss +
                        self.config.valueLossCoeff * valueLoss -
                        self.config.entropyCoeff * entropy)

            # Compute gradients
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
            logger.error(f"Worker {self.workerId} trajectory update failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None, None, None

    def computeReturns(self, rewards: List[float], lastValue: float, done: bool) -> List[float]:
        """Compute n-step returns"""
        returns = []
        R = lastValue if not done else 0.0

        for r in reversed(rewards):
            R = r + self.config.discountFactor * R
            returns.insert(0, R)

        return returns


# =============================== Trainer ===============================
class A3CTrainer:
    """Enhanced A3C Trainer with better monitoring and management"""

    def __init__(self, config: A3CConfig):
        self.config = config
        self.globalAgent = None
        self.workers = []

        # Training monitoring
        self.startTime = None
        self.lastLogTime = None

    def train(self):
        """Start enhanced training process"""
        # Create model directory
        os.makedirs(self.config.modelSavePath, exist_ok=True)

        # Initialize global agent
        stateShape = (4, self.config.screenSize, self.config.screenSize)
        numActions = 4  # Breakout actions

        self.globalAgent = A3CAgent(stateShape, numActions, self.config)

        # Try to load existing model
        modelLoaded = self.globalAgent.loadModel(self.config.bestModelPath)

        # Create worker threads
        self.workers = []
        for i in range(self.config.numProcesses):
            worker = A3CWorker(i, self.globalAgent, self.config)
            self.workers.append(worker)

        logger.info(f"Starting training with {self.config.numProcesses} worker threads")
        if modelLoaded:
            logger.info(f"Resuming from checkpoint, current global step: {self.globalAgent.globalStep}")

        # Start timing
        self.startTime = time.time()
        self.lastLogTime = time.time()

        # Start worker threads
        for worker in self.workers:
            worker.start()
            logger.info(f"Started worker thread {worker.workerId}")
            time.sleep(0.5)  # Stagger thread starts

        # Monitor training progress
        try:
            lastGlobalStep = self.globalAgent.globalStep
            lastSaveStep = self.globalAgent.globalStep

            while self.globalAgent.globalStep < self.config.trainingTimesteps:
                time.sleep(5)  # Check every 5 seconds

                currentStep = self.globalAgent.globalStep
                currentTime = time.time()

                # Calculate training speed
                if currentStep > lastGlobalStep:
                    stepsPerSec = (currentStep - lastGlobalStep) / (currentTime - self.lastLogTime)
                    elapsedTime = currentTime - self.startTime
                    remainingTime = (self.config.trainingTimesteps - currentStep) / stepsPerSec if stepsPerSec > 0 else 0

                    # Log progress
                    logger.info(f"Progress: {currentStep}/{self.config.trainingTimesteps} "
                               f"({currentStep/self.config.trainingTimesteps*100:.1f}%) | "
                               f"Speed: {stepsPerSec:.1f} steps/sec | "
                               f"Elapsed: {self.formatTime(elapsedTime)} | "
                               f"ETA: {self.formatTime(remainingTime)} | "
                               f"Episodes: {self.globalAgent.episodesCompleted} | "
                               f"Best Reward: {self.globalAgent.bestReward:.1f}")

                    lastGlobalStep = currentStep
                    self.lastLogTime = currentTime

                    # Save checkpoint
                    if currentStep - lastSaveStep >= self.config.checkpointInterval:
                        checkpointPath = os.path.join(
                            self.config.modelSavePath,
                            f"checkpoint_step_{currentStep}.pth"
                        )
                        self.globalAgent.saveModel(checkpointPath)
                        lastSaveStep = currentStep
                        logger.info(f"Checkpoint saved at step {currentStep}")

                else:
                    logger.warning("No training progress detected in the last interval...")

                    # Check thread status
                    aliveCount = sum(1 for w in self.workers if w.is_alive())
                    if aliveCount == 0:
                        logger.error("All worker threads have stopped!")
                        break
                    else:
                        logger.info(f"{aliveCount}/{self.config.numProcesses} worker threads still alive")

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        except Exception as e:
            logger.error(f"Training error: {e}")
            import traceback
            logger.error(traceback.format_exc())
        finally:
            # Save final model
            finalModelPath = os.path.join(self.config.modelSavePath, "final_model.pth")
            self.globalAgent.saveModel(finalModelPath)

            # Wait for threads to finish
            for worker in self.workers:
                if worker.is_alive():
                    worker.join(timeout=5.0)
                    logger.info(f"Worker thread {worker.workerId} terminated")

            totalTime = time.time() - self.startTime
            logger.info(f"Training completed! Final global step: {self.globalAgent.globalStep}")
            logger.info(f"Best reward achieved: {self.globalAgent.bestReward:.1f}")
            logger.info(f"Total training time: {self.formatTime(totalTime)}")

            # Print final statistics
            self.printTrainingStatistics()

    def formatTime(self, seconds: float) -> str:
        """Format time in human readable format"""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            return f"{seconds/3600:.1f}h"

    def printTrainingStatistics(self):
        """Print final training statistics"""
        stats = self.globalAgent.trainingStats

        if stats['episode_rewards']:
            avgReward = np.mean(stats['episode_rewards'][-100:])  # Last 100 episodes
            avgLength = np.mean(stats['episode_lengths'][-100:])

            logger.info("Final Training Statistics:")
            logger.info(f"  Average Reward (last 100): {avgReward:.2f}")
            logger.info(f"  Average Length (last 100): {avgLength:.2f}")
            logger.info(f"  Total Episodes: {len(stats['episode_rewards'])}")

            if stats['value_losses']:
                logger.info(f"  Final Value Loss: {stats['value_losses'][-1]:.4f}")
            if stats['policy_losses']:
                logger.info(f"  Final Policy Loss: {stats['policy_losses'][-1]:.4f}")


# =============================== Main Function ===============================
def main():
    """Main training function"""
    try:
        # Configuration
        config = A3CConfig()

        print("A3C Reinforcement Learning Training System")
        print("=" * 50)
        print(f"Environment: {config.environmentName}")
        print(f"Training Steps: {config.trainingTimesteps}")
        print(f"Number of Workers: {config.numProcesses}")
        print(f"Device: {device}")
        print("=" * 50)

        # Start training
        logger.info("Starting A3C training!")

        trainer = A3CTrainer(config)
        trainer.train()

        logger.info("Training completed successfully!")

    except Exception as e:
        logger.error(f"Error during training execution: {e}")
        import traceback
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()
