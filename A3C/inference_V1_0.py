"""
A3C Real-time Inference with Visualization - Fixed Version
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from PIL import Image
import matplotlib.pyplot as plt
from collections import deque
import time
import os
from typing import Tuple, List, Dict, Any

# =============================== Neural Network ===============================
class A3CNetwork(nn.Module):
    """A3C Neural Network - Same as training"""

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

        with torch.no_grad():
            sampleInput = torch.zeros(1, *inputShape)
            convOutput = self.convLayers(sampleInput)
            self.featureSize = convOutput.view(1, -1).size(1)

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

# =============================== Environment Wrapper ===============================
class InferenceEnvironmentWrapper:
    """Environment wrapper for inference with real-time visualization"""

    def __init__(self, envName: str, frameSkip: int = 4, screenSize: int = 84):
        self.env = gym.make(envName, render_mode='rgb_array')
        self.frameSkip = frameSkip
        self.screenSize = screenSize
        self.frameBuffer = deque(maxlen=2)
        self.lives = 0
        self.originalLives = 5
        self.currentRawFrame = None

    def reset(self) -> torch.Tensor:
        # 修复Gymnasium API调用[citation:3]
        state, _ = self.env.reset()
        processedState = self.preprocessFrame(state)
        self.currentRawFrame = state

        self.frameBuffer.clear()
        for _ in range(2):
            self.frameBuffer.append(processedState)

        self.lives = self.env.unwrapped.ale.lives()
        self.originalLives = self.lives

        return torch.tensor(np.stack(self.frameBuffer), dtype=torch.float32)

    def step(self, action: int) -> Tuple[torch.Tensor, float, bool, Dict]:
        totalReward = 0.0
        done = False
        lostLife = False

        for _ in range(self.frameSkip):
            # 修复Gymnasium API调用[citation:3]
            nextState, reward, terminated, truncated, stepInfo = self.env.step(action)
            self.currentRawFrame = nextState

            currentLives = self.env.unwrapped.ale.lives()
            if currentLives < self.lives:
                lostLife = True
                self.lives = currentLives
                reward = -1.0
            elif reward > 0:
                reward = 1.0
            else:
                reward = 0.0

            totalReward += reward

            if terminated or truncated or lostLife:
                done = True
                break

        processedNextState = self.preprocessFrame(nextState)
        self.frameBuffer.append(processedNextState)

        nextStateTensor = torch.tensor(np.stack(self.frameBuffer), dtype=torch.float32)
        return nextStateTensor, totalReward, done, {"lost_life": lostLife}

    def preprocessFrame(self, frame: np.ndarray) -> np.ndarray:
        """Frame preprocessing"""
        if len(frame.shape) == 3:
            frame = np.mean(frame, axis=2)

        frame = frame[34:194, :]

        img = Image.fromarray(frame.astype(np.uint8))
        img = img.resize((self.screenSize, self.screenSize), Image.BILINEAR)
        frame = np.array(img).astype(np.float32) / 255.0

        return frame

    def getCurrentFrame(self) -> np.ndarray:
        """Get current raw frame for visualization"""
        return self.currentRawFrame

    @property
    def actionSpace(self):
        return self.env.action_space

    def close(self):
        self.env.close()

# =============================== Real-time Visualizer ===============================
class RealTimeVisualizer:
    """Real-time visualization for A3C inference"""

    def __init__(self):
        self.fig = None
        self.axes = None
        self.setupPlot()

    def setupPlot(self):
        """Setup matplotlib plot for real-time visualization"""
        plt.ion()  # Turn on interactive mode
        self.fig, self.axes = plt.subplots(1, 2, figsize=(15, 6))

        # Configure fonts for English display[citation:5][citation:10]
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['font.size'] = 10

        # Main plot for game visualization
        self.axes[0].set_title('Breakout - A3C Agent', fontsize=16, fontweight='bold')
        self.axes[0].set_xlabel('Screen Width')
        self.axes[0].set_ylabel('Screen Height')

        # Statistics plot
        self.axes[1].set_title('Agent Statistics', fontsize=16, fontweight='bold')

        self.fig.tight_layout()
        plt.show(block=False)

    def updateDisplay(self, rawFrame: np.ndarray, stats: Dict[str, Any]):
        """Update the visualization with new frame and statistics"""
        try:
            # Clear previous plots
            for ax in self.axes:
                ax.clear()

            # Plot 1: Game visualization
            self.axes[0].imshow(rawFrame)
            self.axes[0].set_title('Breakout - A3C Agent Play', fontsize=14, fontweight='bold')
            self.axes[0].set_xlabel('Screen Width')
            self.axes[0].set_ylabel('Screen Height')
            self.axes[0].grid(False)

            # Plot 2: Statistics
            self.axes[1].set_title('Agent Performance Statistics', fontsize=14, fontweight='bold')
            self.axes[1].set_xlim(0, 10)
            self.axes[1].set_ylim(0, 10)
            self.axes[1].axis('off')

            # Display statistics as text
            statY = 9.0
            lineHeight = 0.7

            statsText = [
                f"Episode: {stats.get('episode', 1)}",
                f"Total Reward: {stats.get('totalReward', 0):.1f}",
                f"Current Step: {stats.get('currentStep', 0)}",
                f"Current Action: {stats.get('currentAction', 'N/A')}",
                f"State Value: {stats.get('stateValue', 0):.3f}",
                f"Lives Remaining: {stats.get('lives', 3)}",
                f"Game Status: {stats.get('gameStatus', 'Playing')}"
            ]

            for i, text in enumerate(statsText):
                self.axes[1].text(0.5, statY - i * lineHeight, text,
                                 fontsize=12, fontweight='bold',
                                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7),
                                 ha='center', va='center')

            # Refresh display
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(0.01)  # Small pause to update display

        except Exception as e:
            print(f"Visualization update error: {e}")

# =============================== A3C Inference Agent ===============================
class A3CInferenceAgent:
    """A3C Agent for real-time inference and visualization"""

    def __init__(self, modelPath: str, screenSize: int = 84):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Inference Device: {self.device}")

        self.screenSize = screenSize
        self.stateShape = (2, screenSize, screenSize)
        self.numActions = 4  # Breakout actions

        # Initialize network
        self.network = A3CNetwork(self.stateShape, self.numActions).to(self.device)

        # Load trained model with security fix[citation:1][citation:2][citation:6]
        self.loadModel(modelPath)

        # Initialize environment and visualizer
        self.envWrapper = InferenceEnvironmentWrapper(
            "BreakoutNoFrameskip-v4",
            frameSkip=4,
            screenSize=screenSize
        )

        self.visualizer = RealTimeVisualizer()

        # Inference statistics
        self.inferenceStats = {
            'episode': 0,
            'totalReward': 0,
            'currentStep': 0,
            'currentAction': 'N/A',
            'stateValue': 0,
            'lives': 3,
            'gameStatus': 'Playing'
        }

    def loadModel(self, modelPath: str):
        """Load trained model weights with security fix"""
        if os.path.exists(modelPath):
            try:
                # 修复PyTorch安全加载问题[citation:1][citation:2]
                import torch.serialization

                # 方法1: 首先尝试使用weights_only=False (仅在信任模型来源时使用)[citation:6]
                try:
                    checkpoint = torch.load(modelPath, map_location=self.device, weights_only=False)
                except Exception as e:
                    print(f"Standard loading failed: {e}")
                    # 方法2: 尝试使用安全上下文管理器[citation:1]
                    try:
                        # 添加安全全局变量以允许numpy对象
                        import numpy as np
                        torch.serialization.add_safe_globals([np.core.multiarray.scalar])
                        checkpoint = torch.load(modelPath, map_location=self.device, weights_only=True)
                    except Exception as e2:
                        print(f"Safe loading failed: {e2}")
                        raise

                if 'modelStateDict' in checkpoint:
                    modelState = checkpoint['modelStateDict']
                else:
                    modelState = checkpoint

                # Handle potential key mismatches
                currentModelState = self.network.state_dict()
                filteredState = {k: v for k, v in modelState.items()
                               if k in currentModelState and currentModelState[k].shape == v.shape}

                currentModelState.update(filteredState)
                self.network.load_state_dict(currentModelState)

                print(f"Successfully loaded model from {modelPath}")
                print(f"Loaded {len(filteredState)}/{len(modelState)} parameters")

            except Exception as e:
                print(f"Error loading model: {e}")
                raise
        else:
            raise FileNotFoundError(f"Model file not found: {modelPath}")

    def runInference(self, numEpisodes: int = 3, maxSteps: int = 1000):
        """Run inference with real-time visualization"""
        print(f"Starting A3C Inference for {numEpisodes} episodes...")

        for episode in range(numEpisodes):
            print(f"\n=== Episode {episode + 1} ===")

            state = self.envWrapper.reset().to(self.device)
            episodeReward = 0
            episodeSteps = 0
            done = False

            self.inferenceStats['episode'] = episode + 1
            self.inferenceStats['totalReward'] = 0
            self.inferenceStats['lives'] = self.envWrapper.lives

            while not done and episodeSteps < maxSteps:
                # Get action from network
                action, _, stateValue = self.network.getAction(state.unsqueeze(0))
                actionNum = action.item()

                # Map action number to meaningful description
                actionDescriptions = {
                    0: "NOOP",
                    1: "FIRE",
                    2: "RIGHT",
                    3: "LEFT"
                }
                actionDesc = actionDescriptions.get(actionNum, "UNKNOWN")

                # Take step in environment
                nextState, reward, done, info = self.envWrapper.step(actionNum)
                nextState = nextState.to(self.device)

                # Update statistics
                episodeReward += reward
                episodeSteps += 1

                self.inferenceStats.update({
                    'totalReward': episodeReward,
                    'currentStep': episodeSteps,
                    'currentAction': actionDesc,
                    'stateValue': stateValue.item(),
                    'lives': self.envWrapper.lives,
                    'gameStatus': 'Game Over' if done else 'Playing'
                })

                # Update visualization
                rawFrame = self.envWrapper.getCurrentFrame()
                self.visualizer.updateDisplay(rawFrame, self.inferenceStats)

                # Update state
                state = nextState

                # Small delay for better visualization
                time.sleep(0.05)

                # Print progress
                if episodeSteps % 50 == 0:
                    print(f"Step {episodeSteps}, Reward: {episodeReward:.1f}, Action: {actionDesc}")

            print(f"Episode {episode + 1} completed!")
            print(f"Total Reward: {episodeReward:.1f}, Steps: {episodeSteps}")

            # Wait a bit between episodes
            if episode < numEpisodes - 1:
                print("Starting next episode in 3 seconds...")
                time.sleep(3)

    def close(self):
        """Cleanup resources"""
        self.envWrapper.close()
        plt.close('all')

# =============================== Main Inference Function ===============================
def main():
    """Main function for A3C real-time inference"""
    try:
        # Configuration
        modelPath = "./A3CModels/best_model.pth"  # Update this path to your trained model

        # Check if model exists
        if not os.path.exists(modelPath):
            print(f"Error: Model file not found at {modelPath}")
            print("Please make sure you have a trained model at the specified path.")
            print("You may need to train the model first using your training script.")
            return

        # Create inference agent
        inferenceAgent = A3CInferenceAgent(modelPath)

        print("A3C Real-time Inference System")
        print("=" * 50)
        print("Controls:")
        print("- The left panel shows the game visualization")
        print("- The right panel shows agent statistics")
        print("- The inference will run for specified number of episodes")
        print("=" * 50)

        # Run inference
        inferenceAgent.runInference(numEpisodes=3, maxSteps=1000)

        # Cleanup
        inferenceAgent.close()

        print("\nInference completed successfully!")

    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main()
