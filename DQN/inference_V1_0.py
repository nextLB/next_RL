"""
Pong Game Inference Program - Load Trained Model with Real-Time Visualization
Enhanced Version with Improved Readability, Robustness, and Extensibility
"""

import torch
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from PIL import Image
import time
import os
from collections import deque
import logging
from dataclasses import dataclass, asdict
import json
from typing import Dict, List, Tuple, Optional, Any
import warnings


@dataclass
class TrainingConfig:
    """Training Configuration Parameters - Same as Training Program"""
    environmentName: str = "PongNoFrameskip-v4"
    learningRate: float = 0.00025
    discountFactor: float = 0.99
    batchSize: int = 32
    replayBufferCapacity: int = 50000
    targetUpdateFrequency: int = 1000
    learningStartSteps: int = 1000
    learningUpdateFrequency: int = 4
    initialEpsilon: float = 1.0
    finalEpsilon: float = 0.1
    epsilonDecaySteps: int = 50000
    frameSkip: int = 4
    screenSize: int = 84
    useSimpleResNet: bool = True
    trainingEpisodes: int = 1000
    targetAverageReward: float = 15.0
    saveImages: bool = True
    imageSaveDir: str = "recordedPongEpisodes"
    numEpisodesToRecord: int = 10


class AtariEnvironmentPreprocessor:
    """
    Atari Environment Preprocessor - Consistent with Training
    Handles frame preprocessing and environment interaction
    """

    def __init__(self, environment: gym.Env, frameSkip: int = 4, screenSize: int = 84) -> None:
        self.environment = environment
        self.frameSkip = frameSkip
        self.screenSize = screenSize
        self.frameBuffer = deque(maxlen=4)

    def reset(self) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Reset environment and return preprocessed initial state"""
        state, info = self.environment.reset()
        processedState = self._preprocessFrame(state)

        # Fill initial buffer with same frame
        self.frameBuffer.extend([processedState] * 4)

        stateTensor = torch.tensor(np.stack(self.frameBuffer), dtype=torch.float32)
        return stateTensor, info

    def step(self, action: int) -> Tuple[torch.Tensor, float, bool, Dict[str, Any]]:
        """Execute action and return preprocessed results"""
        totalReward = 0.0
        terminated = False
        truncated = False
        info = {}

        # Use frame skipping for efficiency
        for _ in range(self.frameSkip):
            nextState, reward, terminated, truncated, stepInfo = self.environment.step(action)
            totalReward += reward
            info.update(stepInfo)

            if terminated or truncated:
                break

        done = terminated or truncated
        processedNextState = self._preprocessFrame(nextState)
        self.frameBuffer.append(processedNextState)

        nextStateTensor = torch.tensor(np.stack(self.frameBuffer), dtype=torch.float32)
        return nextStateTensor, totalReward, done, info

    def _preprocessFrame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame: grayscale, resize, normalize"""
        try:
            # Convert to grayscale
            if len(frame.shape) == 3:
                frame = np.mean(frame, axis=2)

            # Resize
            img = Image.fromarray(frame.astype(np.uint8))
            img = img.resize((self.screenSize, self.screenSize), Image.BILINEAR)
            frame = np.array(img)

            # Normalize to [0, 1]
            frame = frame.astype(np.float32) / 255.0

            return frame
        except Exception as e:
            logging.error(f"Frame preprocessing failed: {e}")
            raise

    @property
    def actionSpace(self) -> gym.Space:
        """Get action space from environment"""
        return self.environment.action_space

    def close(self) -> None:
        """Close environment"""
        self.environment.close()


class ResidualBlock(torch.nn.Module):
    """Residual Block - Consistent with Training"""

    def __init__(self, inChannels: int, outChannels: int, stride: int = 1) -> None:
        super().__init__()

        self.conv1 = torch.nn.Conv2d(
            inChannels, outChannels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = torch.nn.BatchNorm2d(outChannels)
        self.conv2 = torch.nn.Conv2d(
            outChannels, outChannels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = torch.nn.BatchNorm2d(outChannels)

        # Shortcut connection - ensure dimension matching
        self.shortcut = torch.nn.Sequential()
        if stride != 1 or inChannels != outChannels:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(inChannels, outChannels, kernel_size=1, stride=stride, bias=False),
                torch.nn.BatchNorm2d(outChannels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection"""
        residual = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # Ensure residual connection dimension matching
        residual = self.shortcut(residual)
        out += residual
        out = F.relu(out)

        return out


class ResNetDeepQNetwork(torch.nn.Module):
    """ResNet DQN Network - Consistent with Training"""

    def __init__(self, inputShape: Tuple[int, int, int], numActions: int) -> None:
        super().__init__()

        self.inChannels = 64

        # Initial convolution layer - adapted for 84x84 input
        self.conv1 = torch.nn.Conv2d(
            inputShape[0], 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = torch.nn.BatchNorm2d(64)

        # ResNet layers - adapted for 84x84 input
        self.layer1 = self._makeLayer(64, 64, 2, stride=1)  # 84x84 -> 84x84
        self.layer2 = self._makeLayer(64, 128, 2, stride=2)  # 84x84 -> 42x42
        self.layer3 = self._makeLayer(128, 256, 2, stride=2)  # 42x42 -> 21x21
        self.layer4 = self._makeLayer(256, 512, 2, stride=2)  # 21x21 -> 11x11

        # Adaptive average pooling to fixed size
        self.adaptiveAvgPool = torch.nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layer
        self.fc = torch.nn.Linear(512, numActions)

    def _makeLayer(self, inChannels: int, outChannels: int, numBlocks: int, stride: int) -> torch.nn.Sequential:
        """Create ResNet layer with specified parameters"""
        strides = [stride] + [1] * (numBlocks - 1)
        layers = []

        for currentStride in strides:
            layers.append(ResidualBlock(inChannels, outChannels, currentStride))
            inChannels = outChannels

        return torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network"""
        # Initial convolution
        x = F.relu(self.bn1(self.conv1(x)))

        # ResNet layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Global pooling and fully connected
        x = self.adaptiveAvgPool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class PongGameInference:
    """
    Pong Game Inference Class - Load Trained Model with Configuration
    Enhanced with better error handling and extensibility
    """

    def __init__(
        self,
        modelPath: str,
        environmentName: str = "PongNoFrameskip-v4",
        renderMode: str = "rgb_array"
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._printDeviceInfo()

        self.modelPath = modelPath
        self.environmentName = environmentName
        self.renderMode = renderMode

        # Load configuration from trained model
        self.trainingConfig = self._loadTrainingConfig()

        # Initialize environment and preprocessor with loaded config
        self._initializeEnvironment()

        # Load model
        self._loadTrainedModel()

        # Initialize visualization
        self._initializeVisualization()

        # Action mapping (Pong game)
        self.actionMap = {
            0: "NOOP",
            1: "FIRE",
            2: "RIGHT",
            3: "LEFT",
            4: "RIGHTFIRE",
            5: "LEFTFIRE"
        }

    def _printDeviceInfo(self) -> None:
        """Print device information for debugging"""
        deviceInfo = f"Using Device: {self.device}"
        if self.device.type == 'cuda':
            deviceInfo += f" ({torch.cuda.get_device_name()})"
        print(deviceInfo)

    def _loadTrainingConfig(self) -> TrainingConfig:
        """Load training configuration from model file with PyTorch 2.6+ compatibility"""
        try:
            # For PyTorch 2.6+, handle weights_only parameter
            checkpoint = torch.load(self.modelPath, map_location=self.device, weights_only=False)

            if 'config' in checkpoint:
                configData = checkpoint['config']

                # Handle different config data types
                if isinstance(configData, TrainingConfig):
                    trainingConfig = configData
                    print("Successfully loaded TrainingConfig object from model file")
                elif isinstance(configData, dict):
                    trainingConfig = TrainingConfig(**configData)
                    print("Successfully loaded training configuration from dictionary")
                else:
                    print(f"Unknown config type: {type(configData)}, using default configuration")
                    trainingConfig = TrainingConfig()
            else:
                print("No configuration found in model file, using default configuration")
                trainingConfig = TrainingConfig()

            return trainingConfig

        except Exception as e:
            print(f"Failed to load training configuration: {e}")
            print("Using default configuration")
            return TrainingConfig()

    def _initializeEnvironment(self) -> None:
        """Initialize game environment with loaded configuration"""
        try:
            # Use gymnasium instead of gym
            self.environment = gym.make(self.environmentName, render_mode=self.renderMode)
            self.preprocessedEnvironment = AtariEnvironmentPreprocessor(
                self.environment,
                frameSkip=self.trainingConfig.frameSkip,
                screenSize=self.trainingConfig.screenSize
            )
            print(f"Environment initialized: {self.environmentName}")
            print(f"Using configuration - FrameSkip: {self.trainingConfig.frameSkip}, "
                  f"ScreenSize: {self.trainingConfig.screenSize}")
        except Exception as e:
            print(f"Environment initialization failed: {e}")
            self._tryFallbackEnvironment()

    def _tryFallbackEnvironment(self) -> None:
        """Try to initialize fallback environment"""
        try:
            envName = self.environmentName.replace("NoFrameskip", "")
            self.environment = gym.make(envName, render_mode=self.renderMode)
            self.preprocessedEnvironment = AtariEnvironmentPreprocessor(
                self.environment,
                frameSkip=self.trainingConfig.frameSkip,
                screenSize=self.trainingConfig.screenSize
            )
            print(f"Using fallback environment: {envName}")
        except Exception as e:
            print(f"Fallback environment also failed: {e}")
            raise

    def _loadTrainedModel(self) -> None:
        """Load trained model with PyTorch 2.6+ compatibility"""
        try:
            # Get state shape and action count
            stateShape = (4, self.trainingConfig.screenSize, self.trainingConfig.screenSize)
            numActions = self.preprocessedEnvironment.actionSpace.n

            # Create network architecture
            self.model = ResNetDeepQNetwork(stateShape, numActions).to(self.device)

            # Load trained weights with PyTorch 2.6+ compatibility
            checkpoint = torch.load(self.modelPath, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['policyNetworkState'])
            self.model.eval()  # Set to evaluation mode

            print(f"Model loaded successfully: {self.modelPath}")
            print(f"Action Space: {numActions} actions")
            print(f"State Shape: {stateShape}")

        except Exception as e:
            print(f"Model loading failed: {e}")
            raise

    def _initializeVisualization(self) -> None:
        """Initialize matplotlib visualization with English labels"""
        plt.ion()  # Enable interactive mode
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 5))
        self.fig.suptitle('Pong Game AI Inference Visualization', fontsize=16, fontweight='bold')

        # Initialize image display
        self.rawImage = self.ax1.imshow(np.zeros((210, 160, 3), dtype=np.uint8))
        self.ax1.set_title('Raw Game Screen')
        self.ax1.axis('off')

        # Processed frame display
        processedFrame = np.zeros((self.trainingConfig.screenSize, self.trainingConfig.screenSize))
        self.processedImage = self.ax2.imshow(processedFrame, cmap='gray', vmin=0, vmax=1)
        self.ax2.set_title('Preprocessed Screen (Network Input)')
        self.ax2.axis('off')

        # Add information text box
        self.infoText = self.fig.text(
            0.02, 0.02, '', fontsize=10,
            bbox=dict(facecolor='white', alpha=0.8),
            transform=self.fig.transFigure
        )

        # Add Q-value bar chart
        self.qValueAx = self.fig.add_axes([0.75, 0.15, 0.2, 0.3])
        self.qValueBars = self.qValueAx.bar(range(6), [0]*6, color='skyblue', alpha=0.7)
        self.qValueAx.set_title('Q-Values for Each Action')
        self.qValueAx.set_xlabel('Action')
        self.qValueAx.set_ylabel('Q-Value')
        self.qValueAx.set_xticks(range(6))
        self.qValueAx.set_xticklabels(['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'R-FIRE', 'L-FIRE'], rotation=45)

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.25)

    def selectAction(self, state: torch.Tensor) -> Tuple[int, np.ndarray]:
        """Select action based on current state using the trained model"""
        with torch.no_grad():
            # Ensure correct input dimensions
            if len(state.shape) == 3:
                state = state.unsqueeze(0)  # Add batch dimension

            state = state.to(self.device)
            qValues = self.model(state)
            action = qValues.max(1)[1].item()

            # Get Q-values for all actions for display
            qValuesList = qValues.cpu().numpy()[0]

            return action, qValuesList

    def updateVisualization(
        self,
        rawFrame: np.ndarray,
        processedState: torch.Tensor,
        action: int,
        qValues: np.ndarray,
        reward: float,
        step: int,
        totalReward: float
    ) -> None:
        """Update visualization display with current game state"""
        # Update raw screen
        self.rawImage.set_data(rawFrame)

        # Update preprocessed screen (show latest frame)
        if hasattr(processedState, 'cpu'):
            latestProcessedFrame = processedState[-1].cpu().numpy()
        else:
            latestProcessedFrame = processedState[-1]
        self.processedImage.set_data(latestProcessedFrame)
        self.processedImage.set_clim(0, 1)  # Ensure consistent color scaling

        # Update Q-value bar chart
        if len(qValues) == len(self.qValueBars):
            for bar, value in zip(self.qValueBars, qValues):
                bar.set_height(value)
            if len(qValues) > 0:
                self.qValueAx.set_ylim(min(qValues) - 0.1, max(qValues) + 0.1)

        # Update information text
        actionName = self.actionMap.get(action, f"Unknown({action})")
        infoString = (
            f"Step: {step}\n"
            f"Action: {action} ({actionName})\n"
            f"Current Reward: {reward:.1f}\n"
            f"Total Reward: {totalReward:.1f}\n"
            f"Q-Value: {qValues[action]:.3f} (Max: {max(qValues):.3f})"
        )

        self.infoText.set_text(infoString)

        # Refresh display
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        # Add brief delay for observation
        time.sleep(0.05)

    def runSingleEpisode(self, episodeNum: int, maxSteps: int = 1000) -> Tuple[float, int]:
        """Run a single inference episode and return total reward and steps"""
        print(f"\n=== Episode {episodeNum + 1} ===")

        # Reset environment
        state, _ = self.preprocessedEnvironment.reset()
        totalReward = 0
        step = 0

        # Get initial screen
        rawFrame = self.environment.render()

        while step < maxSteps:
            # Select action
            action, qValues = self.selectAction(state)

            # Execute action
            # 这里就是真实的在环境中执行推理出来的下一步动作
            nextState, reward, done, info = self.preprocessedEnvironment.step(action)
            totalReward += reward

            # Get current raw screen
            rawFrame = self.environment.render()

            # Update visualization
            self.updateVisualization(rawFrame, state, action, qValues, reward, step, totalReward)

            # Update state
            state = nextState
            step += 1

            # Check if episode ended
            if done:
                print(f"Episode {episodeNum + 1} completed! Steps: {step}, Total Reward: {totalReward}")
                break

        else:
            print(f"Episode {episodeNum + 1} reached max steps! Steps: {step}, Total Reward: {totalReward}")

        return totalReward, step

    def runInference(self, numEpisodes: int = 3, maxStepsPerEpisode: int = 1000) -> None:
        """Run inference with visualization for specified number of episodes"""
        print(f"Starting inference, running {numEpisodes} episodes...")
        print(f"Using configuration from training: FrameSkip={self.trainingConfig.frameSkip}, "
              f"ScreenSize={self.trainingConfig.screenSize}")

        episodeRewards = []

        for episode in range(numEpisodes):
            try:
                totalReward, steps = self.runSingleEpisode(episode, maxStepsPerEpisode)
                episodeRewards.append(totalReward)

                # Pause between episodes
                if episode < numEpisodes - 1:
                    print("Preparing next episode...")
                    time.sleep(2)
            except Exception as e:
                logging.error(f"Episode {episode + 1} failed: {e}")
                episodeRewards.append(0.0)

        # Print summary
        self._printInferenceSummary(episodeRewards)

    def _printInferenceSummary(self, episodeRewards: List[float]) -> None:
        """Print inference performance summary"""
        print("\n=== Inference Summary ===")
        print(f"Average Reward: {np.mean(episodeRewards):.2f}")
        print(f"Max Reward: {max(episodeRewards):.2f}")
        print(f"Min Reward: {min(episodeRewards):.2f}")
        print(f"Number of Episodes: {len(episodeRewards)}")

        print("\nInference completed!")

    def close(self) -> None:
        """Close environment and visualization"""
        try:
            self.preprocessedEnvironment.close()
            plt.ioff()
            plt.close()
            print("Resources cleaned up successfully")
        except Exception as e:
            logging.warning(f"Error during cleanup: {e}")


def findModelFile() -> Optional[str]:
    """Find model file in possible locations with error handling"""
    possiblePaths = [
        "./DQN_V1_0_models/gpu_resnet_dqn_best_PongNoFrameskip_v4.pth",
        "./DQN_V1_0_models/gpu_resnet_dqn_final_PongNoFrameskip_v4.pth",
        "./DQN_V1_0_models/gpu_resnet_dqn_best_PongNoFrameskip-v4.pth",
        "./gpu_resnet_dqn_best_PongNoFrameskip_v4.pth",
        "./models/gpu_resnet_dqn_best_PongNoFrameskip_v4.pth"
    ]

    for path in possiblePaths:
        if os.path.exists(path):
            print(f"Found model file: {path}")
            return path

    print("No model file found in expected locations")
    return None


def configureLogging() -> None:
    """Configure logging settings"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('./log/pong_inference.log')
        ]
    )


def main() -> None:
    """Main function with enhanced error handling"""
    configureLogging()

    print("Pong Game Inference Program - Starting...")

    # Find model file
    modelPath = findModelFile()

    if modelPath is None:
        print("No model file found. Please ensure you have a trained model file.")
        print("Looking for files in:")
        print("  - ./DQN_V1_0_models/gpu_resnet_dqn_best_PongNoFrameskip_v4.pth")
        print("  - ./DQN_V1_0_models/gpu_resnet_dqn_final_PongNoFrameskip_v4.pth")
        print("  - ./gpu_resnet_dqn_best_PongNoFrameskip_v4.pth")
        return

    inferenceEngine = None
    try:
        # Create inference engine
        inferenceEngine = PongGameInference(modelPath=modelPath)

        # Run inference
        inferenceEngine.runInference(numEpisodes=3, maxStepsPerEpisode=500)

    except KeyboardInterrupt:
        print("\nInference interrupted by user")
    except Exception as e:
        logging.error(f"Error during inference: {e}")
        print(f"Fatal error occurred: {e}")
    finally:
        # Ensure proper cleanup
        if inferenceEngine is not None:
            inferenceEngine.close()
        print("Program terminated")


if __name__ == "__main__":
    main()