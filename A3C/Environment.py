"""
Environment wrappers and preprocessors
"""

import gymnasium as gym
import numpy as np
import torch
from PIL import Image
from collections import deque
from typing import Tuple, Dict, Any
import logging


class EnvironmentWrapper:
    """Environment Wrapper for A3C"""

    def __init__(self, envName: str, frameSkip: int = 4, screenSize: int = 84):
        self.env = gym.make(envName, render_mode='rgb_array')
        self.frameSkip = frameSkip
        self.screenSize = screenSize
        self.frameBuffer = deque(maxlen=2)
        self.lives = 0
        self.originalLives = 5

    def reset(self) -> torch.Tensor:
        state, _ = self.env.reset()
        processedState = self.preprocessFrame(state)

        # Initialize frame buffer
        self.frameBuffer.clear()
        for _ in range(2):
            self.frameBuffer.append(processedState)

        # Get initial lives
        self.lives = self.env.unwrapped.ale.lives()
        self.originalLives = self.lives

        return torch.tensor(np.stack(self.frameBuffer), dtype=torch.float32)

    def step(self, action: int) -> Tuple[torch.Tensor, float, bool, Dict]:
        totalReward = 0.0
        done = False
        lostLife = False

        for _ in range(self.frameSkip):
            nextState, reward, terminated, truncated, stepInfo = self.env.step(action)

            # Check life loss
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
        return nextStateTensor, totalReward, done, {"lostLife": lostLife}

    def preprocessFrame(self, frame: np.ndarray) -> np.ndarray:
        """Improved frame preprocessing"""
        if len(frame.shape) == 3:
            # Convert to grayscale and crop
            frame = np.mean(frame, axis=2)

        # Crop Breakout irrelevant areas
        frame = frame[34:194, :]

        img = Image.fromarray(frame.astype(np.uint8))
        img = img.resize((self.screenSize, self.screenSize), Image.BILINEAR)
        frame = np.array(img).astype(np.float32) / 255.0

        return frame

    @property
    def actionSpace(self):
        return self.env.action_space

    def close(self):
        self.env.close()


class AtariEnvironmentPreprocessor:
    """Atari environment preprocessor wrapper for DQN"""

    def __init__(self, environment, frameSkip: int = 4, screenSize: int = 84):
        self.environment = environment
        self.frameSkip = frameSkip
        self.screenSize = screenSize
        self.frameBuffer = deque(maxlen=4)

    def reset(self) -> Tuple[torch.Tensor, dict]:
        """Reset environment and return preprocessed initial state"""
        try:
            state, info = self.environment.reset()
            processedState = self._preprocessFrame(state)

            # Fill initial buffer with same frames
            self.frameBuffer.extend([processedState] * 4)

            stateTensor = torch.tensor(np.stack(self.frameBuffer), dtype=torch.float32)
            return stateTensor, info
        except Exception as e:
            logging.error(f"Failed to reset environment: {e}")
            raise

    def step(self, action: int) -> Tuple[torch.Tensor, float, bool, dict]:
        """Execute action and return preprocessed results"""
        try:
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
        except Exception as e:
            logging.error(f"Failed to execute action: {e}")
            raise

    def _preprocessFrame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame: grayscale, resize, normalize"""
        try:
            # Convert to grayscale
            if len(frame.shape) == 3:
                frame = np.mean(frame, axis=2)  # Use numpy for efficiency

            # Resize
            img = Image.fromarray(frame.astype(np.uint8))
            img = img.resize((self.screenSize, self.screenSize), Image.BILINEAR)
            frame = np.array(img)

            # Normalize to [0, 1]
            frame = frame.astype(np.float32) / 255.0

            return frame
        except Exception as e:
            logging.error(f"Failed to preprocess frame: {e}")
            raise

    @property
    def actionSpace(self):
        return self.environment.action_space

    @property
    def observationSpace(self):
        return self.environment.observation_space

    def close(self) -> None:
        """Close environment"""
        self.environment.close()

