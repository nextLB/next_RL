"""
环境预处理模块
"""

import gymnasium as gym
import numpy as np
from PIL import Image
from collections import deque
from typing import Tuple
from Config import A3CConfig

class AtariEnvironmentPreprocessor:
    """Atari环境预处理包装器"""

    def __init__(self, environment, frameSkip: int = 4, screenSize: int = 84):
        self.environment = environment
        self.frameSkip = frameSkip
        self.screenSize = screenSize
        self.frameBuffer = deque(maxlen=4)

    def reset(self) -> Tuple[np.ndarray, dict]:
        """重置环境并返回预处理后的初始状态"""
        state, info = self.environment.reset()
        processedState = self._preprocessFrame(state)

        # 用相同的帧填充初始缓冲区
        self.frameBuffer.extend([processedState] * 4)

        stateArray = np.stack(self.frameBuffer)
        return stateArray, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        totalReward = 0.0
        terminated = False
        truncated = False

        # 执行frameSkip次动作
        for _ in range(self.frameSkip):
            nextState, reward, terminated, truncated, stepInfo = self.environment.step(action)
            totalReward += reward
            if terminated or truncated:
                break

        # 重要：对奖励进行裁剪
        totalReward = np.clip(totalReward, -1, 1)

        done = terminated or truncated
        processedNextState = self._preprocessFrame(nextState)
        self.frameBuffer.append(processedNextState)

        nextStateArray = np.stack(self.frameBuffer)
        return nextStateArray, totalReward, done, stepInfo

    def _preprocessFrame(self, frame: np.ndarray) -> np.ndarray:
        """预处理帧：灰度化、调整大小、归一化"""
        # 转换为灰度图
        if len(frame.shape) == 3:
            frame = np.mean(frame, axis=2)

        # 调整大小
        img = Image.fromarray(frame.astype(np.uint8))
        img = img.resize((self.screenSize, self.screenSize), Image.BILINEAR)
        frame = np.array(img)

        # 归一化到 [0, 1]
        frame = frame.astype(np.float32) / 255.0

        return frame

    def close(self) -> None:
        """关闭环境"""
        self.environment.close()

def createEnvironment(config: A3CConfig):
    """创建环境函数"""
    env = gym.make(config.environmentName)
    preprocessedEnv = AtariEnvironmentPreprocessor(
        env,
        frameSkip=config.frameSkip,
        screenSize=config.screenSize
    )
    return preprocessedEnv

def getNumActions(config: A3CConfig) -> int:
    """获取动作空间大小"""
    env = gym.make(config.environmentName)
    numActions = env.action_space.n
    env.close()
    return numActions

