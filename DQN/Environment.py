"""
    环境处理器的实现
"""

import Config
import gymnasium as gym
from datetime import datetime
import os
import json
import numpy as np
from PIL import Image
import torch
from collections import deque
from typing import Tuple



# 可视化数据集要用的环境处理器
class EnvironmentRecorder:
    """环境记录器，用于保存原始环境数据为PNG图片"""
    def __init__(self, config: Config):
        self.config = config
        self.environment = gym.make(config.environmentName, render_mode='rgb_array')
        self.setupSaveDirectory()
        self.episodeMetadata = []

    def setupSaveDirectory(self):
        """创建保存目录和子目录"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.saveDir = f"{self.config.imageSaveDir}_{timestamp}"
        os.makedirs(self.saveDir, exist_ok=True)

        # 创建episodes子目录
        self.episodesDir = os.path.join(self.saveDir, "episodes")
        os.makedirs(self.episodesDir, exist_ok=True)

    def preprocessFrame(self, frame):
        """预处理帧 - 这里保存原始帧，不进行灰度化等处理"""
        return frame

    def saveFrameWithInfo(self, frame, episode, step, action, reward, terminated, truncated, info):
        """保存帧图像和相关信息"""
        try:
            # 预处理帧
            processedFrame = self.preprocessFrame(frame)

            # 确保数据格式正确
            if processedFrame.dtype != np.uint8:
                processedFrame = np.clip(processedFrame * 255, 0, 255).astype(np.uint8)

            # 创建PIL图像
            img = Image.fromarray(processedFrame)

            # 构建文件名和路径
            episodeDir = os.path.join(self.episodesDir, f"episode_{episode:04d}")
            os.makedirs(episodeDir, exist_ok=True)

            filename = f"step_{step:04d}_action_{action}_reward_{reward:.1f}.png"
            filepath = os.path.join(episodeDir, filename)

            # 保存图像
            img.save(filepath)

            # 保存帧信息到元数据
            ######################################
            ######################################
            ######################################
            #     关于帧数据的保存格式与字段的解释      #
            # 关于动作系统    action
            #   0:NOOP  -   无操作
            #   1:FIRE  -   发球
            #   2:RIGHT -   向右移动
            #   3:LEFT  -   向左移动
            #   4:RIGHTFIRE -   向右移动并发球
            #   5:LEFTFIRE  -   向左移动并发球
            ######################################
            ######################################
            ######################################
            frameInfo = {
                'episode': episode,             # 回合编号
                'step': step,                   # 当前步数(从0开始)
                'action': int(action),          # 执行的动作编号
                'reward': float(reward),        # 这一步获得的奖励
                'terminated': bool(terminated), # 是否终止（游戏结束）
                'truncated': bool(truncated),   # 是否被截断（时间限制等）
                'filename': filename,           # 保存的截图文件名
                'timestamp': datetime.now().isoformat()     # 时间戳
            }

            # 加入info中剩余的信息
            if info:
                frameInfo.update(info)  # 添加环境返回的info

            return frameInfo
        except Exception as e:
            return None

    def recordEpisode(self, episodeNum, maxSteps=1000):
        """记录一个完整的episode"""
        try:
            state, info = self.environment.reset()
            episodeFrames = []
            totalReward = 0

            episodeDir = os.path.join(self.episodesDir, f"episode_{episodeNum:04d}")
            os.makedirs(episodeDir, exist_ok=True)

            for step in range(maxSteps):
                # 随机动作
                action = self.environment.action_space.sample()

                nextState, reward, terminated, truncated, info = self.environment.step(action)

                # 保存当前帧和相关信息
                frameInfo = self.saveFrameWithInfo(
                    state, episodeNum, step, action, reward, terminated, truncated, info
                )
                if frameInfo:
                    episodeFrames.append(frameInfo)

                # 更新状态
                state = nextState
                totalReward += reward

                # 检查是否结束
                if terminated or truncated:
                    break

            # 保存episode的总体元数据
            episodeMetadata = {
                'episodeNumber': episodeNum,
                'totalReward': totalReward,
                'totalSteps': maxSteps,
                'frames': episodeFrames,
                'environment': self.config.environmentName,
                'timestamp': datetime.now().isoformat()
            }

            metadataFile = os.path.join(episodeDir, "metadata.json")
            with open(metadataFile, 'w') as f:
                json.dump(episodeMetadata, f, indent=2)

            self.episodeMetadata.append(episodeMetadata)

            return totalReward
        except Exception as e:
            return 0.0

    def saveSummary(self):
        """保存所有episode的摘要信息"""
        try:
            summary = {
                'totalEpisodes': len(self.episodeMetadata),
                'environment': self.config.environmentName,
                'config': self.config.__dict__,
                'episodes': self.episodeMetadata,
                'recordedAt': datetime.now().isoformat()
            }

            summaryFile = os.path.join(self.saveDir, "recordingSummary.json")
            with open(summaryFile, 'w') as f:
                json.dump(summary, f, indent=2)

        except Exception as e:
            return

    def close(self):
        """关闭环境并保存摘要"""
        self.saveSummary()
        self.environment.close()




class AtariEnvironmentPreprocessor:
    """Atari环境预处理包装器"""

    def __init__(self, environment, frameSkip: int = 4, screenSize: int = 84):
        self.environment = environment
        self.frameSkip = frameSkip
        self.screenSize = screenSize
        self.frameBuffer = deque(maxlen=4)

    def reset(self) -> Tuple[torch.Tensor, dict]:
        """重置环境并返回预处理后的初始状态"""
        try:
            state, info = self.environment.reset()
            processedState = self._preprocessFrame(state)

            # 用相同的帧填充初始缓冲区
            self.frameBuffer.extend([processedState] * 4)

            stateTensor = torch.tensor(np.stack(self.frameBuffer), dtype=torch.float32)
            return stateTensor, info
        except Exception as e:
            raise

    def step(self, action: int) -> Tuple[torch.Tensor, float, bool, dict]:
        """执行动作并返回预处理后的结果"""
        try:
            totalReward = 0.0
            terminated = False
            truncated = False
            info = {}

            # 使用帧跳过提高效率
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
            raise

    def _preprocessFrame(self, frame: np.ndarray) -> np.ndarray:
        """预处理帧：灰度化、调整大小、归一化"""
        try:
            # 转换为灰度图
            if len(frame.shape) == 3:
                frame = np.mean(frame, axis=2)  # 使用numpy提高效率

            # 调整大小
            img = Image.fromarray(frame.astype(np.uint8))
            img = img.resize((self.screenSize, self.screenSize), Image.BILINEAR)
            frame = np.array(img)

            # 归一化到 [0, 1]
            frame = frame.astype(np.float32) / 255.0

            return frame
        except Exception as e:
            raise

    @property
    def actionSpace(self):
        return self.environment.action_space

    @property
    def observationSpace(self):
        return self.environment.observation_space

    def close(self) -> None:
        """关闭环境"""
        self.environment.close()






