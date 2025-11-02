"""
环境处理相关类
"""
import gymnasium as gym
import numpy as np
import torch
from PIL import Image
from collections import deque
from typing import Tuple, Dict, Any
import json
import os
from datetime import datetime
import logging
from Config import PPOConfig, LunarLanderConfig
import cv2

logger = logging.getLogger(__name__)


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
            logger.error(f"重置环境失败: {e}")
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
            logger.error(f"执行动作失败: {e}")
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
            logger.error(f"预处理帧失败: {e}")
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


class EnvironmentRecorder:
    """环境记录器，用于保存原始环境数据为PNG图片"""

    def __init__(self, config: PPOConfig):
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

        logger.info(f"环境数据将保存到: {self.saveDir}")

    def preprocessFrame(self, frame):
        """预处理帧 - 这里保存原始帧，不进行灰度化等处理"""
        return frame

    def saveFrameWithInfo(self, frame, episode, step, action, reward,
                           terminated, truncated, info):
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
            frameInfo = {
                'episode': episode,
                'step': step,
                'action': int(action),
                'reward': float(reward),
                'terminated': bool(terminated),
                'truncated': bool(truncated),
                'filename': filename,
                'timestamp': datetime.now().isoformat()
            }
            if info:
                frameInfo.update(info)  # 添加环境返回的info

            return frameInfo
        except Exception as e:
            logger.error(f"保存帧信息失败: {e}")
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

            # 保存episode的元数据
            episodeMetadata = {
                'episodeNumber': episodeNum,
                'totalReward': totalReward,
                'totalSteps': step + 1,
                'frames': episodeFrames,
                'environment': self.config.environmentName,
                'timestamp': datetime.now().isoformat()
            }

            metadataFile = os.path.join(episodeDir, "metadata.json")
            with open(metadataFile, 'w') as f:
                json.dump(episodeMetadata, f, indent=2)

            self.episodeMetadata.append(episodeMetadata)
            logger.info(f"Episode {episodeNum}: {step + 1} 步, 总奖励: {totalReward}")

            return totalReward
        except Exception as e:
            logger.error(f"记录episode失败: {e}")
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

            logger.info(f"摘要已保存到: {summaryFile}")
        except Exception as e:
            logger.error(f"保存摘要失败: {e}")

    def close(self):
        """关闭环境并保存摘要"""
        self.saveSummary()
        self.environment.close()



# 月球着陆游戏的环境包装器
class LunarLanderEnvironment(object):
    def __init__(self, config: LunarLanderConfig):
        self.frameSize = config.screenSize
        self.environmentName = config.environmentName
        self.environment = gym.make(self.environmentName, render_mode='rgb_array')
        self.saveEnvironmentVideosPath = config.saveVideosPath
        self.saveVideosSteps = config.saveVideosSteps
        self.saveVideosEpisode = config.saveVideosEpisode

    # 保存环境数据视频
    def save_episode_videos(self):
        for i in range(self.saveVideosEpisode):
            state, info = self.environment.reset()
            episodeFrames = []
            episodeData = []  # 用于保存每个episode的数据
            os.makedirs(self.saveEnvironmentVideosPath, exist_ok=True)

            for step in range(self.saveVideosSteps):
                # 随机动作
                action = self.environment.action_space.sample()

                # 获取当前步返回的信息
                nextState, reward, terminated, truncated, info = self.environment.step(action)

                # 创建当前步的数据字典
                stepData = {
                    'step': step,
                    'timestamp': datetime.now().isoformat(),
                    'position': {
                        'x': float(nextState[0]),
                        'y': float(nextState[1])
                    },
                    'velocity': {
                        'x': float(nextState[2]),
                        'y': float(nextState[3])
                    },
                    'angle': float(nextState[4]),
                    'angular_velocity': float(nextState[5]),
                    'left_leg_contact': bool(nextState[6]),
                    'right_leg_contact': bool(nextState[7]),
                    'action': int(action) if hasattr(action, '__int__') else action.tolist(),
                    'reward': float(reward),
                    'terminated': bool(terminated),
                    'truncated': bool(truncated),
                    'info': info
                }

                # 添加到episode数据中
                episodeData.append(stepData)

                # 获取当前步的图片帧
                frame = self.environment.render()
                if frame is not None:
                    episodeFrames.append(frame)

            # 创建视频流，保存为.mp4的视频
            frameHeight, frameWidth = episodeFrames[0].shape[:2]
            videoFilename = os.path.join(self.saveEnvironmentVideosPath, f"lunar_lander_episode_{i}_video.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            videoWriter = cv2.VideoWriter(videoFilename, fourcc, 30.0, (frameWidth, frameHeight))
            for j in range(len(episodeFrames)):
                videoWriter.write(cv2.cvtColor(episodeFrames[j], cv2.COLOR_RGB2BGR))
            print(f'环境示例视频: {videoFilename} 保存完毕')

            # 保存JSON数据
            json_filename = os.path.join(self.saveEnvironmentVideosPath, f"lunar_lander_episode_{i}_data.json")
            with open(json_filename, 'w', encoding='utf-8') as f:
                json.dump({
                    'episode': i,
                    'total_steps': len(episodeData),
                    'timestamp': datetime.now().isoformat(),
                    'steps': episodeData
                }, f, indent=2, ensure_ascii=False)

            print(f'环境数据JSON: {json_filename} 保存完毕')


    # 获取环境的动作空间
    @property
    def actionSpace(self):
        return self.environment.action_space

    # 获取环境的状态空间
    @property
    def observationSpace(self):
        return self.environment.observation_space

    # 重置环境
    def reset(self):
        state, info = self.environment.reset()
        return state, info

    # 获取当前步的图像帧(rgb格式)
    def get_rgb_frame(self):
        frame = self.environment.render()
        return frame

    # 获取当前步的图像帧(单通道的灰度图格式) 本方法中还有对于灰度图的预处理步骤
    def get_gray_frame(self):
        frame = self.environment.render()
        # 转换为灰度图
        if len(frame.shape) == 3:
            frame = np.mean(frame, axis=2)  # 使用numpy提高效率

        # 调整大小
        img = Image.fromarray(frame.astype(np.uint8))
        img = img.resize((self.frameSize, self.frameSize), Image.BILINEAR)
        frame = np.array(img)

        # 归一化到 [0, 1]
        frame = frame.astype(np.float32) / 255.0

        return frame


    # 执行下一步的交互动作
    def step(self, action: int):
        nextState, reward, terminated, truncated, info = self.environment.step(action)
        done = terminated or truncated
        return nextState, reward, done, info





