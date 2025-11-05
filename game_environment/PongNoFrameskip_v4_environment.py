"""
    关于PongNoFrameskip-v4这个游戏环境的搭建程序文件
"""

# 请注意！！！！！！！！！！！！！！！！！！！！
# 在Pong游戏训练的过程中，每赢一球得+1分，输一球得-1分  所以最初的训练奖励值是在-20左右徘徊




import numpy as np
from PIL import Image
import gymnasium as gym


class PNFSV4Environment:
    def __init__(self, config):
        self.environmentName = "PongNoFrameskip-v4"
        self.config = config
        self.env = gym.make(self.config.environmentName, render_mode='rgb_array')

    # 对于Pong游戏的图像帧进行预处理
    def preprocess_frame_to_gray(self, frame):
        # 转换为灰度图
        if len(frame.shape) == 3:
            frame = np.mean(frame, axis=2)  # 使用numpy提高效率

        # 调整大小
        img = Image.fromarray(frame.astype(np.uint8))
        img = img.resize((self.config.imageShape[1], self.config.imageShape[2]), Image.BILINEAR)
        frame = np.array(img)

        # 归一化到 [0, 1]
        frame = frame.astype(np.float32) / 255.0

        return frame

    # 重置环境并返回预处理后的初始状态帧
    def reset(self):
        state, info = self.env.reset()
        processedState = self.preprocess_frame_to_gray(state)
        return processedState, info

    # 执行动作并返回预处理后的结果
    def step(self, action):
        nextState, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        processedNextState = self.preprocess_frame_to_gray(nextState)
        return processedNextState, reward, done, info

    @property
    def actionSpace(self):
        return self.env.action_space

    @property
    def observationSpace(self):
        return self.env.observation_space

    def close(self) -> None:
        """关闭环境"""
        self.env.close()






