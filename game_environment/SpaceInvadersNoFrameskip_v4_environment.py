
"""
    关于SpaceInvadersNoFrameskip-v4这个游戏环境的搭建程序文件
"""

# 请注意！！！！！！！！！！！！！！！！！！！！
# 在Space Invaders游戏训练的过程中，击落敌人得5-30分不等，所以最初的训练奖励值就是会在正数往上的





import numpy as np
from PIL import Image
import gymnasium as gym
import os
import cv2
import datetime


class SINFSV4Environment:
    def __init__(self, config):
        self.environmentName = "SpaceInvadersNoFrameskip-v4"
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

    def record_random_episodes(self, num_episodes=3, output_dir="random_episodes", max_steps=500):
        """
        在训练开始前记录随机策略的多个回合

        Args:
            num_episodes: 要记录的回合数
            output_dir: 保存视频和图片的目录
            max_steps: 每个回合的最大步数
        """
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        print(f"开始记录 {num_episodes} 个随机策略回合到目录: {output_dir}")

        for episode in range(1, num_episodes + 1):
            print(f"记录第 {episode}/{num_episodes} 个随机回合...")

            # 重置环境
            state, info = self.env.reset()
            frames = [state.copy()]

            done = False
            total_reward = 0
            step_count = 0

            # 运行随机策略
            while not done and step_count < max_steps:
                # 随机选择动作
                action = self.env.action_space.sample()

                # 执行动作
                next_state, reward, done, info = self.step(action)
                frames.append(next_state.copy())

                total_reward += reward
                step_count += 1
                state = next_state

            # 保存这个回合的视频和关键帧
            self._save_episode(frames, episode, output_dir, total_reward, step_count)

            print(f"回合 {episode} 完成: 总奖励 = {total_reward}, 步数 = {step_count}")

        print("所有随机回合记录完成！")

    def _save_episode(self, frames, episode_num, output_dir, total_reward, step_count):
        """保存单个回合的视频和关键帧"""
        if not frames:
            return

        # 生成时间戳
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # 创建这个回合的专属目录
        episode_dir = os.path.join(output_dir, f"episode_{episode_num}_{timestamp}")
        os.makedirs(episode_dir, exist_ok=True)

        # 保存视频
        if len(frames) > 0:
            height, width = frames[0].shape[:2]
            fps = 30  # 帧率

            video_filename = f"episode_{episode_num}.mp4"
            video_path = os.path.join(episode_dir, video_filename)

            # 使用OpenCV创建视频写入器
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

            # 写入所有帧
            for frame in frames:
                # 确保帧的数据类型是uint8
                if frame.dtype != np.uint8:
                    frame = frame.astype(np.uint8)

                # 转换颜色空间从RGB到BGR（OpenCV使用BGR）
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                video_writer.write(frame_bgr)

            video_writer.release()

        # 保存关键帧图片
        if len(frames) > 0:
            # 保存初始帧
            initial_frame = frames[0]
            if initial_frame.dtype != np.uint8:
                initial_frame = initial_frame.astype(np.uint8)
            initial_path = os.path.join(episode_dir, "initial_frame.png")
            cv2.imwrite(initial_path, cv2.cvtColor(initial_frame, cv2.COLOR_RGB2BGR))

            # 保存结束帧（如果有多个帧）
            if len(frames) > 1:
                final_frame = frames[-1]
                if final_frame.dtype != np.uint8:
                    final_frame = final_frame.astype(np.uint8)
                final_path = os.path.join(episode_dir, "final_frame.png")
                cv2.imwrite(final_path, cv2.cvtColor(final_frame, cv2.COLOR_RGB2BGR))

            # 保存一些中间关键帧（每25%的进度保存一帧）
            if len(frames) > 4:
                intervals = [0.25, 0.5, 0.75]
                for interval in intervals:
                    idx = min(len(frames) - 1, int(len(frames) * interval))
                    key_frame = frames[idx]
                    if key_frame.dtype != np.uint8:
                        key_frame = key_frame.astype(np.uint8)
                    key_path = os.path.join(episode_dir, f"frame_{int(interval * 100)}percent.png")
                    cv2.imwrite(key_path, cv2.cvtColor(key_frame, cv2.COLOR_RGB2BGR))

        # 保存回合信息到文本文件
        info_filename = "episode_info.txt"
        info_path = os.path.join(episode_dir, info_filename)

        with open(info_path, 'w') as f:
            f.write(f"随机策略回合信息\n")
            f.write(f"==================\n")
            f.write(f"回合编号: {episode_num}\n")
            f.write(f"记录时间: {timestamp}\n")
            f.write(f"总奖励: {total_reward}\n")
            f.write(f"总步数: {step_count}\n")
            f.write(f"环境名称: {self.environmentName}\n")
            f.write(f"动作空间大小: {self.env.action_space.n}\n")

        print(f"回合 {episode_num} 的记录已保存到: {episode_dir}")
























