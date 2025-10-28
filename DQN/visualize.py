# 将 import gym 改为 import gymnasium as gym
import gymnasium as gym
import os
import numpy as np
from PIL import Image
from dataclasses import dataclass
from datetime import datetime
import json
from dataclasses import asdict


@dataclass
class TrainingConfig:
    environmentName: str = "PongNoFrameskip-v4"
    learningRate: float = 0.00025
    discountFactor: float = 0.99
    batchSize: int = 32
    replayBufferCapacity: int = 30000
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
    save_images: bool = True
    image_save_dir: str = "recorded_pong_episodes"


class EnhancedEnvironmentRecorder:
    def __init__(self, config: TrainingConfig):
        self.config = config
        # 使用 gymnasium
        self.env = gym.make(config.environmentName, render_mode='rgb_array')
        self.setup_save_directory()
        self.episode_metadata = []

    def setup_save_directory(self):
        """创建保存目录和子目录"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = f"{self.config.image_save_dir}_{timestamp}"
        os.makedirs(self.save_dir, exist_ok=True)

        # 创建episodes子目录
        self.episodes_dir = os.path.join(self.save_dir, "episodes")
        os.makedirs(self.episodes_dir, exist_ok=True)

        print(f"环境数据将保存到: {self.save_dir}")

    def preprocess_frame(self, frame):
        """预处理帧"""
        return frame

    def save_frame_with_info(self, frame, episode, step, action, reward, terminated, truncated, info):
        """保存帧图像和相关信息"""
        # 预处理帧
        processed_frame = self.preprocess_frame(frame)

        # 确保数据格式正确
        if processed_frame.dtype != np.uint8:
            processed_frame = np.clip(processed_frame * 255, 0, 255).astype(np.uint8)

        # 创建PIL图像
        img = Image.fromarray(processed_frame)

        # 构建文件名和路径
        episode_dir = os.path.join(self.episodes_dir, f"episode_{episode:04d}")
        os.makedirs(episode_dir, exist_ok=True)

        filename = f"step_{step:04d}_action_{action}_reward_{reward:.1f}.png"
        filepath = os.path.join(episode_dir, filename)

        # 保存图像
        img.save(filepath)

        # 保存帧信息到元数据
        frame_info = {
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
            frame_info.update(info)  # 添加环境返回的info

        return frame_info

    def record_episode(self, episode_num, max_steps=1000):
        """记录一个完整的episode"""
        # Gymnasium 的 reset 返回两个值
        state, info = self.env.reset()
        episode_frames = []
        total_reward = 0

        episode_dir = os.path.join(self.episodes_dir, f"episode_{episode_num:04d}")
        os.makedirs(episode_dir, exist_ok=True)

        for step in range(max_steps):
            # 随机动作
            action = self.env.action_space.sample()

            # Gymnasium 的 step 返回五个值
            next_state, reward, terminated, truncated, info = self.env.step(action)

            # 保存当前帧和相关信息
            frame_info = self.save_frame_with_info(
                state, episode_num, step, action, reward, terminated, truncated, info
            )
            episode_frames.append(frame_info)

            # 更新状态
            state = next_state
            total_reward += reward

            # 检查是否结束
            if terminated or truncated:
                break

        # 保存episode的元数据
        episode_metadata = {
            'episode_number': episode_num,
            'total_reward': total_reward,
            'total_steps': step + 1,
            'frames': episode_frames,
            'environment': self.config.environmentName,
            'timestamp': datetime.now().isoformat()
        }

        metadata_file = os.path.join(episode_dir, "metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(episode_metadata, f, indent=2)

        self.episode_metadata.append(episode_metadata)
        print(f"Episode {episode_num}: {step + 1} 步, 总奖励: {total_reward}")

        return total_reward

    def save_summary(self):
        """保存所有episode的摘要信息"""
        summary = {
            'total_episodes': len(self.episode_metadata),
            'environment': self.config.environmentName,
            'config': asdict(self.config),
            'episodes': self.episode_metadata,
            'recorded_at': datetime.now().isoformat()
        }

        summary_file = os.path.join(self.save_dir, "recording_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"摘要已保存到: {summary_file}")

    def close(self):
        """关闭环境并保存摘要"""
        self.save_summary()
        self.env.close()


# 使用示例
def demo_environment_recording():
    config = TrainingConfig(
        environmentName="PongNoFrameskip-v4",
        save_images=True,
        image_save_dir="recorded_pong_episodes"
    )

    recorder = EnhancedEnvironmentRecorder(config)

    try:
        # 记录2个episode作为演示
        for episode in range(2):
            recorder.record_episode(episode, max_steps=500)
    finally:
        recorder.close()

    print("环境录制完成！所有帧已保存为PNG格式。")


if __name__ == "__main__":
    demo_environment_recording()