"""
PPO (Proximal Policy Optimization) 算法实现 - GPU加速版本
基于原DQN代码框架，按照PPO论文完整复现
V1.1 2025.10.30 - PPO实现 + GPU经验缓冲区加速

整理优化：
1. 增强代码可读性和结构
2. 改进错误处理和鲁棒性
3. 优化内存管理和GPU使用
4. 添加详细的类型注解和文档
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque, namedtuple
import gymnasium as gym
import matplotlib.pyplot as plt
from PIL import Image
import logging
from typing import Tuple, List, Dict, Any, Optional, Generator
import gc
import psutil
import os
import time
from dataclasses import dataclass
from contextlib import contextmanager
import json
from datetime import datetime


# =============================== 配置类 ===============================
@dataclass
class PPOConfig:
    """PPO训练配置参数"""
    environment_name: str = "PongNoFrameskip-v4"
    learning_rate: float = 0.00025
    clip_epsilon: float = 0.1
    discount_factor: float = 0.99
    gae_lambda: float = 0.95
    value_loss_coeff: float = 0.5
    entropy_coeff: float = 0.01
    ppo_epochs: int = 3
    batch_size: int = 32
    horizon: int = 128  # 每个actor的时间步数
    num_actors: int = 8  # 并行actor数量
    training_timesteps: int = 10000000
    target_average_reward: float = 15.0
    frame_skip: int = 4
    screen_size: int = 84
    save_images: bool = True
    image_save_dir: str = "recordedPongPPO"
    num_episodes_to_record: int = 3
    use_adam: bool = True
    adam_epsilon: float = 1e-5
    max_grad_norm: float = 0.5
    buffer_on_gpu: bool = True  # 经验缓冲区是否放在GPU上
    log_interval: int = 5  # 日志输出间隔
    save_interval: int = 50  # 模型保存间隔


# =============================== 全局设置 ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./log/ppo_resnet_gpu_memory.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# =============================== 工具类 ===============================
class MemoryManager:
    """内存管理器，用于监控和优化内存使用"""

    @staticmethod
    def get_gpu_memory_usage() -> Tuple[float, float]:
        """获取GPU内存使用情况"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024 ** 3  # GB
            cached = torch.cuda.memory_reserved() / 1024 ** 3  # GB
            return allocated, cached
        return 0.0, 0.0

    @staticmethod
    def get_system_memory_usage() -> float:
        """获取系统内存使用情况"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 ** 3  # GB

    @staticmethod
    def clear_memory():
        """清理内存"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


@contextmanager
def memory_monitor(operation_name: str):
    """监控内存使用的上下文管理器"""
    start_time = time.time()
    gpu_memory_before = MemoryManager.get_gpu_memory_usage()
    system_memory_before = MemoryManager.get_system_memory_usage()

    try:
        yield
    finally:
        end_time = time.time()
        gpu_memory_after = MemoryManager.get_gpu_memory_usage()
        system_memory_after = MemoryManager.get_system_memory_usage()

        logger.debug(
            f"{operation_name} - "
            f"耗时: {end_time - start_time:.3f}s | "
            f"GPU内存变化: {gpu_memory_after[0] - gpu_memory_before[0]:+.2f}GB | "
            f"系统内存变化: {system_memory_after - system_memory_before:+.2f}GB"
        )


# =============================== 网络结构 ===============================
class ResidualBlock(nn.Module):
    """修复的残差块，确保尺寸匹配"""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 快捷连接 - 确保尺寸匹配
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        residual = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # 确保残差连接尺寸匹配
        residual = self.shortcut(residual)
        out += residual
        out = F.relu(out)

        return out


class PPONetwork(nn.Module):
    """PPO网络 - 包含策略网络和价值网络"""

    def __init__(self, input_shape: Tuple[int, int, int], num_actions: int):
        super().__init__()

        self.in_channels = 64
        self.num_actions = num_actions

        # 共享的特征提取层
        self.conv1 = nn.Conv2d(input_shape[0], 64, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # ResNet层 - 适配84x84输入
        self.layer1 = self._make_layer(64, 64, 2, stride=1)  # 84x84 -> 84x84
        self.layer2 = self._make_layer(64, 128, 2, stride=2)  # 84x84 -> 42x42
        self.layer3 = self._make_layer(128, 256, 2, stride=2)  # 42x42 -> 21x21
        self.layer4 = self._make_layer(256, 512, 2, stride=2)  # 21x21 -> 11x11

        # 自适应平均池化到固定尺寸
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 策略头
        self.policy_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )

        # 价值头
        self.value_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        # 初始化权重
        self._initialize_weights()

    def _make_layer(self, in_channels: int, out_channels: int,
                   num_blocks: int, stride: int) -> nn.Sequential:
        """创建ResNet层"""
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for current_stride in strides:
            layers.append(ResidualBlock(in_channels, out_channels, current_stride))
            in_channels = out_channels

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """前向传播 - 返回动作概率、状态价值和动作log概率"""
        # 共享特征提取
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.adaptive_avg_pool(x)
        x = x.view(x.size(0), -1)

        # 策略头
        policy_logits = self.policy_head(x)
        action_probs = F.softmax(policy_logits, dim=-1)

        # 价值头
        state_value = self.value_head(x)

        return action_probs, state_value.squeeze(-1), policy_logits

    def get_action(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor,
                                                     torch.Tensor, torch.Tensor]:
        """选择动作并返回相关信息"""
        with torch.no_grad():
            action_probs, state_value, policy_logits = self.forward(state)
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            return action, log_prob, state_value, action_probs


# =============================== 经验缓冲区 ===============================
PPOExperience = namedtuple('PPOExperience',
                          ['state', 'action', 'reward', 'value', 'log_prob', 'done'])


class PPOBuffer:
    """PPO专用的经验缓冲区 - GPU加速版本"""

    def __init__(self, horizon: int, num_actors: int,
                 state_shape: Tuple[int, int, int], device: torch.device = device):
        self.horizon = horizon
        self.num_actors = num_actors
        self.state_shape = state_shape
        self.device = device

        # 初始化缓冲区 - 直接在GPU上创建
        self.states = torch.zeros((horizon, num_actors) + state_shape, device=device)
        self.actions = torch.zeros((horizon, num_actors), dtype=torch.long, device=device)
        self.rewards = torch.zeros((horizon, num_actors), device=device)
        self.values = torch.zeros((horizon, num_actors), device=device)
        self.log_probs = torch.zeros((horizon, num_actors), device=device)
        self.dones = torch.zeros((horizon, num_actors), dtype=torch.bool, device=device)

        self.advantages = torch.zeros((horizon, num_actors), device=device)
        self.returns = torch.zeros((horizon, num_actors), device=device)

        self.step = 0

    def push(self, state: torch.Tensor, action: torch.Tensor, reward: torch.Tensor,
             value: torch.Tensor, log_prob: torch.Tensor, done: torch.Tensor):
        """添加经验到缓冲区 - 所有张量已经在GPU上"""
        if self.step < self.horizon:
            self.states[self.step] = state
            self.actions[self.step] = action
            self.rewards[self.step] = reward
            self.values[self.step] = value
            self.log_probs[self.step] = log_prob
            self.dones[self.step] = done

            self.step += 1

    def compute_advantages_and_returns(self, last_values: torch.Tensor,
                                     gamma: float = 0.99, gae_lambda: float = 0.95):
        """计算优势函数和回报 - 在GPU上执行"""
        advantages = torch.zeros_like(self.rewards)
        last_advantage = 0

        for t in reversed(range(self.horizon)):
            if t == self.horizon - 1:
                next_value = last_values
                next_non_terminal = 1.0 - self.dones[t].float()
            else:
                next_value = self.values[t + 1]
                next_non_terminal = 1.0 - self.dones[t].float()

            delta = (self.rewards[t] + gamma * next_value * next_non_terminal
                    - self.values[t])
            advantages[t] = last_advantage = (delta + gamma * gae_lambda
                                            * next_non_terminal * last_advantage)

        self.returns = advantages + self.values

        # 标准化优势函数
        if advantages.std() > 0:  # 避免除零
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        self.advantages = advantages

    def get_batches(self, batch_size: int) -> Generator:
        """生成训练批次 - 数据已经在GPU上，无需转移"""
        # 展平所有数据
        states = self.states.view(-1, *self.state_shape)
        actions = self.actions.view(-1)
        old_log_probs = self.log_probs.view(-1)
        advantages = self.advantages.view(-1)
        returns = self.returns.view(-1)

        # 随机打乱
        indices = torch.randperm(states.size(0), device=self.device)

        for start_idx in range(0, states.size(0), batch_size):
            end_idx = min(start_idx + batch_size, states.size(0))
            batch_indices = indices[start_idx:end_idx]

            yield (
                states[batch_indices],
                actions[batch_indices],
                old_log_probs[batch_indices],
                advantages[batch_indices],
                returns[batch_indices]
            )

    def clear(self):
        """清空缓冲区"""
        self.step = 0
        self.advantages.zero_()
        self.returns.zero_()


# =============================== 环境处理 ===============================
class AtariEnvironmentPreprocessor:
    """Atari环境预处理包装器"""

    def __init__(self, environment, frame_skip: int = 4, screen_size: int = 84):
        self.environment = environment
        self.frame_skip = frame_skip
        self.screen_size = screen_size
        self.frame_buffer = deque(maxlen=4)

    def reset(self) -> Tuple[torch.Tensor, dict]:
        """重置环境并返回预处理后的初始状态"""
        try:
            state, info = self.environment.reset()
            processed_state = self._preprocess_frame(state)

            # 用相同的帧填充初始缓冲区
            self.frame_buffer.extend([processed_state] * 4)

            state_tensor = torch.tensor(np.stack(self.frame_buffer), dtype=torch.float32)
            return state_tensor, info
        except Exception as e:
            logger.error(f"重置环境失败: {e}")
            raise

    def step(self, action: int) -> Tuple[torch.Tensor, float, bool, dict]:
        """执行动作并返回预处理后的结果"""
        try:
            total_reward = 0.0
            terminated = False
            truncated = False
            info = {}

            # 使用帧跳过提高效率
            for _ in range(self.frame_skip):
                next_state, reward, terminated, truncated, step_info = self.environment.step(action)
                total_reward += reward
                info.update(step_info)

                if terminated or truncated:
                    break

            done = terminated or truncated
            processed_next_state = self._preprocess_frame(next_state)
            self.frame_buffer.append(processed_next_state)

            next_state_tensor = torch.tensor(np.stack(self.frame_buffer), dtype=torch.float32)
            return next_state_tensor, total_reward, done, info
        except Exception as e:
            logger.error(f"执行动作失败: {e}")
            raise

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """预处理帧：灰度化、调整大小、归一化"""
        try:
            # 转换为灰度图
            if len(frame.shape) == 3:
                frame = np.mean(frame, axis=2)  # 使用numpy提高效率

            # 调整大小
            img = Image.fromarray(frame.astype(np.uint8))
            img = img.resize((self.screen_size, self.screen_size), Image.BILINEAR)
            frame = np.array(img)

            # 归一化到 [0, 1]
            frame = frame.astype(np.float32) / 255.0

            return frame
        except Exception as e:
            logger.error(f"预处理帧失败: {e}")
            raise

    @property
    def action_space(self):
        return self.environment.action_space

    @property
    def observation_space(self):
        return self.environment.observation_space

    def close(self) -> None:
        """关闭环境"""
        self.environment.close()


class EnvironmentRecorder:
    """环境记录器，用于保存原始环境数据为PNG图片"""

    def __init__(self, config: PPOConfig):
        self.config = config
        self.environment = gym.make(config.environment_name, render_mode='rgb_array')
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

        logger.info(f"环境数据将保存到: {self.save_dir}")

    def preprocess_frame(self, frame):
        """预处理帧 - 这里保存原始帧，不进行灰度化等处理"""
        return frame

    def save_frame_with_info(self, frame, episode, step, action, reward,
                           terminated, truncated, info):
        """保存帧图像和相关信息"""
        try:
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
        except Exception as e:
            logger.error(f"保存帧信息失败: {e}")
            return None

    def record_episode(self, episode_num, max_steps=1000):
        """记录一个完整的episode"""
        try:
            state, info = self.environment.reset()
            episode_frames = []
            total_reward = 0

            episode_dir = os.path.join(self.episodes_dir, f"episode_{episode_num:04d}")
            os.makedirs(episode_dir, exist_ok=True)

            for step in range(max_steps):
                # 随机动作
                action = self.environment.action_space.sample()

                next_state, reward, terminated, truncated, info = self.environment.step(action)

                # 保存当前帧和相关信息
                frame_info = self.save_frame_with_info(
                    state, episode_num, step, action, reward, terminated, truncated, info
                )
                if frame_info:
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
                'environment': self.config.environment_name,
                'timestamp': datetime.now().isoformat()
            }

            metadata_file = os.path.join(episode_dir, "metadata.json")
            with open(metadata_file, 'w') as f:
                json.dump(episode_metadata, f, indent=2)

            self.episode_metadata.append(episode_metadata)
            logger.info(f"Episode {episode_num}: {step + 1} 步, 总奖励: {total_reward}")

            return total_reward
        except Exception as e:
            logger.error(f"记录episode失败: {e}")
            return 0.0

    def save_summary(self):
        """保存所有episode的摘要信息"""
        try:
            summary = {
                'total_episodes': len(self.episode_metadata),
                'environment': self.config.environment_name,
                'config': self.config.__dict__,
                'episodes': self.episode_metadata,
                'recorded_at': datetime.now().isoformat()
            }

            summary_file = os.path.join(self.save_dir, "recording_summary.json")
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)

            logger.info(f"摘要已保存到: {summary_file}")
        except Exception as e:
            logger.error(f"保存摘要失败: {e}")

    def close(self):
        """关闭环境并保存摘要"""
        self.save_summary()
        self.environment.close()


# =============================== PPO智能体 ===============================
class PPOAgent:
    """PPO智能体"""

    def __init__(self, state_shape: Tuple[int, int, int], num_actions: int, config: PPOConfig):
        self.num_actions = num_actions
        self.state_shape = state_shape
        self.config = config

        self.network = PPONetwork(state_shape, num_actions).to(device)

        # 优化器
        if config.use_adam:
            self.optimizer = optim.Adam(
                self.network.parameters(),
                lr=config.learning_rate,
                eps=config.adam_epsilon
            )
        else:
            self.optimizer = optim.RMSprop(
                self.network.parameters(),
                lr=config.learning_rate,
                eps=config.adam_epsilon
            )

        # 训练状态
        self.steps_completed = 0
        self.episodes_completed = 0

        logger.info(f"PPO智能体初始化完成: 状态形状={state_shape}, 动作数量={num_actions}")

    def compute_loss(self, states: torch.Tensor, actions: torch.Tensor, old_log_probs: torch.Tensor,
                    advantages: torch.Tensor, returns: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """计算PPO损失"""
        action_probs, values, policy_logits = self.network(states)

        # 策略损失
        dist = torch.distributions.Categorical(action_probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        ratio = torch.exp(log_probs - old_log_probs)

        # PPO裁剪目标
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.config.clip_epsilon,
                          1.0 + self.config.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # 价值损失
        value_loss = F.mse_loss(values, returns)

        # 总损失
        total_loss = (policy_loss
                     + self.config.value_loss_coeff * value_loss
                     - self.config.entropy_coeff * entropy)

        # 额外统计信息
        clip_fraction = torch.mean((torch.abs(ratio - 1.0) > self.config.clip_epsilon).float()).item()
        explained_variance = 1 - F.mse_loss(values, returns) / returns.var()

        stats = {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
            'clip_fraction': clip_fraction,
            'explained_variance': explained_variance.item(),
            'approx_kl': (old_log_probs - log_probs).mean().item()
        }

        return total_loss, stats

    def update(self, buffer: PPOBuffer) -> Dict[str, float]:
        """使用PPO算法更新网络"""
        total_stats = {
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'entropy': 0.0,
            'clip_fraction': 0.0,
            'explained_variance': 0.0,
            'approx_kl': 0.0
        }

        num_updates = 0

        for epoch in range(self.config.ppo_epochs):
            for batch in buffer.get_batches(self.config.batch_size):
                states, actions, old_log_probs, advantages, returns = batch

                self.optimizer.zero_grad()
                loss, stats = self.compute_loss(states, actions, old_log_probs, advantages, returns)
                loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                # 累积统计信息
                for key in total_stats:
                    total_stats[key] += stats[key]
                num_updates += 1

        # 平均统计信息
        if num_updates > 0:
            for key in total_stats:
                total_stats[key] /= num_updates

        return total_stats

    def save_checkpoint(self, file_path: str) -> None:
        """保存模型检查点"""
        try:
            checkpoint = {
                'network_state': self.network.state_dict(),
                'optimizer_state': self.optimizer.state_dict(),
                'steps_completed': self.steps_completed,
                'episodes_completed': self.episodes_completed,
                'config': self.config
            }
            torch.save(checkpoint, file_path)
            logger.info(f"PPO模型已保存到: {file_path}")
        except Exception as e:
            logger.error(f"保存PPO模型失败: {e}")
            raise

    def load_checkpoint(self, file_path: str) -> None:
        """加载模型检查点"""
        try:
            checkpoint = torch.load(file_path, map_location=device)
            self.network.load_state_dict(checkpoint['network_state'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            self.steps_completed = checkpoint['steps_completed']
            self.episodes_completed = checkpoint['episodes_completed']
            logger.info(f"PPO模型已从 {file_path} 加载")
        except Exception as e:
            logger.error(f"加载PPO模型失败: {e}")
            raise

    def get_training_statistics(self) -> Dict[str, Any]:
        """获取训练统计信息"""
        gpu_allocated, gpu_cached = MemoryManager.get_gpu_memory_usage()
        system_memory = MemoryManager.get_system_memory_usage()

        return {
            'steps_completed': self.steps_completed,
            'episodes_completed': self.episodes_completed,
            'gpu_memory_allocated': gpu_allocated,
            'gpu_memory_cached': gpu_cached,
            'system_memory': system_memory,
        }

    def clear_memory(self):
        """清理内存"""
        MemoryManager.clear_memory()


# =============================== PPO训练器 ===============================
class PPOTrainer:
    """PPO训练器"""

    def __init__(self, config: PPOConfig):
        self.config = config
        self.environment = None
        self.preprocessed_environment = None
        self.agent = None
        self.buffer = None

    def initialize_environment(self) -> None:
        """初始化环境"""
        try:
            # 使用NoFrameskip版本
            self.environment = gym.make(self.config.environment_name, render_mode='rgb_array')
            self.preprocessed_environment = AtariEnvironmentPreprocessor(
                self.environment,
                frame_skip=self.config.frame_skip,
                screen_size=self.config.screen_size
            )
            logger.info(f"环境初始化完成: {self.config.environment_name}")
        except Exception as e:
            logger.error(f"环境初始化失败: {e}")
            # 回退到普通版本
            try:
                env_name = self.config.environment_name.replace("NoFrameskip", "")
                self.environment = gym.make(env_name, render_mode='rgb_array')
                self.preprocessed_environment = AtariEnvironmentPreprocessor(
                    self.environment,
                    frame_skip=self.config.frame_skip,
                    screen_size=self.config.screen_size
                )
                logger.info(f"使用回退环境: {env_name}")
            except Exception as e2:
                logger.error(f"回退环境也失败: {e2}")
                raise

    def record_initial_episodes(self) -> None:
        """在训练之前记录原始环境数据为PNG图片"""
        if not self.config.save_images:
            logger.info("图像保存功能已禁用，跳过环境记录")
            return

        logger.info("开始记录原始环境数据为PNG图片...")
        recorder = EnvironmentRecorder(self.config)

        try:
            # 记录指定数量的episode
            for episode in range(self.config.num_episodes_to_record):
                logger.info(f"记录第 {episode + 1}/{self.config.num_episodes_to_record} 个episode...")
                recorder.record_episode(episode, max_steps=500)

            logger.info(f"环境记录完成！所有帧已保存到: {recorder.save_dir}")
        except Exception as e:
            logger.error(f"环境记录过程中发生错误: {e}")
        finally:
            recorder.close()

    def collect_experience(self) -> Tuple[float, int, torch.Tensor]:
        """收集经验数据 - 确保所有数据都在GPU上"""
        total_reward = 0
        episode_count = 0

        # 初始化状态 - 确保在GPU上
        states = []
        for _ in range(self.config.num_actors):
            state, _ = self.preprocessed_environment.reset()
            states.append(state)
        states = torch.stack(states).to(device)

        # 收集经验
        for step in range(self.config.horizon):
            with torch.no_grad():
                actions, log_probs, values, _ = self.agent.network.get_action(states)

            # 执行动作
            next_states = []
            rewards = []
            dones = []
            infos = []

            for i in range(self.config.num_actors):
                next_state, reward, done, info = self.preprocessed_environment.step(actions[i].item())
                next_states.append(next_state)
                rewards.append(reward)
                dones.append(done)
                infos.append(info)

                total_reward += reward
                if done:
                    episode_count += 1

            # 转换为张量并确保在GPU上
            next_states = torch.stack(next_states).to(device)
            rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
            dones = torch.tensor(dones, dtype=torch.bool, device=device)

            # 存储经验 - 所有数据已经在GPU上
            self.buffer.push(states, actions, rewards, values, log_probs, dones)

            # 更新状态
            states = next_states

            # 如果环境结束，重置
            for i in range(self.config.num_actors):
                if dones[i]:
                    state, _ = self.preprocessed_environment.reset()
                    states[i] = state.to(device)

        # 计算最后一个状态的价值
        with torch.no_grad():
            _, last_values, _ = self.agent.network(states)

        return total_reward, episode_count, last_values

    def train(self) -> Tuple[PPOAgent, List[float], List[float]]:
        """训练PPO智能体"""
        # 首先记录原始环境数据
        self.record_initial_episodes()

        if self.preprocessed_environment is None:
            self.initialize_environment()

        num_actions = self.preprocessed_environment.action_space.n
        state_shape = (4, self.config.screen_size, self.config.screen_size)

        self.agent = PPOAgent(state_shape, num_actions, self.config)
        self.buffer = PPOBuffer(self.config.horizon, self.config.num_actors, state_shape, device)

        # 训练统计
        episode_rewards = []
        moving_average_rewards = []
        training_stats_history = []

        best_average_reward = -float('inf')
        total_timesteps = 0
        update_count = 0

        logger.info(
            f"开始训练 {self.config.environment_name}, 使用PPO算法, 目标时间步数: {self.config.training_timesteps}")

        while total_timesteps < self.config.training_timesteps:
            try:
                # 收集经验
                total_reward, episode_count, last_values = self.collect_experience()
                total_timesteps += self.config.horizon * self.config.num_actors
                self.agent.steps_completed = total_timesteps
                self.agent.episodes_completed += episode_count

                # 计算优势函数和回报 - 在GPU上执行
                self.buffer.compute_advantages_and_returns(
                    last_values,
                    self.config.discount_factor,
                    self.config.gae_lambda
                )

                # 更新网络
                stats = self.agent.update(self.buffer)
                training_stats_history.append(stats)
                update_count += 1

                # 记录奖励
                if episode_count > 0:
                    episode_rewards.append(total_reward / episode_count)
                else:
                    episode_rewards.append(total_reward)

                # 计算移动平均奖励
                if len(episode_rewards) >= 50:
                    moving_average = np.mean(episode_rewards[-50:])
                else:
                    moving_average = np.mean(episode_rewards)
                moving_average_rewards.append(moving_average)

                # 更新最佳模型
                if moving_average > best_average_reward and len(episode_rewards) >= 20:
                    best_average_reward = moving_average
                    self.agent.save_checkpoint(
                        f"./PPO_V1_0_models/ppo_resnet_best_{self.config.environment_name.replace('/', '_')}.pth"
                    )

                # 定期日志输出
                if update_count % self.config.log_interval == 0:
                    training_stats = self.agent.get_training_statistics()
                    logger.info(
                        f"时间步 {total_timesteps:8d} | "
                        f"平均奖励: {episode_rewards[-1]:7.2f} | "
                        f"移动平均: {moving_average:7.2f} | "
                        f"策略损失: {stats['policy_loss']:7.4f} | "
                        f"价值损失: {stats['value_loss']:7.4f} | "
                        f"熵: {stats['entropy']:7.4f} | "
                        f"裁剪比例: {stats['clip_fraction']:7.4f}"
                    )

                # 定期保存检查点
                if update_count % self.config.save_interval == 0 and update_count > 0:
                    self.agent.save_checkpoint(
                        f"./PPO_V1_0_models/ppo_resnet_checkpoint_{self.config.environment_name.replace('/', '_')}_step_{total_timesteps}.pth"
                    )

                # 检查停止条件
                if (len(moving_average_rewards) >= 30 and
                        moving_average_rewards[-1] >= self.config.target_average_reward):
                    logger.info(f"达到目标性能! 在时间步 {total_timesteps}")
                    self.agent.save_checkpoint(
                        f"./PPO_V1_0_models/ppo_resnet_final_{self.config.environment_name.replace('/', '_')}.pth"
                    )
                    break

                # 清空缓冲区
                self.buffer.clear()

            except Exception as e:
                logger.error(f"训练过程中发生错误: {e}")
                MemoryManager.clear_memory()
                continue

        self._plot_training_results(episode_rewards, moving_average_rewards, training_stats_history)
        return self.agent, episode_rewards, moving_average_rewards

    def _plot_training_results(self, episode_rewards: List[float],
                             moving_average_rewards: List[float],
                             training_stats: List[Dict[str, float]]) -> None:
        """绘制训练结果图表"""
        try:
            plt.figure(figsize=(15, 12))

            # 奖励曲线
            plt.subplot(3, 2, 1)
            plt.plot(episode_rewards, alpha=0.6, label='每回合奖励')
            plt.plot(moving_average_rewards, 'r-', linewidth=2, label='移动平均奖励 (50回合)')
            plt.title(f'{self.config.environment_name} - PPO回合奖励')
            plt.xlabel('更新次数')
            plt.ylabel('奖励')
            plt.legend()
            plt.grid(True)

            # 策略损失
            plt.subplot(3, 2, 2)
            policy_losses = [stats['policy_loss'] for stats in training_stats]
            plt.plot(policy_losses)
            plt.title(f'{self.config.environment_name} - PPO策略损失')
            plt.xlabel('更新次数')
            plt.ylabel('策略损失')
            plt.grid(True)

            # 价值损失
            plt.subplot(3, 2, 3)
            value_losses = [stats['value_loss'] for stats in training_stats]
            plt.plot(value_losses)
            plt.title(f'{self.config.environment_name} - PPO价值损失')
            plt.xlabel('更新次数')
            plt.ylabel('价值损失')
            plt.grid(True)

            # 熵
            plt.subplot(3, 2, 4)
            entropies = [stats['entropy'] for stats in training_stats]
            plt.plot(entropies)
            plt.title(f'{self.config.environment_name} - PPO熵')
            plt.xlabel('更新次数')
            plt.ylabel('熵')
            plt.grid(True)

            # 裁剪比例
            plt.subplot(3, 2, 5)
            clip_fractions = [stats['clip_fraction'] for stats in training_stats]
            plt.plot(clip_fractions)
            plt.title(f'{self.config.environment_name} - PPO裁剪比例')
            plt.xlabel('更新次数')
            plt.ylabel('裁剪比例')
            plt.grid(True)

            # KL散度
            plt.subplot(3, 2, 6)
            approx_kls = [stats['approx_kl'] for stats in training_stats]
            plt.plot(approx_kls)
            plt.title(f'{self.config.environment_name} - PPO近似KL散度')
            plt.xlabel('更新次数')
            plt.ylabel('KL散度')
            plt.grid(True)

            plt.tight_layout()
            plot_file_name = f'ppo_training_results_{self.config.environment_name.replace("/", "_")}.png'
            plt.savefig(plot_file_name, dpi=150, bbox_inches='tight')
            plt.close()
            logger.info(f"训练图表已保存到: {plot_file_name}")
        except Exception as e:
            logger.error(f"绘制训练结果失败: {e}")

    def close(self) -> None:
        """关闭环境"""
        if self.preprocessed_environment:
            self.preprocessed_environment.close()
        MemoryManager.clear_memory()


# =============================== 主函数 ===============================
def main():
    """主函数"""
    try:
        # 设置内存优化
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True

        # 创建必要的目录
        os.makedirs('./log', exist_ok=True)
        os.makedirs('./PPO_V1_0_models', exist_ok=True)

        # PPO训练配置 - 按照论文中的超参数
        config = PPOConfig()

        logger.info("开始PPO训练!")

        # 创建训练器并开始训练
        trainer = PPOTrainer(config)
        trained_agent, rewards, moving_averages = trainer.train()

        # 输出训练结果
        logger.info("PPO训练完成!")
        if moving_averages:
            logger.info(f"最终移动平均奖励: {moving_averages[-1]:.2f}")
            logger.info(f"最大移动平均奖励: {max(moving_averages):.2f}")

        trainer.close()

    except KeyboardInterrupt:
        logger.info("训练被用户中断")
    except Exception as e:
        logger.error(f"训练过程中发生错误: {e}")
    finally:
        # 最终清理
        MemoryManager.clear_memory()
        logger.info("程序执行完毕")


if __name__ == "__main__":
    main()