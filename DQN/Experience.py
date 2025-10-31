"""
经验回放缓冲区模块
"""

import torch
import random
from collections import deque, namedtuple
from Log import memoryMonitor
from Memory import MemoryManager
import logging


logger = logging.getLogger(__name__)

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'nextState', 'done'])


class ExperienceReplayBuffer:
    """GPU加速的经验回放缓冲区"""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
        self.gpuMemoryWarningIssued = False

    def push(self, state: torch.Tensor, action: int, reward: float, nextState: torch.Tensor, done: bool) -> None:
        """添加经验到缓冲区"""
        with memoryMonitor("经验回放缓冲区添加"):
            try:
                # 确保状态在GPU上以加速训练
                stateGpu = state.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")) if not state.is_cuda else state
                nextStateGpu = nextState.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")) if not nextState.is_cuda else nextState

                self.buffer.append(Experience(stateGpu, action, reward, nextStateGpu, done))

                # 内存使用监控
                if len(self.buffer) % 1000 == 0:
                    self._checkMemoryUsage()

            except torch.cuda.OutOfMemoryError as e:
                logger.warning(f"GPU显存不足，回退到CPU存储: {e}")
                # 回退到CPU存储
                stateCpu = state.cpu() if state.is_cuda else state
                nextStateCpu = nextState.cpu() if nextState.is_cuda else nextState
                self.buffer.append(Experience(stateCpu, action, reward, nextStateCpu, done))

            except Exception as e:
                logger.error(f"添加经验到缓冲区失败: {e}")

    def sample(self, batchSize: int) -> tuple:
        """从缓冲区中随机采样一批经验"""
        if len(self.buffer) < batchSize:
            raise ValueError(f"缓冲区中经验数量不足: {len(self.buffer)} < {batchSize}")

        with memoryMonitor("经验回放缓冲区采样"):
            try:
                experiences = random.sample(self.buffer, batchSize)

                # 使用列表推导式提高效率
                states = torch.stack([exp.state for exp in experiences])
                actions = torch.tensor([exp.action for exp in experiences], dtype=torch.long, device=states.device)
                rewards = torch.tensor([exp.reward for exp in experiences], dtype=torch.float32, device=states.device)
                nextStates = torch.stack([exp.nextState for exp in experiences])
                dones = torch.tensor([exp.done for exp in experiences], dtype=torch.float32, device=states.device)

                return states, actions, rewards, nextStates, dones

            except torch.cuda.OutOfMemoryError as e:
                logger.error(f"采样时GPU显存不足: {e}")
                self._emergencyMemoryCleanup()
                raise

            except Exception as e:
                logger.error(f"采样经验失败: {e}")
                raise

    def _checkMemoryUsage(self):
        """检查GPU内存使用情况"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024 ** 3
            if allocated > 4.0 and not self.gpuMemoryWarningIssued:
                logger.warning(f"GPU内存使用较高: {allocated:.2f} GB")
                self.gpuMemoryWarningIssued = True

    def _emergencyMemoryCleanup(self):
        """紧急内存清理"""
        logger.warning("执行紧急内存清理...")
        MemoryManager.clearMemory()

    def __len__(self) -> int:
        return len(self.buffer)

    def getUsagePercentage(self) -> float:
        """返回缓冲区的使用百分比"""
        return len(self.buffer) / self.capacity * 100

    def clearGpuMemory(self):
        """清理GPU内存中的状态张量"""
        logger.info("清理GPU内存中的经验缓冲区...")
        # 使用列表推导式提高效率
        cpuBuffer = deque(
            [
                Experience(
                    exp.state.cpu() if exp.state.is_cuda else exp.state,
                    exp.action,
                    exp.reward,
                    exp.nextState.cpu() if exp.nextState.is_cuda else exp.nextState,
                    exp.done
                )
                for exp in self.buffer
            ],
            maxlen=self.capacity
        )

        self.buffer = cpuBuffer
        MemoryManager.clearMemory()
        logger.info("GPU内存清理完成")