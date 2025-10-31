"""
Memory management and experience replay buffer
"""

import torch
import random
from collections import deque, namedtuple
from typing import Tuple, List
from Log import memoryMonitor, MemoryManager
import logging


Experience = namedtuple('Experience', ['state', 'action', 'reward', 'nextState', 'done'])


class ExperienceReplayBuffer:
    """GPU-accelerated experience replay buffer"""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
        self.gpuMemoryWarningIssued = False

    def push(self, state: torch.Tensor, action: int, reward: float,
             nextState: torch.Tensor, done: bool) -> None:
        """Add experience to buffer"""
        with memoryMonitor("ExperienceReplayBuffer Push"):
            try:
                # Ensure states are on GPU for training acceleration
                stateGpu = state.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")) if not state.is_cuda else state
                nextStateGpu = nextState.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")) if not nextState.is_cuda else nextState

                self.buffer.append(Experience(stateGpu, action, reward, nextStateGpu, done))

                # Memory usage monitoring
                if len(self.buffer) % 1000 == 0:
                    self._checkMemoryUsage()

            except torch.cuda.OutOfMemoryError as e:
                logging.warning(f"GPU memory不足，fallback to CPU storage: {e}")
                # Fallback to CPU storage
                stateCpu = state.cpu() if state.is_cuda else state
                nextStateCpu = nextState.cpu() if nextState.is_cuda else nextState
                self.buffer.append(Experience(stateCpu, action, reward, nextStateCpu, done))

            except Exception as e:
                logging.error(f"Failed to add experience to buffer: {e}")

    def sample(self, batchSize: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a batch of experiences from buffer"""
        if len(self.buffer) < batchSize:
            raise ValueError(f"Insufficient experiences in buffer: {len(self.buffer)} < {batchSize}")

        with memoryMonitor("ExperienceReplayBuffer Sample"):
            try:
                experiences = random.sample(self.buffer, batchSize)

                # Use list comprehension for efficiency
                states = torch.stack([exp.state for exp in experiences])
                actions = torch.tensor([exp.action for exp in experiences], dtype=torch.long,
                                     device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
                rewards = torch.tensor([exp.reward for exp in experiences], dtype=torch.float32,
                                     device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
                nextStates = torch.stack([exp.nextState for exp in experiences])
                dones = torch.tensor([exp.done for exp in experiences], dtype=torch.float32,
                                   device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

                return states, actions, rewards, nextStates, dones

            except torch.cuda.OutOfMemoryError as e:
                logging.error(f"GPU memory不足 during sampling: {e}")
                self._emergencyMemoryCleanup()
                raise

            except Exception as e:
                logging.error(f"Failed to sample experiences: {e}")
                raise

    def _checkMemoryUsage(self):
        """Check GPU memory usage"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024 ** 3
            if allocated > 4.0 and not self.gpuMemoryWarningIssued:
                logging.warning(f"High GPU memory usage: {allocated:.2f} GB")
                self.gpuMemoryWarningIssued = True

    def _emergencyMemoryCleanup(self):
        """Emergency memory cleanup"""
        logging.warning("Performing emergency memory cleanup...")
        MemoryManager.clearMemory()

    def __len__(self) -> int:
        return len(self.buffer)

    def getUsagePercentage(self) -> float:
        """Return buffer usage percentage"""
        return len(self.buffer) / self.capacity * 100

    def clearGpuMemory(self):
        """Clear state tensors in GPU memory"""
        logging.info("Clearing experience buffer GPU memory...")
        # Use list comprehension for efficiency
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
        logging.info("GPU memory cleanup completed")




