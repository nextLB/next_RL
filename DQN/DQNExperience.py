"""
    各环境信息经验池的构建
"""


import torch
import random
from collections import deque, namedtuple

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'nextState', 'done'])



class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)


    def push(self, state: torch.Tensor, action: int, reward: float, nextState: torch.Tensor, done: bool) -> None:
        """添加经验到缓冲区"""
        # 确保状态在GPU上以加速训练
        nextState = torch.from_numpy(nextState)
        stateGpu = state.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        nextStateGpu = nextState.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.buffer.append(Experience(stateGpu, action, reward, nextStateGpu, done))

    def sample(self, batchSize: int) -> tuple:
        """从缓冲区中随机采样经验"""
        experiences = random.sample(self.buffer, batchSize)
        # 使用列表推导式提高效率
        states = torch.stack([exp.state for exp in experiences])
        actions = torch.tensor([exp.action for exp in experiences], dtype=torch.long, device=states.device)
        rewards = torch.tensor([exp.reward for exp in experiences], dtype=torch.float32, device=states.device)
        nextStates = torch.stack([exp.nextState for exp in experiences])
        dones = torch.tensor([exp.done for exp in experiences], dtype=torch.float32, device=states.device)
        return states, actions, rewards, nextStates, dones


    def __len__(self) -> int:
        return len(self.buffer)




