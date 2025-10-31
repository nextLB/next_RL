"""
A3C简化修复版本 - 修复梯度计算问题
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import gymnasium as gym
from PIL import Image
import logging
from typing import Tuple, List, Dict, Any, Optional
import os
import time
import threading
from threading import Lock
import random


# =============================== 配置类 ===============================
class A3CConfig:
    """A3C训练配置参数 - 简化版本"""
    def __init__(self):
        self.environmentName = "BreakoutNoFrameskip-v4"
        self.learningRate = 0.0007
        self.discountFactor = 0.99
        self.entropyCoeff = 0.01
        self.valueLossCoeff = 0.5
        self.maxGradNorm = 40.0
        self.nStep = 5  # 减少n-step加快更新
        self.numProcesses = 2  # 进一步减少线程数
        self.trainingTimesteps = 100000
        self.frameSkip = 4
        self.screenSize = 84
        self.useLSTM = False
        self.logInterval = 1  # 每个episode都记录
        self.saveInterval = 20
        self.maxEpisodeLength = 1000


# =============================== 全局设置 ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


# =============================== 简化网络结构 ===============================
class SimpleA3CNetwork(nn.Module):
    """简化的A3C网络 - 避免复杂结构导致的问题"""

    def __init__(self, inputShape: Tuple[int, int, int], numActions: int):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(inputShape[0], 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.policy = nn.Linear(64, numActions)
        self.value = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batchSize = x.size(0)
        features = self.conv(x).view(batchSize, -1)

        policyLogits = self.policy(features)
        actionProbs = F.softmax(policyLogits, dim=-1)
        stateValue = self.value(features).squeeze(-1)

        return actionProbs, stateValue

    def getAction(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """选择动作 - 简化版本"""
        with torch.no_grad():
            actionProbs, stateValue = self.forward(state)
            dist = torch.distributions.Categorical(actionProbs)
            action = dist.sample()
            logProb = dist.log_prob(action)

            return action, logProb, stateValue


# =============================== 环境处理 ===============================
class SimpleEnvironmentWrapper:
    """简化的环境包装器"""

    def __init__(self, envName: str, frameSkip: int = 4, screenSize: int = 84):
        self.env = gym.make(envName, render_mode='rgb_array')
        self.frameSkip = frameSkip
        self.screenSize = screenSize
        self.frameBuffer = deque(maxlen=4)

    def reset(self) -> torch.Tensor:
        state, _ = self.env.reset()
        processedState = self._preprocessFrame(state)

        # 填充初始帧缓冲区
        self.frameBuffer.extend([processedState] * 4)

        return torch.tensor(np.stack(self.frameBuffer), dtype=torch.float32)

    def step(self, action: int) -> Tuple[torch.Tensor, float, bool]:
        totalReward = 0.0
        done = False

        for _ in range(self.frameSkip):
            nextState, reward, terminated, truncated, _ = self.env.step(action)
            totalReward += reward
            if terminated or truncated:
                done = True
                break

        processedNextState = self._preprocessFrame(nextState)
        self.frameBuffer.append(processedNextState)

        nextStateTensor = torch.tensor(np.stack(self.frameBuffer), dtype=torch.float32)
        return nextStateTensor, totalReward, done

    def _preprocessFrame(self, frame: np.ndarray) -> np.ndarray:
        """简化预处理"""
        if len(frame.shape) == 3:
            frame = np.mean(frame, axis=2)

        img = Image.fromarray(frame.astype(np.uint8))
        img = img.resize((self.screenSize, self.screenSize), Image.BILINEAR)
        frame = np.array(img).astype(np.float32) / 255.0

        return frame

    @property
    def actionSpace(self):
        return self.env.action_space

    def close(self):
        self.env.close()


# =============================== A3C智能体 ===============================
class SimpleA3CAgent:
    """简化的A3C智能体"""

    def __init__(self, stateShape: Tuple[int, int, int], numActions: int, config: A3CConfig):
        self.globalNetwork = SimpleA3CNetwork(stateShape, numActions).to(device)
        self.optimizer = optim.Adam(self.globalNetwork.parameters(), lr=config.learningRate)

        # 共享网络
        self.globalNetwork.share_memory()

        self.globalStep = 0
        self.episodesCompleted = 0
        self.lock = Lock()
        self.config = config

        logger.info(f"简化的A3C智能体初始化完成")

    def updateModel(self, gradients: List[Tuple[torch.Tensor, torch.Tensor]]):
        """更新模型 - 简化版本"""
        with self.lock:
            self.optimizer.zero_grad()

            # 应用梯度
            for param, grad in gradients:
                if grad is not None:
                    if param.grad is None:
                        param.grad = grad.clone()
                    else:
                        param.grad += grad

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.globalNetwork.parameters(), self.config.maxGradNorm)

            # 优化步骤
            self.optimizer.step()

    def incrementCounters(self, steps: int = 1, episodes: int = 0):
        """更新计数器"""
        with self.lock:
            self.globalStep += steps
            self.episodesCompleted += episodes


# =============================== 工作线程 ===============================
class SimpleA3CWorker(threading.Thread):
    """简化的工作线程 - 确保立即开始训练"""

    def __init__(self, workerId: int, globalAgent: SimpleA3CAgent, config: A3CConfig):
        super().__init__()
        self.workerId = workerId
        self.globalAgent = globalAgent
        self.config = config

        # 本地网络
        self.localNetwork = SimpleA3CNetwork((4, 84, 84), 4).to(device)

        # 环境
        self.envWrapper = None

        logger.info(f"工作线程 {workerId} 创建完成")

    def run(self):
        """工作线程主循环 - 确保立即开始训练"""
        try:
            # 初始化环境
            self.envWrapper = SimpleEnvironmentWrapper(
                self.config.environmentName,
                self.config.frameSkip,
                self.config.screenSize
            )
            logger.info(f"工作线程 {self.workerId} 环境初始化完成")

            episodeCount = 0

            while self.globalAgent.globalStep < self.config.trainingTimesteps:
                # 同步网络
                self.localNetwork.load_state_dict(self.globalAgent.globalNetwork.state_dict())

                # 重置环境
                state = self.envWrapper.reset().to(device)

                episodeReward = 0
                episodeLength = 0
                done = False

                # 存储轨迹 - 只存储状态、动作和奖励，不存储logProbs和values
                states, actions, rewards = [], [], []

                while not done and episodeLength < self.config.maxEpisodeLength:
                    # 选择动作
                    action, logProb, value = self.localNetwork.getAction(state.unsqueeze(0))

                    # 执行动作
                    nextState, reward, done = self.envWrapper.step(action.item())
                    nextState = nextState.to(device)

                    # 存储数据
                    states.append(state)
                    actions.append(action)
                    rewards.append(reward)

                    # 更新状态
                    state = nextState
                    episodeReward += reward
                    episodeLength += 1

                    # n-step更新或episode结束
                    if len(states) >= self.config.nStep or done:
                        self.updateWithTrajectory(states, actions, rewards, done, nextState)
                        states, actions, rewards = [], [], []

                # Episode完成
                episodeCount += 1
                self.globalAgent.incrementCounters(episodes=1)

                logger.info(f"工作线程 {self.workerId} - Episode {episodeCount}: 奖励={episodeReward:.1f}, 长度={episodeLength}")

        except Exception as e:
            logger.error(f"工作线程 {self.workerId} 错误: {e}")
            import traceback
            logger.error(traceback.format_exc())
        finally:
            if self.envWrapper:
                self.envWrapper.close()

    def updateWithTrajectory(self, states, actions, rewards, done, nextState):
        """轨迹更新 - 修复梯度计算问题"""
        try:
            # 准备数据
            statesTensor = torch.stack(states).to(device)
            actionsTensor = torch.stack(actions).to(device)

            # 重新计算log probabilities和values（这次需要梯度）
            actionProbs, stateValues = self.localNetwork(statesTensor)
            dist = torch.distributions.Categorical(actionProbs)
            logProbs = dist.log_prob(actionsTensor)

            # 计算下一个状态的值（用于bootstrap）
            with torch.no_grad():
                if not done:
                    _, nextValue = self.localNetwork(nextState.unsqueeze(0))
                    nextValue = nextValue.squeeze(0)
                else:
                    nextValue = torch.tensor(0.0, device=device)

            # 计算回报
            returns = self.computeReturns(rewards, nextValue.item(), done)
            returnsTensor = torch.tensor(returns, dtype=torch.float32, device=device)

            # 计算优势
            advantages = returnsTensor - stateValues.detach()

            # 计算损失
            policyLoss = -(logProbs * advantages).mean()
            valueLoss = F.mse_loss(stateValues, returnsTensor)

            # 计算熵
            entropy = dist.entropy().mean()

            totalLoss = policyLoss + self.config.valueLossCoeff * valueLoss - self.config.entropyCoeff * entropy

            # 计算梯度
            self.localNetwork.zero_grad()
            totalLoss.backward()

            # 收集梯度
            gradients = []
            for localParam, globalParam in zip(
                self.localNetwork.parameters(),
                self.globalAgent.globalNetwork.parameters()
            ):
                if localParam.grad is not None:
                    gradients.append((globalParam, localParam.grad.clone()))

            # 更新全局模型
            self.globalAgent.updateModel(gradients)
            self.globalAgent.incrementCounters(steps=len(states))

            logger.debug(f"工作线程 {self.workerId} 成功更新, 步数: {len(states)}")

        except Exception as e:
            logger.error(f"工作线程 {self.workerId} 轨迹更新失败: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def computeReturns(self, rewards: List[float], lastValue: float, done: bool) -> List[float]:
        """计算回报"""
        returns = []
        R = lastValue if not done else 0.0

        for r in reversed(rewards):
            R = r + self.config.discountFactor * R
            returns.insert(0, R)

        return returns


# =============================== 训练器 ===============================
class SimpleA3CTrainer:
    """简化的A3C训练器"""

    def __init__(self, config: A3CConfig):
        self.config = config
        self.globalAgent = None
        self.workers = []

    def train(self):
        """开始训练 - 确保立即有输出"""
        # 初始化全局智能体
        stateShape = (4, self.config.screenSize, self.config.screenSize)
        numActions = 4  # Breakout的动作数量

        self.globalAgent = SimpleA3CAgent(stateShape, numActions, self.config)

        # 创建工作线程
        self.workers = []
        for i in range(self.config.numProcesses):
            worker = SimpleA3CWorker(i, self.globalAgent, self.config)
            self.workers.append(worker)

        logger.info(f"开始训练, 使用 {self.config.numProcesses} 个工作线程")

        # 启动工作线程
        for worker in self.workers:
            worker.start()
            logger.info(f"已启动工作线程 {worker.workerId}")
            time.sleep(0.5)  # 稍微错开启动时间

        # 监控训练进度
        try:
            lastGlobalStep = 0
            lastCheckTime = time.time()

            while self.globalAgent.globalStep < self.config.trainingTimesteps:
                time.sleep(2)  # 每2秒检查一次

                currentStep = self.globalAgent.globalStep
                currentTime = time.time()

                if currentStep > lastGlobalStep:
                    stepsPerSec = (currentStep - lastGlobalStep) / (currentTime - lastCheckTime)
                    logger.info(f"训练进度 - 全局步数: {currentStep}, 速度: {stepsPerSec:.1f} 步/秒")

                    lastGlobalStep = currentStep
                    lastCheckTime = currentTime
                else:
                    logger.warning("训练似乎没有进展...")

                    # 检查是否有存活的线程
                    aliveCount = sum(1 for w in self.workers if w.is_alive())
                    if aliveCount == 0:
                        logger.error("所有工作线程都已停止!")
                        break

        except KeyboardInterrupt:
            logger.info("训练被用户中断")
        finally:
            # 等待线程结束
            for worker in self.workers:
                if worker.is_alive():
                    worker.join(timeout=2.0)

            logger.info(f"训练结束! 最终全局步数: {self.globalAgent.globalStep}")


# =============================== 主函数 ===============================
def main():
    """主函数"""
    try:
        # 创建必要的目录
        os.makedirs('./A3C_models', exist_ok=True)

        # 配置
        config = A3CConfig()

        logger.info("开始简化的A3C训练!")

        # 训练
        trainer = SimpleA3CTrainer(config)
        trainer.train()

        logger.info("训练完成!")

    except Exception as e:
        logger.error(f"训练过程中发生错误: {e}")
        import traceback
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    print("开始简化的A3C训练测试...")
    main()
