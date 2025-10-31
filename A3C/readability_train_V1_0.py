"""
A3C Training Module - Fixed Version
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


# =============================== Configuration Class ===============================
class A3CConfig:
    """A3C training configuration parameters"""
    def __init__(self):
        self.environmentName = "BreakoutNoFrameskip-v4"
        self.learningRate = 0.0001  # 降低学习率
        self.discountFactor = 0.99
        self.entropyCoeff = 0.01
        self.valueLossCoeff = 0.5
        self.maxGradNorm = 40.0
        self.nStep = 20  # 增加n-step
        self.numProcesses = 4
        self.trainingTimesteps = 10000000  # 增加训练步数
        self.frameSkip = 4
        self.screenSize = 84
        self.useLSTM = False
        self.logInterval = 10
        self.saveInterval = 10000
        self.maxEpisodeLength = 10000
        self.modelSavePath = "./A3CModels"
        self.bestModelPath = "./A3CModels/best_model.pth"
        self.checkpointInterval = 50000
        self.rewardClip = True  # 添加奖励裁剪
        self.gradientClip = True


# =============================== Global Settings ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("a3c_training.log")
    ]
)
logger = logging.getLogger("A3C_Training")


# =============================== Neural Network ===============================
class A3CNetwork(nn.Module):
    """A3C Neural Network - 简化版本确保兼容性"""

    def __init__(self, inputShape: Tuple[int, int, int], numActions: int):
        super().__init__()

        # 使用标准架构确保兼容性
        self.convLayers = nn.Sequential(
            nn.Conv2d(inputShape[0], 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7))
        )

        # 计算特征大小
        with torch.no_grad():
            sampleInput = torch.zeros(1, *inputShape)
            convOutput = self.convLayers(sampleInput)
            self.featureSize = convOutput.view(1, -1).size(1)

        # 简化的策略和值头
        self.policyHead = nn.Linear(self.featureSize, numActions)
        self.valueHead = nn.Linear(self.featureSize, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batchSize = x.size(0)
        features = self.convLayers(x).view(batchSize, -1)

        policyLogits = self.policyHead(features)
        actionProbs = F.softmax(policyLogits, dim=-1)
        stateValue = self.valueHead(features).squeeze(-1)

        return actionProbs, stateValue

    def getAction(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """选择动作"""
        with torch.no_grad():
            actionProbs, stateValue = self.forward(state)
            actionDistribution = torch.distributions.Categorical(actionProbs)
            action = actionDistribution.sample()
            logProbability = actionDistribution.log_prob(action)

            return action, logProbability, stateValue


# =============================== Environment Wrapper ===============================
class EnvironmentWrapper:
    """环境包装器 - 修复预处理问题"""

    def __init__(self, envName: str, frameSkip: int = 4, screenSize: int = 84):
        self.env = gym.make(envName, render_mode='rgb_array')
        self.frameSkip = frameSkip
        self.screenSize = screenSize
        self.frameBuffer = deque(maxlen=2)  # 减少帧堆叠
        self.lives = 0
        self.originalLives = 5

    def reset(self) -> torch.Tensor:
        state, _ = self.env.reset()
        processedState = self.preprocessFrame(state)

        # 初始化帧缓冲区
        self.frameBuffer.clear()
        for _ in range(2):  # 只堆叠2帧
            self.frameBuffer.append(processedState)

        # 获取初始生命值
        self.lives = self.env.unwrapped.ale.lives()
        self.originalLives = self.lives

        return torch.tensor(np.stack(self.frameBuffer), dtype=torch.float32)

    def step(self, action: int) -> Tuple[torch.Tensor, float, bool, Dict]:
        totalReward = 0.0
        done = False
        lostLife = False

        for _ in range(self.frameSkip):
            nextState, reward, terminated, truncated, stepInfo = self.env.step(action)

            # 检查生命值损失
            currentLives = self.env.unwrapped.ale.lives()
            if currentLives < self.lives:
                lostLife = True
                self.lives = currentLives
                reward = -1.0  # 生命损失惩罚
            elif reward > 0:
                reward = 1.0  # 正奖励标准化
            else:
                reward = 0.0  # 无奖励

            totalReward += reward

            if terminated or truncated or lostLife:
                done = True
                break

        processedNextState = self.preprocessFrame(nextState)
        self.frameBuffer.append(processedNextState)

        nextStateTensor = torch.tensor(np.stack(self.frameBuffer), dtype=torch.float32)
        return nextStateTensor, totalReward, done, {"lost_life": lostLife}

    def preprocessFrame(self, frame: np.ndarray) -> np.ndarray:
        """改进的帧预处理"""
        if len(frame.shape) == 3:
            # 转换为灰度图并裁剪
            frame = np.mean(frame, axis=2)

        # 裁剪Breakout的无关区域
        frame = frame[34:194, :]  # 移除分数和边框

        img = Image.fromarray(frame.astype(np.uint8))
        img = img.resize((self.screenSize, self.screenSize), Image.BILINEAR)
        frame = np.array(img).astype(np.float32) / 255.0

        return frame

    @property
    def actionSpace(self):
        return self.env.action_space

    def close(self):
        self.env.close()


# =============================== A3C Agent ===============================
class A3CAgent:
    """A3C智能体 - 修复模型兼容性问题"""

    def __init__(self, stateShape: Tuple[int, int, int], numActions: int, config: A3CConfig):
        self.globalNetwork = A3CNetwork(stateShape, numActions).to(device)
        self.optimizer = optim.RMSprop(self.globalNetwork.parameters(),
                                     lr=config.learningRate,
                                     alpha=0.99,
                                     eps=1e-5)

        # 共享网络
        self.globalNetwork.share_memory()

        self.globalStep = 0
        self.episodesCompleted = 0
        self.bestReward = -float('inf')
        self.lock = Lock()
        self.config = config

        # 训练统计
        self.trainingStats = {
            'episode_rewards': [],
            'episode_lengths': [],
            'value_losses': [],
            'policy_losses': [],
            'entropies': []
        }

        logger.info("A3C Agent initialized")

    def updateModel(self, gradients: List[Tuple[torch.Tensor, torch.Tensor]]):
        """更新全局模型"""
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
            if self.config.gradientClip:
                torch.nn.utils.clip_grad_norm_(self.globalNetwork.parameters(), self.config.maxGradNorm)

            # 优化步骤
            self.optimizer.step()

    def incrementCounters(self, steps: int = 1, episodes: int = 0):
        """更新计数器"""
        with self.lock:
            self.globalStep += steps
            self.episodesCompleted += episodes

    def saveModel(self, filePath: str):
        """保存模型"""
        with self.lock:
            torch.save({
                'globalStep': self.globalStep,
                'episodesCompleted': self.episodesCompleted,
                'modelStateDict': self.globalNetwork.state_dict(),
                'optimizerStateDict': self.optimizer.state_dict(),
                'bestReward': self.bestReward,
                'trainingStats': self.trainingStats
            }, filePath)
            logger.info(f"Model saved to: {filePath}")

    def loadModel(self, filePath: str):
        """加载模型 - 添加兼容性处理"""
        with self.lock:
            if os.path.exists(filePath):
                try:
                    checkpoint = torch.load(filePath, map_location=device)

                    # 加载模型状态
                    model_state = checkpoint['modelStateDict']

                    # 处理可能的键不匹配
                    current_model_state = self.globalNetwork.state_dict()

                    # 只加载匹配的键
                    filtered_state = {k: v for k, v in model_state.items()
                                    if k in current_model_state and current_model_state[k].shape == v.shape}

                    # 加载匹配的参数
                    current_model_state.update(filtered_state)
                    self.globalNetwork.load_state_dict(current_model_state)

                    # 加载其他状态
                    self.optimizer.load_state_dict(checkpoint['optimizerStateDict'])
                    self.globalStep = checkpoint['globalStep']
                    self.episodesCompleted = checkpoint['episodesCompleted']
                    self.bestReward = checkpoint['bestReward']
                    self.trainingStats = checkpoint.get('trainingStats', self.trainingStats)

                    logger.info(f"Model successfully loaded from {filePath}")
                    logger.info(f"Loaded {len(filtered_state)}/{len(model_state)} parameters")
                    return True

                except Exception as e:
                    logger.warning(f"Error loading model {filePath}: {e}")
                    logger.info("Starting with fresh model...")
                    return False
            else:
                logger.info(f"No existing model found at {filePath}, starting fresh")
                return False

    def updateBestReward(self, reward: float):
        """更新最佳奖励"""
        with self.lock:
            if reward > self.bestReward:
                old_reward = self.bestReward
                self.bestReward = reward
                logger.info(f"New best reward: {old_reward:.1f} -> {reward:.1f}")
                return True
        return False

    def updateTrainingStats(self, statsUpdate: Dict[str, float]):
        """更新训练统计"""
        with self.lock:
            for key, value in statsUpdate.items():
                if key in self.trainingStats:
                    self.trainingStats[key].append(value)


# =============================== Worker Thread ===============================
class A3CWorker(threading.Thread):
    """A3C工作线程 - 改进训练逻辑"""

    def __init__(self, workerId: int, globalAgent: A3CAgent, config: A3CConfig):
        super().__init__()
        self.workerId = workerId
        self.globalAgent = globalAgent
        self.config = config

        # 本地网络
        self.localNetwork = A3CNetwork((2, 84, 84), 4).to(device)  # 改为2帧堆叠

        # 环境
        self.envWrapper = None

        # 工作线程统计
        self.episodeRewards = deque(maxlen=100)
        self.episodeLengths = deque(maxlen=100)

        logger.info(f"Worker thread {workerId} created")

    def run(self):
        """主工作循环"""
        try:
            # 初始化环境
            self.envWrapper = EnvironmentWrapper(
                self.config.environmentName,
                self.config.frameSkip,
                self.config.screenSize
            )
            logger.info(f"Worker thread {self.workerId} environment initialized")

            episodeCount = 0

            while self.globalAgent.globalStep < self.config.trainingTimesteps:
                # 同步网络
                self.localNetwork.load_state_dict(self.globalAgent.globalNetwork.state_dict())

                # 重置环境
                state = self.envWrapper.reset().to(device)

                episodeReward = 0
                episodeLength = 0
                done = False

                # 存储轨迹
                states, actions, rewards, logProbs, values = [], [], [], [], []

                while not done and episodeLength < self.config.maxEpisodeLength:
                    # 选择动作
                    action, logProb, value = self.localNetwork.getAction(state.unsqueeze(0))

                    # 执行动作
                    nextState, reward, done, info = self.envWrapper.step(action.item())
                    nextState = nextState.to(device)

                    # 奖励裁剪
                    if self.config.rewardClip:
                        reward = np.clip(reward, -1, 1)

                    # 存储数据
                    states.append(state)
                    actions.append(action)
                    rewards.append(reward)
                    logProbs.append(logProb)
                    values.append(value)

                    # 更新状态
                    state = nextState
                    episodeReward += reward
                    episodeLength += 1

                    # n-step更新或episode结束
                    if len(states) >= self.config.nStep or done:
                        policyLoss, valueLoss, entropy = self.updateWithTrajectory(
                            states, actions, rewards, logProbs, values, done, nextState
                        )

                        # 更新训练统计
                        if policyLoss is not None:
                            statsUpdate = {
                                'value_losses': valueLoss.item(),
                                'policy_losses': policyLoss.item(),
                                'entropies': entropy.item()
                            }
                            self.globalAgent.updateTrainingStats(statsUpdate)

                        states, actions, rewards, logProbs, values = [], [], [], [], []

                # Episode完成
                episodeCount += 1
                self.globalAgent.incrementCounters(episodes=1)

                # 更新工作线程统计
                self.episodeRewards.append(episodeReward)
                self.episodeLengths.append(episodeLength)

                averageReward = np.mean(self.episodeRewards) if self.episodeRewards else 0
                averageLength = np.mean(self.episodeLengths) if self.episodeLengths else 0

                # 检查是否是最佳奖励
                if self.globalAgent.updateBestReward(episodeReward):
                    self.globalAgent.saveModel(self.config.bestModelPath)

                # 记录episode结果
                if episodeCount % self.config.logInterval == 0:
                    logger.info(f"Worker {self.workerId} - Episode {episodeCount}: "
                               f"Reward={episodeReward:.1f}, Avg Reward={averageReward:.1f}, "
                               f"Length={episodeLength}, Steps={self.globalAgent.globalStep}")

        except Exception as e:
            logger.error(f"Worker thread {self.workerId} error: {e}")
            import traceback
            logger.error(traceback.format_exc())
        finally:
            if self.envWrapper:
                self.envWrapper.close()

    def updateWithTrajectory(self, states, actions, rewards, logProbs, values, done, nextState):
        """轨迹更新"""
        try:
            # 准备数据张量
            statesTensor = torch.stack(states).to(device)
            actionsTensor = torch.stack(actions).to(device)
            logProbsTensor = torch.stack(logProbs).to(device)
            valuesTensor = torch.stack(values).to(device)

            # 计算回报
            with torch.no_grad():
                if not done:
                    _, nextValue = self.localNetwork(nextState.unsqueeze(0))
                    returns = self.computeReturns(rewards, nextValue.item(), done)
                else:
                    returns = self.computeReturns(rewards, 0.0, done)

            returnsTensor = torch.tensor(returns, dtype=torch.float32, device=device)

            # 计算优势
            advantages = returnsTensor - valuesTensor.detach()

            # 优势归一化
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # 计算损失
            policyLoss = -(logProbsTensor * advantages).mean()
            valueLoss = F.mse_loss(valuesTensor, returnsTensor)

            # 计算熵
            actionProbs, _ = self.localNetwork(statesTensor)
            actionDistribution = torch.distributions.Categorical(actionProbs)
            entropy = actionDistribution.entropy().mean()

            # 总损失
            totalLoss = (policyLoss +
                        self.config.valueLossCoeff * valueLoss -
                        self.config.entropyCoeff * entropy)

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

            return policyLoss, valueLoss, entropy

        except Exception as e:
            logger.error(f"Worker {self.workerId} trajectory update failed: {e}")
            return None, None, None

    def computeReturns(self, rewards: List[float], lastValue: float, done: bool) -> List[float]:
        """计算n-step回报"""
        returns = []
        R = lastValue if not done else 0.0

        for r in reversed(rewards):
            R = r + self.config.discountFactor * R
            returns.insert(0, R)

        return returns


# =============================== Trainer ===============================
class A3CTrainer:
    """A3C训练器"""

    def __init__(self, config: A3CConfig):
        self.config = config
        self.globalAgent = None
        self.workers = []

        # 训练监控
        self.startTime = None

    def train(self):
        """开始训练"""
        # 创建模型目录
        os.makedirs(self.config.modelSavePath, exist_ok=True)

        # 初始化全局智能体
        stateShape = (2, self.config.screenSize, self.config.screenSize)  # 改为2帧堆叠
        numActions = 4  # Breakout动作

        self.globalAgent = A3CAgent(stateShape, numActions, self.config)

        # 尝试加载现有模型
        modelLoaded = self.globalAgent.loadModel(self.config.bestModelPath)

        # 创建工作线程
        self.workers = []
        for i in range(self.config.numProcesses):
            worker = A3CWorker(i, self.globalAgent, self.config)
            self.workers.append(worker)

        logger.info(f"Starting training with {self.config.numProcesses} worker threads")
        if modelLoaded:
            logger.info(f"Resuming from checkpoint, current global step: {self.globalAgent.globalStep}")

        # 开始计时
        self.startTime = time.time()

        # 启动工作线程
        for worker in self.workers:
            worker.start()
            time.sleep(0.5)  # 错开启动时间

        # 监控训练进度
        try:
            lastGlobalStep = self.globalAgent.globalStep
            lastLogTime = time.time()

            while self.globalAgent.globalStep < self.config.trainingTimesteps:
                time.sleep(10)  # 每10秒检查一次

                currentStep = self.globalAgent.globalStep
                currentTime = time.time()

                # 计算训练速度
                if currentStep > lastGlobalStep:
                    stepsPerSec = (currentStep - lastGlobalStep) / (currentTime - lastLogTime)
                    elapsedTime = currentTime - self.startTime
                    remainingTime = (self.config.trainingTimesteps - currentStep) / stepsPerSec if stepsPerSec > 0 else 0

                    # 记录进度
                    logger.info(f"Progress: {currentStep}/{self.config.trainingTimesteps} "
                               f"({currentStep/self.config.trainingTimesteps*100:.2f}%) | "
                               f"Speed: {stepsPerSec:.1f} steps/sec | "
                               f"Elapsed: {self.formatTime(elapsedTime)} | "
                               f"ETA: {self.formatTime(remainingTime)} | "
                               f"Episodes: {self.globalAgent.episodesCompleted} | "
                               f"Best Reward: {self.globalAgent.bestReward:.1f}")

                    lastGlobalStep = currentStep
                    lastLogTime = currentTime

                    # 保存检查点
                    if currentStep % self.config.checkpointInterval == 0:
                        checkpointPath = os.path.join(
                            self.config.modelSavePath,
                            f"checkpoint_step_{currentStep}.pth"
                        )
                        self.globalAgent.saveModel(checkpointPath)
                        logger.info(f"Checkpoint saved at step {currentStep}")

                else:
                    logger.warning("No training progress detected...")

                    # 检查线程状态
                    aliveCount = sum(1 for w in self.workers if w.is_alive())
                    if aliveCount == 0:
                        logger.error("All worker threads have stopped!")
                        break

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        except Exception as e:
            logger.error(f"Training error: {e}")
            import traceback
            logger.error(traceback.format_exc())
        finally:
            # 保存最终模型
            finalModelPath = os.path.join(self.config.modelSavePath, "final_model.pth")
            self.globalAgent.saveModel(finalModelPath)

            # 等待线程结束
            for worker in self.workers:
                if worker.is_alive():
                    worker.join(timeout=5.0)

            totalTime = time.time() - self.startTime
            logger.info(f"Training completed! Final global step: {self.globalAgent.globalStep}")
            logger.info(f"Best reward achieved: {self.globalAgent.bestReward:.1f}")
            logger.info(f"Total training time: {self.formatTime(totalTime)}")

    def formatTime(self, seconds: float) -> str:
        """格式化时间"""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            return f"{seconds/3600:.1f}h"


# =============================== Main Function ===============================
def main():
    """主训练函数"""
    try:
        # 配置
        config = A3CConfig()

        print("A3C Reinforcement Learning Training System")
        print("=" * 50)
        print(f"Environment: {config.environmentName}")
        print(f"Training Steps: {config.trainingTimesteps}")
        print(f"Number of Workers: {config.numProcesses}")
        print(f"Device: {device}")
        print("=" * 50)

        # 删除旧的模型文件以避免兼容性问题
        if os.path.exists(config.bestModelPath):
            os.remove(config.bestModelPath)
            logger.info("Removed old model file to avoid compatibility issues")

        # 开始训练
        logger.info("Starting A3C training!")

        trainer = A3CTrainer(config)
        trainer.train()

        logger.info("Training completed successfully!")

    except Exception as e:
        logger.error(f"Error during training execution: {e}")
        import traceback
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()
