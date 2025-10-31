"""
A3C完整训练版本 - 包含模型保存和完整功能
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
    """A3C训练配置参数"""
    def __init__(self):
        self.environmentName = "BreakoutNoFrameskip-v4"
        self.learningRate = 0.0007
        self.discountFactor = 0.99
        self.entropyCoeff = 0.01
        self.valueLossCoeff = 0.5
        self.maxGradNorm = 40.0
        self.nStep = 5
        self.numProcesses = 2
        self.trainingTimesteps = 100000
        self.frameSkip = 4
        self.screenSize = 84
        self.useLSTM = False
        self.logInterval = 1
        self.saveInterval = 20
        self.maxEpisodeLength = 1000
        self.modelSavePath = "./A3CModels"
        self.bestModelPath = "./A3CModels/best_model.pth"


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


# =============================== 神经网络结构 ===============================
class A3CNetwork(nn.Module):
    """A3C神经网络"""

    def __init__(self, inputShape: Tuple[int, int, int], numActions: int):
        super().__init__()

        self.convLayers = nn.Sequential(
            nn.Conv2d(inputShape[0], 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.policyHead = nn.Linear(64, numActions)
        self.valueHead = nn.Linear(64, 1)

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


# =============================== 环境包装器 ===============================
class EnvironmentWrapper:
    """环境包装器"""

    def __init__(self, envName: str, frameSkip: int = 4, screenSize: int = 84):
        self.env = gym.make(envName, render_mode='rgb_array')
        self.frameSkip = frameSkip
        self.screenSize = screenSize
        self.frameBuffer = deque(maxlen=4)

    def reset(self) -> torch.Tensor:
        state, _ = self.env.reset()
        processedState = self.preprocessFrame(state)

        # 填充初始帧缓冲区
        self.frameBuffer.extend([processedState] * 4)

        return torch.tensor(np.stack(self.frameBuffer), dtype=torch.float32)

    def step(self, action: int) -> Tuple[torch.Tensor, float, bool, Dict]:
        totalReward = 0.0
        done = False
        info = {}

        for _ in range(self.frameSkip):
            nextState, reward, terminated, truncated, stepInfo = self.env.step(action)
            totalReward += reward
            info.update(stepInfo)
            if terminated or truncated:
                done = True
                break

        processedNextState = self.preprocessFrame(nextState)
        self.frameBuffer.append(processedNextState)

        nextStateTensor = torch.tensor(np.stack(self.frameBuffer), dtype=torch.float32)
        return nextStateTensor, totalReward, done, info

    def preprocessFrame(self, frame: np.ndarray) -> np.ndarray:
        """预处理帧"""
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
class A3CAgent:
    """A3C智能体"""

    def __init__(self, stateShape: Tuple[int, int, int], numActions: int, config: A3CConfig):
        self.globalNetwork = A3CNetwork(stateShape, numActions).to(device)
        self.optimizer = optim.Adam(self.globalNetwork.parameters(), lr=config.learningRate)

        # 共享网络
        self.globalNetwork.share_memory()

        self.globalStep = 0
        self.episodesCompleted = 0
        self.bestReward = -float('inf')
        self.lock = Lock()
        self.config = config

        logger.info("A3C智能体初始化完成")

    def updateModel(self, gradients: List[Tuple[torch.Tensor, torch.Tensor]]):
        """更新模型"""
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

    def saveModel(self, filePath: str):
        """保存模型"""
        with self.lock:
            torch.save({
                'globalStep': self.globalStep,
                'episodesCompleted': self.episodesCompleted,
                'modelStateDict': self.globalNetwork.state_dict(),
                'optimizerStateDict': self.optimizer.state_dict(),
                'bestReward': self.bestReward
            }, filePath)
            logger.info(f"模型已保存到: {filePath}")

    def loadModel(self, filePath: str):
        """加载模型"""
        with self.lock:
            if os.path.exists(filePath):
                checkpoint = torch.load(filePath)
                self.globalNetwork.load_state_dict(checkpoint['modelStateDict'])
                self.optimizer.load_state_dict(checkpoint['optimizerStateDict'])
                self.globalStep = checkpoint['globalStep']
                self.episodesCompleted = checkpoint['episodesCompleted']
                self.bestReward = checkpoint['bestReward']
                logger.info(f"模型已从 {filePath} 加载")
                return True
            else:
                logger.warning(f"模型文件 {filePath} 不存在")
                return False

    def updateBestReward(self, reward: float):
        """更新最佳奖励"""
        with self.lock:
            if reward > self.bestReward:
                self.bestReward = reward
                return True
        return False


# =============================== 工作线程 ===============================
class A3CWorker(threading.Thread):
    """A3C工作线程"""

    def __init__(self, workerId: int, globalAgent: A3CAgent, config: A3CConfig):
        super().__init__()
        self.workerId = workerId
        self.globalAgent = globalAgent
        self.config = config

        # 本地网络
        self.localNetwork = A3CNetwork((4, 84, 84), 4).to(device)

        # 环境
        self.envWrapper = None

        logger.info(f"工作线程 {workerId} 创建完成")

    def run(self):
        """工作线程主循环"""
        try:
            # 初始化环境
            self.envWrapper = EnvironmentWrapper(
                self.config.environmentName,
                self.config.frameSkip,
                self.config.screenSize
            )
            logger.info(f"工作线程 {self.workerId} 环境初始化完成")

            episodeCount = 0
            episodeRewards = deque(maxlen=100)  # 记录最近100个episode的奖励

            while self.globalAgent.globalStep < self.config.trainingTimesteps:
                # 同步网络
                self.localNetwork.load_state_dict(self.globalAgent.globalNetwork.state_dict())

                # 重置环境
                state = self.envWrapper.reset().to(device)

                episodeReward = 0
                episodeLength = 0
                done = False

                # 存储轨迹
                states, actions, rewards = [], [], []

                while not done and episodeLength < self.config.maxEpisodeLength:
                    # 选择动作
                    action, logProb, value = self.localNetwork.getAction(state.unsqueeze(0))

                    # 执行动作
                    nextState, reward, done, _ = self.envWrapper.step(action.item())
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

                # 记录奖励
                episodeRewards.append(episodeReward)
                averageReward = np.mean(episodeRewards) if episodeRewards else 0

                # 检查是否是最佳奖励
                if self.globalAgent.updateBestReward(episodeReward):
                    self.globalAgent.saveModel(self.config.bestModelPath)
                    logger.info(f"工作线程 {self.workerId} - 新的最佳奖励: {episodeReward:.1f}")

                logger.info(f"工作线程 {self.workerId} - Episode {episodeCount}: "
                           f"奖励={episodeReward:.1f}, 平均奖励={averageReward:.1f}, 长度={episodeLength}")

        except Exception as e:
            logger.error(f"工作线程 {self.workerId} 错误: {e}")
            import traceback
            logger.error(traceback.format_exc())
        finally:
            if self.envWrapper:
                self.envWrapper.close()

    def updateWithTrajectory(self, states, actions, rewards, done, nextState):
        """轨迹更新"""
        try:
            # 准备数据
            statesTensor = torch.stack(states).to(device)
            actionsTensor = torch.stack(actions).to(device)

            # 重新计算log probabilities和values（需要梯度）
            actionProbs, stateValues = self.localNetwork(statesTensor)
            actionDistribution = torch.distributions.Categorical(actionProbs)
            logProbabilities = actionDistribution.log_prob(actionsTensor)

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
            policyLoss = -(logProbabilities * advantages).mean()
            valueLoss = F.mse_loss(stateValues, returnsTensor)

            # 计算熵
            entropy = actionDistribution.entropy().mean()

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
class A3CTrainer:
    """A3C训练器"""

    def __init__(self, config: A3CConfig):
        self.config = config
        self.globalAgent = None
        self.workers = []

    def train(self):
        """开始训练"""
        # 创建模型保存目录
        os.makedirs(self.config.modelSavePath, exist_ok=True)

        # 初始化全局智能体
        stateShape = (4, self.config.screenSize, self.config.screenSize)
        numActions = 4  # Breakout的动作数量

        self.globalAgent = A3CAgent(stateShape, numActions, self.config)

        # 尝试加载现有模型
        modelLoaded = self.globalAgent.loadModel(self.config.bestModelPath)

        # 创建工作线程
        self.workers = []
        for i in range(self.config.numProcesses):
            worker = A3CWorker(i, self.globalAgent, self.config)
            self.workers.append(worker)

        logger.info(f"开始训练, 使用 {self.config.numProcesses} 个工作线程")
        if modelLoaded:
            logger.info(f"从检查点继续训练, 当前全局步数: {self.globalAgent.globalStep}")

        # 启动工作线程
        for worker in self.workers:
            worker.start()
            logger.info(f"已启动工作线程 {worker.workerId}")
            time.sleep(0.5)  # 稍微错开启动时间

        # 监控训练进度
        try:
            lastGlobalStep = 0
            lastCheckTime = time.time()
            lastSaveTime = time.time()

            while self.globalAgent.globalStep < self.config.trainingTimesteps:
                time.sleep(2)  # 每2秒检查一次

                currentStep = self.globalAgent.globalStep
                currentTime = time.time()

                if currentStep > lastGlobalStep:
                    stepsPerSec = (currentStep - lastGlobalStep) / (currentTime - lastCheckTime)
                    logger.info(f"训练进度 - 全局步数: {currentStep}/{self.config.trainingTimesteps} "
                               f"({currentStep/self.config.trainingTimesteps*100:.1f}%), "
                               f"速度: {stepsPerSec:.1f} 步/秒")

                    lastGlobalStep = currentStep
                    lastCheckTime = currentTime

                    # 定期保存模型
                    if currentTime - lastSaveTime > 300:  # 每5分钟保存一次
                        modelPath = os.path.join(
                            self.config.modelSavePath,
                            f"checkpoint_step_{currentStep}.pth"
                        )
                        self.globalAgent.saveModel(modelPath)
                        lastSaveTime = currentTime
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
            # 保存最终模型
            finalModelPath = os.path.join(self.config.modelSavePath, "final_model.pth")
            self.globalAgent.saveModel(finalModelPath)

            # 等待线程结束
            for worker in self.workers:
                if worker.is_alive():
                    worker.join(timeout=2.0)

            logger.info(f"训练结束! 最终全局步数: {self.globalAgent.globalStep}")
            logger.info(f"最佳奖励: {self.globalAgent.bestReward:.1f}")


# =============================== 模型测试器 ===============================
class ModelTester:
    """模型测试器"""

    def __init__(self, config: A3CConfig):
        self.config = config
        self.envWrapper = EnvironmentWrapper(
            config.environmentName,
            config.frameSkip,
            config.screenSize
        )

        stateShape = (4, config.screenSize, config.screenSize)
        numActions = 4
        self.model = A3CNetwork(stateShape, numActions).to(device)

    def loadModel(self, modelPath: str):
        """加载模型"""
        checkpoint = torch.load(modelPath, map_location=device)
        self.model.load_state_dict(checkpoint['modelStateDict'])
        logger.info(f"测试模型已加载: {modelPath}")

    def test(self, numEpisodes: int = 10, render: bool = True):
        """测试模型"""
        totalRewards = []

        for episode in range(numEpisodes):
            state = self.envWrapper.reset().to(device)
            episodeReward = 0
            done = False

            while not done:
                if render:
                    self.envWrapper.env.render()

                action, _, _ = self.model.getAction(state.unsqueeze(0))
                nextState, reward, done, _ = self.envWrapper.step(action.item())
                nextState = nextState.to(device)

                state = nextState
                episodeReward += reward

            totalRewards.append(episodeReward)
            logger.info(f"测试Episode {episode + 1}: 奖励 = {episodeReward:.1f}")

        averageReward = np.mean(totalRewards)
        logger.info(f"平均测试奖励: {averageReward:.1f}")

        return averageReward


# =============================== 主函数 ===============================
def main():
    """主函数"""
    try:
        # 配置
        config = A3CConfig()

        print("A3C强化学习训练系统")
        print("1. 训练模型")
        print("2. 测试模型")
        choice = input("请选择操作 (1/2): ").strip()

        if choice == "1":
            logger.info("开始A3C训练!")

            # 训练
            trainer = A3CTrainer(config)
            trainer.train()

            logger.info("训练完成!")

        elif choice == "2":
            logger.info("开始模型测试!")

            # 测试
            tester = ModelTester(config)

            modelPath = input("请输入模型路径 (留空使用最佳模型): ").strip()
            if not modelPath:
                modelPath = config.bestModelPath

            if os.path.exists(modelPath):
                tester.loadModel(modelPath)
                numEpisodes = int(input("请输入测试episode数量 (默认10): ") or "10")
                tester.test(numEpisodes=numEpisodes, render=True)
            else:
                logger.error(f"模型文件不存在: {modelPath}")

        else:
            logger.error("无效选择")

    except Exception as e:
        logger.error(f"程序执行过程中发生错误: {e}")
        import traceback
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()
