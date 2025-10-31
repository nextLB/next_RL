"""
    A3C (Asynchronous Advantage Actor-Critic) 算法实现
    基于论文 "Asynchronous Methods for Deep Reinforcement Learning"
    针对 PongNoFrameskip-v4 环境进行训练
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import numpy as np
import gymnasium as gym
from collections import deque
import matplotlib.pyplot as plt
from PIL import Image
import logging
import os
import time
from typing import Tuple, List, Dict, Any
from dataclasses import dataclass
import json
from datetime import datetime


# 配置类
@dataclass
class A3CConfig:
    """A3C训练配置参数"""
    environmentName: str = "PongNoFrameskip-v4"
    learningRate: float = 0.0007
    discountFactor: float = 0.99
    numProcesses: int = 16
    tMax: int = 5
    entropyCoefficient: float = 0.01
    valueLossCoefficient: float = 0.5
    maxGradientNorm: float = 50.0
    frameSkip: int = 4
    screenSize: int = 84
    trainingSteps: int = 80000000
    saveModelFrequency: int = 1000000
    logInterval: int = 10000
    useLSTM: bool = False
    lstmSize: int = 256
    rmsPropAlpha: float = 0.99
    rmsPropEpsilon: float = 0.1
    useSharedRMSProp: bool = True


# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./log/a3c_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class AtariEnvironmentPreprocessor:
    """Atari环境预处理包装器"""

    def __init__(self, environment, frameSkip: int = 4, screenSize: int = 84):
        self.environment = environment
        self.frameSkip = frameSkip
        self.screenSize = screenSize
        self.frameBuffer = deque(maxlen=4)

    def reset(self) -> Tuple[np.ndarray, dict]:
        """重置环境并返回预处理后的初始状态"""
        state, info = self.environment.reset()
        processedState = self._preprocessFrame(state)

        # 用相同的帧填充初始缓冲区
        self.frameBuffer.extend([processedState] * 4)

        stateArray = np.stack(self.frameBuffer)
        return stateArray, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """执行动作并返回预处理后的结果"""
        totalReward = 0.0
        terminated = False
        truncated = False
        info = {}

        # 使用帧跳过提高效率
        for _ in range(self.frameSkip):
            nextState, reward, terminated, truncated, stepInfo = self.environment.step(action)
            totalReward += reward
            info.update(stepInfo)

            if terminated or truncated:
                break

        done = terminated or truncated
        processedNextState = self._preprocessFrame(nextState)
        self.frameBuffer.append(processedNextState)

        nextStateArray = np.stack(self.frameBuffer)
        return nextStateArray, totalReward, done, info

    def _preprocessFrame(self, frame: np.ndarray) -> np.ndarray:
        """预处理帧：灰度化、调整大小、归一化"""
        # 转换为灰度图
        if len(frame.shape) == 3:
            frame = np.mean(frame, axis=2)

        # 调整大小
        img = Image.fromarray(frame.astype(np.uint8))
        img = img.resize((self.screenSize, self.screenSize), Image.BILINEAR)
        frame = np.array(img)

        # 归一化到 [0, 1]
        frame = frame.astype(np.float32) / 255.0

        return frame

    @property
    def actionSpace(self):
        return self.environment.action_space

    @property
    def observationSpace(self):
        return self.environment.observation_space

    def close(self) -> None:
        """关闭环境"""
        self.environment.close()


class ActorCriticNetwork(nn.Module):
    """A3C Actor-Critic 网络"""

    def __init__(self, inputShape: Tuple[int, int, int], numActions: int, useLSTM: bool = False, lstmSize: int = 256):
        super().__init__()
        self.useLSTM = useLSTM
        self.numActions = numActions

        # 卷积层
        self.conv1 = nn.Conv2d(inputShape[0], 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)

        # 计算卷积层输出尺寸
        convOutputSize = self._getConvOutputSize(inputShape)

        # 全连接层
        self.fc = nn.Linear(convOutputSize, 256)

        if useLSTM:
            self.lstm = nn.LSTM(256, lstmSize, batch_first=True)
            policyInputSize = lstmSize
            valueInputSize = lstmSize
        else:
            policyInputSize = 256
            valueInputSize = 256

        # 策略头 (Actor)
        self.policyHead = nn.Linear(policyInputSize, numActions)

        # 价值头 (Critic)
        self.valueHead = nn.Linear(valueInputSize, 1)

        # 初始化权重
        self._initializeWeights()

    def _getConvOutputSize(self, inputShape: Tuple[int, int, int]) -> int:
        """计算卷积层输出尺寸"""
        with torch.no_grad():
            x = torch.zeros(1, *inputShape)
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            return x.view(1, -1).size(1)

    def _initializeWeights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)

    def forward(self, x: torch.Tensor, hiddenState: Tuple[torch.Tensor, torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """前向传播"""
        batchSize = x.size(0)
        sequenceLength = x.size(1) if len(x.shape) > 4 else 1

        # 重塑输入以处理序列
        if len(x.shape) == 5:  # (batch, sequence, channels, height, width)
            x = x.view(batchSize * sequenceLength, *x.shape[2:])

        # 卷积层
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))

        # 重塑回序列格式
        if len(x.shape) == 2 and sequenceLength > 1:
            x = x.view(batchSize, sequenceLength, -1)

        # LSTM层
        if self.useLSTM:
            if hiddenState is None:
                hiddenState = (torch.zeros(1, batchSize, self.lstm.hidden_size).to(x.device),
                             torch.zeros(1, batchSize, self.lstm.hidden_size).to(x.device))

            x, newHiddenState = self.lstm(x, hiddenState)
        else:
            newHiddenState = hiddenState

        # 策略和价值输出
        policyLogits = self.policyHead(x)
        value = self.valueHead(x)

        # 如果处理的是序列，取最后一个时间步
        if len(policyLogits.shape) == 3:  # (batch, sequence, actions)
            policyLogits = policyLogits[:, -1, :]
            value = value[:, -1, :]

        return policyLogits, value, newHiddenState

    def getAction(self, state: torch.Tensor, hiddenState: Tuple[torch.Tensor, torch.Tensor] = None) -> Tuple[int, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """根据状态选择动作"""
        with torch.no_grad():
            policyLogits, value, newHiddenState = self.forward(state.unsqueeze(0), hiddenState)
            policy = F.softmax(policyLogits, dim=-1)
            action = policy.multinomial(1).item()
            return action, policyLogits, newHiddenState


class SharedAdam(torch.optim.Adam):
    """共享的Adam优化器，用于多进程训练"""

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

        # 共享状态
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = torch.zeros(1, dtype=torch.float).share_memory_()
                state['exp_avg'] = torch.zeros_like(p.data).share_memory_()
                state['exp_avg_sq'] = torch.zeros_like(p.data).share_memory_()


class A3CAgent:
    """A3C智能体"""

    def __init__(self, sharedModel: ActorCriticNetwork, optimizer: torch.optim.Optimizer,
                 config: A3CConfig, processId: int = 0):
        self.sharedModel = sharedModel
        self.optimizer = optimizer
        self.config = config
        self.processId = processId

        # 本地模型
        self.localModel = ActorCriticNetwork(
            (4, config.screenSize, config.screenSize),
            self._getNumActions(),
            config.useLSTM,
            config.lstmSize
        ).to(device)

        # 环境
        self.environment = self._createEnvironment()
        self.preprocessedEnvironment = AtariEnvironmentPreprocessor(
            self.environment,
            frameSkip=config.frameSkip,
            screenSize=config.screenSize
        )

        # 训练状态
        self.currentStep = 0
        self.episodeReward = 0
        self.episodeLength = 0
        self.totalSteps = 0

        logger.info(f"进程 {processId} A3C智能体初始化完成")

    def _getNumActions(self) -> int:
        """获取动作空间大小"""
        env = gym.make(self.config.environmentName)
        numActions = env.action_space.n
        env.close()
        return numActions

    def _createEnvironment(self):
        """创建环境"""
        return gym.make(self.config.environmentName, render_mode='rgb_array')

    def synchronizeModels(self):
        """同步全局模型参数到本地模型"""
        self.localModel.load_state_dict(self.sharedModel.state_dict())

    def computeLoss(self, rollouts: List[Tuple]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """计算A3C损失"""
        states, actions, rewards, values, hiddenStates, dones = zip(*rollouts)

        # 转换为张量
        states = torch.stack(states).to(device)
        actions = torch.tensor(actions, dtype=torch.long).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(device)
        values = torch.stack(values).squeeze().to(device)
        dones = torch.tensor(dones, dtype=torch.float).to(device)

        # 计算n步回报和优势
        returns = self._computeReturns(rewards, values[-1], dones)
        advantages = returns - values[:-1]

        # 获取策略对数概率
        policyLogits, currentValues, _ = self.localModel(states[:-1])
        logProbs = F.log_softmax(policyLogits, dim=-1)
        actionLogProbs = logProbs.gather(1, actions.unsqueeze(1)).squeeze()

        # 策略损失 (Actor)
        policyLoss = -(actionLogProbs * advantages.detach()).mean()

        # 价值损失 (Critic)
        valueLoss = advantages.pow(2).mean()

        # 熵正则化
        entropy = -(logProbs * torch.exp(logProbs)).sum(-1).mean()
        entropyLoss = -self.config.entropyCoefficient * entropy

        totalLoss = policyLoss + self.config.valueLossCoefficient * valueLoss + entropyLoss

        return totalLoss, policyLoss, valueLoss, entropyLoss

    def _computeReturns(self, rewards: torch.Tensor, lastValue: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
        """计算n步回报"""
        returns = torch.zeros_like(rewards)
        R = lastValue

        for t in reversed(range(len(rewards))):
            R = rewards[t] + self.config.discountFactor * R * (1 - dones[t])
            returns[t] = R

        return returns

    def train(self, globalCounter: mp.Value):
        """训练循环"""
        self.synchronizeModels()

        state, _ = self.preprocessedEnvironment.reset()
        state = torch.FloatTensor(state).to(device)

        hiddenState = None
        rollouts = []
        episodeRewards = []

        while globalCounter.value < self.config.trainingSteps:
            # 收集经验
            for _ in range(self.config.tMax):
                self.currentStep += 1
                self.totalSteps += 1

                # 选择动作
                action, policyLogits, nextHiddenState = self.localModel.getAction(state, hiddenState)

                # 执行动作
                nextState, reward, done, _ = self.preprocessedEnvironment.step(action)
                nextState = torch.FloatTensor(nextState).to(device)

                # 存储经验
                with torch.no_grad():
                    _, value, _ = self.localModel(state.unsqueeze(0), hiddenState)

                rollouts.append((state, action, reward, value, hiddenState, done))

                # 更新状态
                state = nextState
                hiddenState = nextHiddenState
                self.episodeReward += reward
                self.episodeLength += 1

                # 如果episode结束
                if done:
                    episodeRewards.append(self.episodeReward)

                    if len(episodeRewards) % 10 == 0:
                        avgReward = np.mean(episodeRewards[-10:])
                        logger.info(f"进程 {self.processId} - 平均奖励 (最近10回合): {avgReward:.2f}, 总步数: {globalCounter.value}")

                    # 重置环境
                    state, _ = self.preprocessedEnvironment.reset()
                    state = torch.FloatTensor(state).to(device)
                    hiddenState = None
                    self.episodeReward = 0
                    self.episodeLength = 0

            # 计算最终状态的价值
            with torch.no_grad():
                _, lastValue, _ = self.localModel(state.unsqueeze(0), hiddenState)

            # 添加最终价值到rollouts
            rollouts.append((state, 0, 0, lastValue, hiddenState, False))

            # 计算损失并更新
            loss, policyLoss, valueLoss, entropyLoss = self.computeLoss(rollouts)

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.localModel.parameters(), self.config.maxGradientNorm)

            # 将梯度添加到共享模型
            for localParam, sharedParam in zip(self.localModel.parameters(), self.sharedModel.parameters()):
                if sharedParam.grad is None:
                    sharedParam.grad = localParam.grad
                else:
                    sharedParam.grad += localParam.grad

            # 更新全局模型
            self.optimizer.step()

            # 同步模型
            self.synchronizeModels()

            # 清空rollouts
            rollouts = []

            # 更新全局计数器
            with globalCounter.get_lock():
                globalCounter.value += self.currentStep
                currentGlobalStep = globalCounter.value

            self.currentStep = 0

            # 保存模型
            if currentGlobalStep % self.config.saveModelFrequency == 0 and self.processId == 0:
                self._saveModel(currentGlobalStep)

            # 日志记录
            if currentGlobalStep % self.config.logInterval == 0 and self.processId == 0:
                logger.info(f"全局步数: {currentGlobalStep}, 损失: {loss.item():.4f}")

        self.preprocessedEnvironment.close()

    def _saveModel(self, step: int):
        """保存模型"""
        modelPath = f"./A3C_models/a3c_model_{self.config.environmentName.replace('/', '_')}_step_{step}.pth"
        os.makedirs(os.path.dirname(modelPath), exist_ok=True)

        torch.save({
            'model_state_dict': self.sharedModel.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step': step,
            'config': self.config
        }, modelPath)

        logger.info(f"模型已保存到: {modelPath}")


def trainA3C(config: A3CConfig):
    """训练A3C算法"""
    logger.info("开始A3C训练")

    # 创建共享模型和优化器
    env = gym.make(config.environmentName)
    numActions = env.action_space.n
    env.close()

    sharedModel = ActorCriticNetwork(
        (4, config.screenSize, config.screenSize),
        numActions,
        config.useLSTM,
        config.lstmSize
    ).to(device)

    sharedModel.share_memory()

    optimizer = SharedAdam(sharedModel.parameters(), lr=config.learningRate)

    # 全局计数器
    globalCounter = mp.Value('i', 0)

    # 创建并启动进程
    processes = []

    def worker(processId: int, globalCounter: mp.Value):
        try:
            agent = A3CAgent(sharedModel, optimizer, config, processId)
            agent.train(globalCounter)
        except Exception as e:
            logger.error(f"进程 {processId} 发生错误: {e}")

    for i in range(config.numProcesses):
        process = mp.Process(target=worker, args=(i, globalCounter))
        process.start()
        processes.append(process)
        time.sleep(0.1)  # 避免同时创建过多进程

    # 等待所有进程完成
    for process in processes:
        process.join()

    logger.info("A3C训练完成")


def testTrainedModel(config: A3CConfig, modelPath: str, numEpisodes: int = 10):
    """测试训练好的模型"""
    logger.info(f"开始测试模型: {modelPath}")

    # 加载模型
    checkpoint = torch.load(modelPath)
    trainedModel = ActorCriticNetwork(
        (4, config.screenSize, config.screenSize),
        config._getNumActions(),
        config.useLSTM,
        config.lstmSize
    ).to(device)

    trainedModel.load_state_dict(checkpoint['model_state_dict'])
    trainedModel.eval()

    # 创建环境
    environment = gym.make(config.environmentName, render_mode='human')
    preprocessedEnvironment = AtariEnvironmentPreprocessor(
        environment,
        frameSkip=config.frameSkip,
        screenSize=config.screenSize
    )

    episodeRewards = []

    for episode in range(numEpisodes):
        state, _ = preprocessedEnvironment.reset()
        state = torch.FloatTensor(state).to(device)
        hiddenState = None

        episodeReward = 0
        done = False

        while not done:
            # 选择动作
            with torch.no_grad():
                action, _, nextHiddenState = trainedModel.getAction(state, hiddenState)

            # 执行动作
            nextState, reward, done, _ = preprocessedEnvironment.step(action)
            nextState = torch.FloatTensor(nextState).to(device)

            # 更新状态
            state = nextState
            hiddenState = nextHiddenState
            episodeReward += reward

            # 稍微延迟以便观察
            time.sleep(0.01)

        episodeRewards.append(episodeReward)
        logger.info(f"测试回合 {episode + 1}: 奖励 = {episodeReward}")

    avgReward = np.mean(episodeRewards)
    logger.info(f"平均测试奖励: {avgReward:.2f}")

    preprocessedEnvironment.close()

    return avgReward


def plotTrainingResults(trainingLogs: List[Dict]):
    """绘制训练结果"""
    if not trainingLogs:
        logger.warning("没有训练日志可绘制")
        return

    steps = [log['step'] for log in trainingLogs]
    rewards = [log['reward'] for log in trainingLogs]

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(steps, rewards)
    plt.title('A3C Training - Average Reward')
    plt.xlabel('Training Steps')
    plt.ylabel('Average Reward')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.hist(rewards, bins=20, alpha=0.7)
    plt.title('A3C Training - Reward Distribution')
    plt.xlabel('Reward')
    plt.ylabel('Frequency')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('a3c_training_results.png', dpi=150, bbox_inches='tight')
    plt.show()


def main():
    """主函数"""
    try:
        # 训练配置
        config = A3CConfig()

        logger.info("开始A3C多进程训练")
        logger.info(f"配置参数: {config}")

        # 开始训练
        trainA3C(config)

        logger.info("A3C训练完成!")

    except KeyboardInterrupt:
        logger.info("训练被用户中断")
    except Exception as e:
        logger.error(f"训练过程中发生错误: {e}")
    finally:
        logger.info("程序执行完毕")


if __name__ == "__main__":
    # 注意：在Windows上运行多进程时，需要将代码放在if __name__ == "__main__"中
    main()
