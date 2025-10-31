"""
    A3C (Asynchronous Advantage Actor-Critic) 算法实现 - 形状修复版本
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
import queue

# 修复多进程启动问题
if mp.get_start_method() != 'spawn':
    mp.set_start_method('spawn', force=True)

# 配置类
@dataclass
class A3CConfig:
    """A3C训练配置参数"""
    environmentName: str = "PongNoFrameskip-v4"
    learningRate: float = 0.0007
    discountFactor: float = 0.99
    numProcesses: int = 2
    tMax: int = 5
    entropyCoefficient: float = 0.01
    valueLossCoefficient: float = 0.5
    maxGradientNorm: float = 50.0
    frameSkip: int = 4
    screenSize: int = 84
    trainingSteps: int = 100000
    saveModelFrequency: int = 10000
    logInterval: int = 1000

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

    def close(self) -> None:
        """关闭环境"""
        self.environment.close()

class ActorCriticNetwork(nn.Module):
    """A3C Actor-Critic 网络"""

    def __init__(self, inputChannels: int, numActions: int):
        super().__init__()
        self.numActions = numActions

        # 卷积层
        self.conv1 = nn.Conv2d(inputChannels, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)

        # 计算卷积层输出尺寸
        convOutputSize = self._getConvOutputSize(inputChannels)

        # 全连接层
        self.fc = nn.Linear(convOutputSize, 256)

        # 策略头 (Actor)
        self.policyHead = nn.Linear(256, numActions)

        # 价值头 (Critic)
        self.valueHead = nn.Linear(256, 1)

        # 初始化权重
        self._initializeWeights()

    def _getConvOutputSize(self, inputChannels: int) -> int:
        """计算卷积层输出尺寸"""
        with torch.no_grad():
            x = torch.zeros(1, inputChannels, 84, 84)
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            return x.view(1, -1).size(1)

    def _initializeWeights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        # 卷积层
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))

        # 策略和价值输出
        policyLogits = self.policyHead(x)
        value = self.valueHead(x)

        return policyLogits, value

    def getAction(self, state: torch.Tensor) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """根据状态选择动作"""
        with torch.no_grad():
            policyLogits, value = self.forward(state.unsqueeze(0))
            policy = F.softmax(policyLogits, dim=-1)
            action = policy.multinomial(1).item()
            return action, policyLogits, value

def createEnvironment(config: A3CConfig):
    """创建环境函数"""
    env = gym.make(config.environmentName)
    preprocessedEnv = AtariEnvironmentPreprocessor(
        env,
        frameSkip=config.frameSkip,
        screenSize=config.screenSize
    )
    return preprocessedEnv

def getNumActions(config: A3CConfig) -> int:
    """获取动作空间大小"""
    env = gym.make(config.environmentName)
    numActions = env.action_space.n
    env.close()
    return numActions

def computeA3CLoss(model: nn.Module, states: List[torch.Tensor], actions: List[int],
                   rewards: List[float], done: bool, config: A3CConfig) -> torch.Tensor:
    """计算A3C损失 - 修复形状问题"""
    # 确保所有状态都需要梯度
    statesTensor = torch.stack(states)

    # 重新计算策略和价值（确保有梯度）
    policyLogits, values = model(statesTensor)

    # 计算n步回报
    R = 0.0
    if not done:
        with torch.no_grad():
            # 计算最后一个状态的价值
            lastState = states[-1].unsqueeze(0)
            _, lastValue = model(lastState)
            R = lastValue.item()

    returns = []
    for r in reversed(rewards):
        R = r + config.discountFactor * R
        returns.insert(0, R)

    returnsTensor = torch.tensor(returns, dtype=torch.float32).to(device)

    # 修复形状问题 - 确保values和returnsTensor形状一致
    values_flat = values.squeeze()
    if values_flat.dim() == 0:  # 如果是标量
        values_flat = values_flat.unsqueeze(0)

    # 计算优势函数
    advantages = returnsTensor - values_flat.detach()

    # 策略损失
    logProbs = F.log_softmax(policyLogits, dim=-1)
    actionsTensor = torch.tensor(actions, dtype=torch.long).to(device)
    actionLogProbs = logProbs.gather(1, actionsTensor.unsqueeze(1)).squeeze()

    # 确保actionLogProbs和advantages形状一致
    if actionLogProbs.dim() == 0:
        actionLogProbs = actionLogProbs.unsqueeze(0)

    policyLoss = -(actionLogProbs * advantages).mean()

    # 价值损失 - 修复形状问题
    # 确保values_flat和returnsTensor形状完全一致
    if values_flat.shape != returnsTensor.shape:
        # 如果形状不匹配，调整values_flat的形状
        if values_flat.numel() == 1 and returnsTensor.numel() > 1:
            values_flat = values_flat.expand_as(returnsTensor)
        elif values_flat.numel() > 1 and returnsTensor.numel() == 1:
            returnsTensor = returnsTensor.expand_as(values_flat)

    valueLoss = F.mse_loss(values_flat, returnsTensor)

    # 熵正则化
    entropy = -(logProbs * torch.exp(logProbs)).sum(-1).mean()
    entropyLoss = -config.entropyCoefficient * entropy

    totalLoss = policyLoss + config.valueLossCoefficient * valueLoss + entropyLoss

    return totalLoss

def worker(processId: int, sharedModel: nn.Module, optimizer: torch.optim.Optimizer,
           config: A3CConfig, globalCounter: mp.Value, trainingQueue: mp.Queue):
    """工作进程函数"""
    try:
        logger.info(f"进程 {processId} 启动")

        # 设置随机种子
        torch.manual_seed(processId)
        np.random.seed(processId)

        # 创建本地模型
        localModel = ActorCriticNetwork(4, getNumActions(config)).to(device)

        # 创建环境
        environment = createEnvironment(config)

        # 训练状态
        episodeReward = 0
        episodeLength = 0
        totalEpisodes = 0

        # 初始化状态
        state, _ = environment.reset()
        state = torch.FloatTensor(state).to(device)

        logger.info(f"进程 {processId} 初始化完成，开始训练")

        while globalCounter.value < config.trainingSteps:
            # 同步模型参数
            localModel.load_state_dict(sharedModel.state_dict())

            # 收集经验
            states, actions, rewards = [], [], []

            for step in range(config.tMax):
                # 选择动作
                action, _, _ = localModel.getAction(state)

                # 执行动作
                nextState, reward, done, _ = environment.step(action)
                nextState = torch.FloatTensor(nextState).to(device)

                # 存储经验
                states.append(state.clone())
                actions.append(action)
                rewards.append(reward)

                # 更新状态
                state = nextState
                episodeReward += reward
                episodeLength += 1

                # 更新全局计数器
                with globalCounter.get_lock():
                    globalCounter.value += 1
                    currentStep = globalCounter.value

                # 如果episode结束
                if done:
                    totalEpisodes += 1

                    # 发送训练统计信息
                    if processId == 0 and totalEpisodes % 5 == 0:
                        trainingQueue.put({
                            'processId': processId,
                            'episode': totalEpisodes,
                            'reward': episodeReward,
                            'step': currentStep
                        })

                    # 重置环境
                    state, _ = environment.reset()
                    state = torch.FloatTensor(state).to(device)
                    episodeReward = 0
                    episodeLength = 0
                    break

            # 计算损失
            loss = computeA3CLoss(localModel, states, actions, rewards, done, config)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(localModel.parameters(), config.maxGradientNorm)

            # 将梯度添加到共享模型
            for localParam, sharedParam in zip(localModel.parameters(), sharedModel.parameters()):
                if localParam.grad is not None:
                    if sharedParam.grad is None:
                        sharedParam.grad = localParam.grad.clone()
                    else:
                        sharedParam.grad += localParam.grad.clone()

            # 更新全局模型
            optimizer.step()

            # 定期保存模型
            if processId == 0 and currentStep % config.saveModelFrequency == 0:
                saveModel(sharedModel, optimizer, currentStep, config)

        environment.close()
        logger.info(f"进程 {processId} 训练完成")

    except Exception as e:
        logger.error(f"进程 {processId} 发生错误: {e}")
        import traceback
        logger.error(traceback.format_exc())

def saveModel(model: nn.Module, optimizer: torch.optim.Optimizer,
              step: int, config: A3CConfig):
    """保存模型"""
    modelPath = f"./A3C_models/a3c_model_{config.environmentName.replace('/', '_')}_step_{step}.pth"
    os.makedirs(os.path.dirname(modelPath), exist_ok=True)

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': step,
        'config': config.__dict__
    }, modelPath)

    logger.info(f"模型已保存到: {modelPath}")

def trainingMonitor(trainingQueue: mp.Queue, config: A3CConfig):
    """训练监控进程"""
    episodeRewards = []
    try:
        while True:
            try:
                data = trainingQueue.get(timeout=10)
                if data is None:  # 终止信号
                    break

                episodeRewards.append(data['reward'])

                if len(episodeRewards) % 5 == 0:
                    avgReward = np.mean(episodeRewards[-5:])
                    logger.info(f"进程 {data['processId']} - 回合 {data['episode']}: 奖励 = {data['reward']}, 平均奖励 (最近5回合): {avgReward:.2f}, 总步数: {data['step']}")

            except queue.Empty:
                continue

    except Exception as e:
        logger.error(f"监控进程错误: {e}")

def trainA3C(config: A3CConfig):
    """训练A3C算法"""
    logger.info("开始A3C训练")

    # 获取动作数量
    numActions = getNumActions(config)
    logger.info(f"动作空间大小: {numActions}")

    # 创建共享模型
    sharedModel = ActorCriticNetwork(4, numActions).to(device)
    sharedModel.share_memory()

    # 创建优化器
    optimizer = torch.optim.Adam(sharedModel.parameters(), lr=config.learningRate)

    # 全局计数器
    globalCounter = mp.Value('i', 0)

    # 训练队列
    trainingQueue = mp.Queue()

    # 启动监控进程
    monitorProcess = mp.Process(target=trainingMonitor, args=(trainingQueue, config))
    monitorProcess.start()

    # 创建工作进程
    processes = []
    for i in range(config.numProcesses):
        process = mp.Process(
            target=worker,
            args=(i, sharedModel, optimizer, config, globalCounter, trainingQueue)
        )
        process.start()
        processes.append(process)
        time.sleep(1)  # 避免同时创建过多进程

    logger.info(f"启动 {len(processes)} 个训练进程")

    # 等待训练完成
    try:
        while globalCounter.value < config.trainingSteps:
            time.sleep(5)
            currentStep = globalCounter.value
            if currentStep % config.logInterval == 0:
                logger.info(f"训练进度: {currentStep}/{config.trainingSteps} ({currentStep/config.trainingSteps*100:.1f}%)")

    except KeyboardInterrupt:
        logger.info("收到中断信号，停止训练")

    # 终止进程
    trainingQueue.put(None)  # 发送终止信号给监控进程

    for process in processes:
        process.terminate()
        process.join(timeout=5)

    monitorProcess.terminate()
    monitorProcess.join(timeout=5)

    # 保存最终模型
    saveModel(sharedModel, optimizer, globalCounter.value, config)

    logger.info("A3C训练完成")

def testTrainedModel(config: A3CConfig, modelPath: str, numEpisodes: int = 5):
    """测试训练好的模型"""
    logger.info(f"开始测试模型: {modelPath}")

    # 加载模型
    checkpoint = torch.load(modelPath)
    trainedModel = ActorCriticNetwork(4, getNumActions(config)).to(device)
    trainedModel.load_state_dict(checkpoint['model_state_dict'])
    trainedModel.eval()

    # 创建环境
    environment = createEnvironment(config)

    episodeRewards = []

    for episode in range(numEpisodes):
        state, _ = environment.reset()
        state = torch.FloatTensor(state).to(device)

        episodeReward = 0
        done = False

        while not done:
            # 选择动作
            with torch.no_grad():
                action, _, _ = trainedModel.getAction(state)

            # 执行动作
            nextState, reward, done, _ = environment.step(action)
            nextState = torch.FloatTensor(nextState).to(device)

            # 更新状态
            state = nextState
            episodeReward += reward

        episodeRewards.append(episodeReward)
        logger.info(f"测试回合 {episode + 1}: 奖励 = {episodeReward}")

    avgReward = np.mean(episodeRewards)
    logger.info(f"平均测试奖励: {avgReward:.2f}")

    environment.close()

    return avgReward

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

        # 测试最终模型
        modelPath = f"./A3C_models/a3c_model_{config.environmentName.replace('/', '_')}_step_{config.trainingSteps}.pth"
        if os.path.exists(modelPath):
            testTrainedModel(config, modelPath)

    except KeyboardInterrupt:
        logger.info("训练被用户中断")
    except Exception as e:
        logger.error(f"训练过程中发生错误: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        logger.info("程序执行完毕")

if __name__ == "__main__":
    main()
