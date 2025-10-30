"""
A3C (Asynchronous Advantage Actor-Critic) 算法实现
基于原PPO代码框架，按照A3C论文完整复现
V1.0 - A3C实现 + 多线程异步训练

主要改进：
1. 实现A3C多线程异步训练架构
2. 添加LSTM支持用于部分环境
3. 实现n-step优势估计
4. 优化多线程同步机制
5. 保持与PPO相同的预处理和网络结构
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
import threading
from threading import Lock
import queue


# =============================== 配置类 ===============================
@dataclass
class A3CConfig:
    """A3C训练配置参数"""
    environmentName: str = "BreakoutNoFrameskip-v4"
    learningRate: float = 0.0007
    discountFactor: float = 0.99
    entropyCoeff: float = 0.01
    valueLossCoeff: float = 0.5
    maxGradNorm: float = 40.0
    nStep: int = 5  # n-step回报
    numProcesses: int = 16  # 并行进程数量
    trainingTimesteps: int = 40000000  # 4000万帧
    targetAverageReward: float = 15.0
    frameSkip: int = 4
    screenSize: int = 84
    useLSTM: bool = False  # 是否使用LSTM
    lstmSize: int = 256  # LSTM隐藏层大小
    useSharedRMSProp: bool = True  # 使用共享RMSProp
    rmsPropAlpha: float = 0.99
    rmsPropEpsilon: float = 0.1
    logInterval: int = 10
    saveInterval: int = 100
    maxEpisodeLength: int = 10000


# =============================== 全局设置 ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./log/a3cTraining.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# =============================== 工具类 ===============================
class MemoryManager:
    """内存管理器，用于监控和优化内存使用"""

    @staticmethod
    def getGpuMemoryUsage() -> Tuple[float, float]:
        """获取GPU内存使用情况"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024 ** 3  # GB
            cached = torch.cuda.memory_reserved() / 1024 ** 3  # GB
            return allocated, cached
        return 0.0, 0.0

    @staticmethod
    def getSystemMemoryUsage() -> float:
        """获取系统内存使用情况"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 ** 3  # GB

    @staticmethod
    def clearMemory():
        """清理内存"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


@contextmanager
def memoryMonitor(operationName: str):
    """监控内存使用的上下文管理器"""
    startTime = time.time()
    gpuMemoryBefore = MemoryManager.getGpuMemoryUsage()
    systemMemoryBefore = MemoryManager.getSystemMemoryUsage()

    try:
        yield
    finally:
        endTime = time.time()
        gpuMemoryAfter = MemoryManager.getGpuMemoryUsage()
        systemMemoryAfter = MemoryManager.getSystemMemoryUsage()

        logger.debug(
            f"{operationName} - "
            f"耗时: {endTime - startTime:.3f}s | "
            f"GPU内存变化: {gpuMemoryAfter[0] - gpuMemoryBefore[0]:+.2f}GB | "
            f"系统内存变化: {systemMemoryAfter - systemMemoryBefore:+.2f}GB"
        )


# =============================== 网络结构 ===============================
class ResidualBlock(nn.Module):
    """修复的残差块，确保尺寸匹配"""

    def __init__(self, inChannels: int, outChannels: int, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(inChannels, outChannels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(outChannels)
        self.conv2 = nn.Conv2d(outChannels, outChannels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outChannels)

        # 快捷连接 - 确保尺寸匹配
        self.shortcut = nn.Sequential()
        if stride != 1 or inChannels != outChannels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inChannels, outChannels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(outChannels)
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


class A3CNetwork(nn.Module):
    """A3C网络 - 包含策略网络和价值网络，支持LSTM"""

    def __init__(self, inputShape: Tuple[int, int, int], numActions: int,
                 useLSTM: bool = False, lstmSize: int = 256):
        super().__init__()

        self.inChannels = 64
        self.numActions = numActions
        self.useLSTM = useLSTM
        self.lstmSize = lstmSize

        # 共享的特征提取层
        self.conv1 = nn.Conv2d(inputShape[0], 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # ResNet层 - 适配84x84输入
        self.layer1 = self._makeLayer(64, 64, 2, stride=1)  # 84x84 -> 84x84
        self.layer2 = self._makeLayer(64, 128, 2, stride=2)  # 84x84 -> 42x42
        self.layer3 = self._makeLayer(128, 256, 2, stride=2)  # 42x42 -> 21x21
        self.layer4 = self._makeLayer(256, 512, 2, stride=2)  # 21x21 -> 11x11

        # 自适应平均池化到固定尺寸
        self.adaptiveAvgPool = nn.AdaptiveAvgPool2d((1, 1))

        # LSTM层（如果使用）
        if useLSTM:
            self.lstm = nn.LSTM(512, lstmSize, batch_first=True)
            featureSize = lstmSize
        else:
            featureSize = 512

        # 策略头
        self.policyHead = nn.Sequential(
            nn.Linear(featureSize, 256),
            nn.ReLU(),
            nn.Linear(256, numActions)
        )

        # 价值头
        self.valueHead = nn.Sequential(
            nn.Linear(featureSize, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        # 初始化权重
        self._initializeWeights()

    def _makeLayer(self, inChannels: int, outChannels: int,
                   numBlocks: int, stride: int) -> nn.Sequential:
        """创建ResNet层"""
        strides = [stride] + [1] * (numBlocks - 1)
        layers = []

        for currentStride in strides:
            layers.append(ResidualBlock(inChannels, outChannels, currentStride))
            inChannels = outChannels

        return nn.Sequential(*layers)

    def _initializeWeights(self):
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

    def forward(self, x: torch.Tensor, hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """前向传播 - 返回动作概率、状态价值和LSTM隐藏状态"""
        batchSize = x.size(0)

        # 共享特征提取
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.adaptiveAvgPool(x)
        x = x.view(batchSize, -1)

        # LSTM处理
        if self.useLSTM:
            # 调整形状以适应LSTM (batch, sequence, features)
            x = x.unsqueeze(1)  # (batch, 1, features)
            x, (hx, cx) = self.lstm(x, hx)
            x = x.squeeze(1)  # (batch, features)
        else:
            hx = None

        # 策略头
        policyLogits = self.policyHead(x)
        actionProbs = F.softmax(policyLogits, dim=-1)

        # 价值头
        stateValue = self.valueHead(x)

        return actionProbs, stateValue.squeeze(-1), hx

    def getAction(self, state: torch.Tensor, hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                  ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """选择动作并返回相关信息"""
        with torch.no_grad():
            actionProbs, stateValue, newHx = self.forward(state, hx)
            dist = torch.distributions.Categorical(actionProbs)
            action = dist.sample()
            logProb = dist.log_prob(action)

            return action, logProb, stateValue, actionProbs, newHx


# =============================== 共享优化器 ===============================
class SharedRMSProp(optim.Optimizer):
    """共享统计信息的RMSProp优化器 - 按照A3C论文实现"""

    def __init__(self, params, lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay)
        super(SharedRMSProp, self).__init__(params, defaults)

        # 初始化共享状态
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['squareAvg'] = torch.zeros_like(p.data)
                # 确保共享张量在正确设备上
                state['squareAvg'] = state['squareAvg'].to(p.device)

    def share_memory(self):
        """共享优化器状态内存"""
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['squareAvg'].share_memory_()

    def step(self, closure=None):
        """执行优化步骤"""
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('RMSProp does not support sparse gradients')

                state = self.state[p]

                squareAvg = state['squareAvg']
                alpha = group['alpha']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])

                # RMSProp更新
                squareAvg.mul_(alpha).addcmul_(grad, grad, value=1 - alpha)
                avg = squareAvg.sqrt().add_(group['eps'])

                p.data.addcdiv_(grad, avg, value=-group['lr'])

        return loss


# =============================== 环境处理 ===============================
class AtariEnvironmentPreprocessor:
    """Atari环境预处理包装器"""

    def __init__(self, environment, frameSkip: int = 4, screenSize: int = 84):
        self.environment = environment
        self.frameSkip = frameSkip
        self.screenSize = screenSize
        self.frameBuffer = deque(maxlen=4)

    def reset(self) -> Tuple[torch.Tensor, dict]:
        """重置环境并返回预处理后的初始状态"""
        try:
            state, info = self.environment.reset()
            processedState = self._preprocessFrame(state)

            # 用相同的帧填充初始缓冲区
            self.frameBuffer.extend([processedState] * 4)

            stateTensor = torch.tensor(np.stack(self.frameBuffer), dtype=torch.float32)
            return stateTensor, info
        except Exception as e:
            logger.error(f"重置环境失败: {e}")
            raise

    def step(self, action: int) -> Tuple[torch.Tensor, float, bool, dict]:
        """执行动作并返回预处理后的结果"""
        try:
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

            nextStateTensor = torch.tensor(np.stack(self.frameBuffer), dtype=torch.float32)
            return nextStateTensor, totalReward, done, info
        except Exception as e:
            logger.error(f"执行动作失败: {e}")
            raise

    def _preprocessFrame(self, frame: np.ndarray) -> np.ndarray:
        """预处理帧：灰度化、调整大小、归一化"""
        try:
            # 转换为灰度图
            if len(frame.shape) == 3:
                frame = np.mean(frame, axis=2)  # 使用numpy提高效率

            # 调整大小
            img = Image.fromarray(frame.astype(np.uint8))
            img = img.resize((self.screenSize, self.screenSize), Image.BILINEAR)
            frame = np.array(img)

            # 归一化到 [0, 1]
            frame = frame.astype(np.float32) / 255.0

            return frame
        except Exception as e:
            logger.error(f"预处理帧失败: {e}")
            raise

    @property
    def actionSpace(self):
        return self.environment.action_space

    @property
    def observationSpace(self):
        return self.environment.observation_space

    def close(self) -> None:
        """关闭环境"""
        self.environment.close()


# =============================== A3C智能体 ===============================
class A3CAgent:
    """A3C智能体 - 包含全局网络和优化器"""

    def __init__(self, stateShape: Tuple[int, int, int], numActions: int, config: A3CConfig):
        self.numActions = numActions
        self.stateShape = stateShape
        self.config = config

        # 全局网络
        self.globalNetwork = A3CNetwork(stateShape, numActions, config.useLSTM, config.lstmSize).to(device)

        # 共享优化器
        if config.useSharedRMSProp:
            self.optimizer = SharedRMSProp(
                self.globalNetwork.parameters(),
                lr=config.learningRate,
                alpha=config.rmsPropAlpha,
                eps=config.rmsPropEpsilon
            )
        else:
            self.optimizer = optim.RMSprop(
                self.globalNetwork.parameters(),
                lr=config.learningRate,
                alpha=config.rmsPropAlpha,
                eps=config.rmsPropEpsilon
            )

        # 共享内存
        self.globalNetwork.share_memory()
        if config.useSharedRMSProp:
            self.optimizer.share_memory()

        # 训练状态
        self.globalStep = 0
        self.episodesCompleted = 0
        self.lock = Lock()

        logger.info(f"A3C智能体初始化完成: 状态形状={stateShape}, 动作数量={numActions}, LSTM={config.useLSTM}")

    def computeLoss(self, states: torch.Tensor, actions: torch.Tensor,
                    returns: torch.Tensor, values: torch.Tensor,
                    logProbs: torch.Tensor, entropy: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """计算A3C损失"""
        # 优势函数
        advantages = returns - values

        # 策略损失 (带熵正则化)
        policyLoss = -(logProbs * advantages.detach()).mean()

        # 价值损失
        valueLoss = advantages.pow(2).mean()

        # 总损失
        totalLoss = (policyLoss
                     + self.config.valueLossCoeff * valueLoss
                     - self.config.entropyCoeff * entropy)

        stats = {
            'policyLoss': policyLoss.item(),
            'valueLoss': valueLoss.item(),
            'entropy': entropy.item(),
            'advantage': advantages.mean().item()
        }

        return totalLoss, stats

    def updateGlobalModel(self, gradients: List[Tuple[torch.Tensor, torch.Tensor]]):
        """使用梯度更新全局模型"""
        with self.lock:
            # 应用梯度
            for param, grad in gradients:
                if grad is not None:
                    param.grad = grad

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.globalNetwork.parameters(), self.config.maxGradNorm)

            # 优化步骤
            self.optimizer.step()
            self.optimizer.zero_grad()

    def saveCheckpoint(self, filePath: str) -> None:
        """保存模型检查点"""
        try:
            checkpoint = {
                'globalNetworkState': self.globalNetwork.state_dict(),
                'optimizerState': self.optimizer.state_dict(),
                'globalStep': self.globalStep,
                'episodesCompleted': self.episodesCompleted,
                'config': self.config
            }
            torch.save(checkpoint, filePath)
            logger.info(f"A3C模型已保存到: {filePath}")
        except Exception as e:
            logger.error(f"保存A3C模型失败: {e}")
            raise

    def loadCheckpoint(self, filePath: str) -> None:
        """加载模型检查点"""
        try:
            checkpoint = torch.load(filePath, map_location=device)
            self.globalNetwork.load_state_dict(checkpoint['globalNetworkState'])
            self.optimizer.load_state_dict(checkpoint['optimizerState'])
            self.globalStep = checkpoint['globalStep']
            self.episodesCompleted = checkpoint['episodesCompleted']
            logger.info(f"A3C模型已从 {filePath} 加载")
        except Exception as e:
            logger.error(f"加载A3C模型失败: {e}")
            raise

    def incrementGlobalStep(self, steps: int = 1):
        """增加全局步数"""
        with self.lock:
            self.globalStep += steps

    def incrementEpisodes(self, episodes: int = 1):
        """增加完成的episode数"""
        with self.lock:
            self.episodesCompleted += episodes


# =============================== A3C工作线程 ===============================
class A3CWorker(threading.Thread):
    """A3C工作线程"""

    def __init__(self, workerId: int, globalAgent: A3CAgent, config: A3CConfig,
                 stateShape: Tuple[int, int, int], numActions: int):
        super().__init__()

        self.workerId = workerId
        self.globalAgent = globalAgent
        self.config = config
        self.stateShape = stateShape
        self.numActions = numActions

        # 本地网络
        self.localNetwork = A3CNetwork(stateShape, numActions, config.useLSTM, config.lstmSize).to(device)

        # 本地环境
        self.environment = None
        self.preprocessedEnvironment = None

        # 训练状态
        self.episodeReward = 0.0
        self.episodeLength = 0
        self.stepsCompleted = 0

        # LSTM状态
        self.hx = None
        self.cx = None

        logger.info(f"工作线程 {workerId} 初始化完成")

    def initializeEnvironment(self) -> None:
        """初始化环境"""
        try:
            self.environment = gym.make(self.config.environmentName, render_mode='rgb_array')
            self.preprocessedEnvironment = AtariEnvironmentPreprocessor(
                self.environment,
                frameSkip=self.config.frameSkip,
                screenSize=self.config.screenSize
            )
        except Exception as e:
            logger.error(f"工作线程 {self.workerId} 环境初始化失败: {e}")
            # 回退到普通版本
            try:
                envName = self.config.environmentName.replace("NoFrameskip", "")
                self.environment = gym.make(envName, render_mode='rgb_array')
                self.preprocessedEnvironment = AtariEnvironmentPreprocessor(
                    self.environment,
                    frameSkip=self.config.frameSkip,
                    screenSize=self.config.screenSize
                )
                logger.info(f"工作线程 {self.workerId} 使用回退环境: {envName}")
            except Exception as e2:
                logger.error(f"工作线程 {self.workerId} 回退环境也失败: {e2}")
                raise

    def syncLocalNetwork(self):
        """同步本地网络参数"""
        self.localNetwork.load_state_dict(self.globalAgent.globalNetwork.state_dict())

    def computeNStepReturns(self, rewards: List[float], values: List[float],
                            lastValue: float, done: bool) -> List[float]:
        """计算n步回报"""
        returns = []
        R = 0 if done else lastValue

        # 反向计算n步回报
        for r, v in zip(reversed(rewards), reversed(values)):
            R = r + self.config.discountFactor * R
            returns.insert(0, R)

        return returns

    def run(self):
        """工作线程主循环"""
        self.initializeEnvironment()

        try:
            while self.globalAgent.globalStep < self.config.trainingTimesteps:
                # 同步本地网络
                self.syncLocalNetwork()

                # 重置环境
                state, _ = self.preprocessedEnvironment.reset()
                state = state.to(device)

                # 重置LSTM状态
                if self.config.useLSTM:
                    self.hx = (torch.zeros(1, 1, self.config.lstmSize).to(device),
                               torch.zeros(1, 1, self.config.lstmSize).to(device))
                else:
                    self.hx = None

                # 重置episode统计
                self.episodeReward = 0.0
                self.episodeLength = 0

                # 存储轨迹数据
                states = []
                actions = []
                rewards = []
                values = []
                logProbs = []
                entropies = []

                episodeDone = False
                stepInEpisode = 0

                while not episodeDone and stepInEpisode < self.config.maxEpisodeLength:
                    # 选择动作
                    action, logProb, value, actionProbs, newHx = self.localNetwork.getAction(
                        state.unsqueeze(0), self.hx
                    )

                    # 执行动作
                    nextState, reward, done, _ = self.preprocessedEnvironment.step(action.item())
                    nextState = nextState.to(device)

                    # 计算熵
                    dist = torch.distributions.Categorical(actionProbs)
                    entropy = dist.entropy()

                    # 存储数据
                    states.append(state)
                    actions.append(action)
                    rewards.append(reward)
                    values.append(value)
                    logProbs.append(logProb)
                    entropies.append(entropy)

                    # 更新状态
                    state = nextState
                    self.hx = newHx

                    # 更新统计
                    self.episodeReward += reward
                    self.episodeLength += 1
                    stepInEpisode += 1
                    self.stepsCompleted += 1

                    # 检查是否达到n步或episode结束
                    if len(states) >= self.config.nStep or done:
                        # 计算最后一个状态的价值
                        with torch.no_grad():
                            if done:
                                lastValue = 0.0
                            else:
                                _, lastValue, _ = self.localNetwork(state.unsqueeze(0), self.hx)
                                lastValue = lastValue.item()

                        # 计算n步回报
                        returns = self.computeNStepReturns(rewards, values, lastValue, done)

                        # 准备训练数据
                        statesTensor = torch.stack(states)
                        actionsTensor = torch.stack(actions)
                        returnsTensor = torch.tensor(returns, dtype=torch.float32, device=device)
                        valuesTensor = torch.stack(values)
                        logProbsTensor = torch.stack(logProbs)
                        entropyTensor = torch.stack(entropies).mean()

                        # 计算损失
                        loss, stats = self.globalAgent.computeLoss(
                            statesTensor, actionsTensor, returnsTensor,
                            valuesTensor, logProbsTensor, entropyTensor
                        )

                        # 计算梯度
                        self.localNetwork.zero_grad()
                        loss.backward()

                        # 收集梯度
                        gradients = []
                        for localParam, globalParam in zip(
                                self.localNetwork.parameters(),
                                self.globalAgent.globalNetwork.parameters()
                        ):
                            if localParam.grad is not None:
                                gradients.append((globalParam, localParam.grad.clone()))

                        # 更新全局模型
                        self.globalAgent.updateGlobalModel(gradients)

                        # 更新全局步数
                        self.globalAgent.incrementGlobalStep(len(states))

                        # 清空轨迹
                        states.clear()
                        actions.clear()
                        rewards.clear()
                        values.clear()
                        logProbs.clear()
                        entropies.clear()

                    episodeDone = done

                # 更新episode统计
                if episodeDone:
                    self.globalAgent.incrementEpisodes(1)

                    # 记录episode结果
                    if self.workerId == 0 and self.globalAgent.episodesCompleted % self.config.logInterval == 0:
                        logger.info(
                            f"工作线程 {self.workerId} | "
                            f"全局步数: {self.globalAgent.globalStep:8d} | "
                            f"Episode: {self.globalAgent.episodesCompleted:6d} | "
                            f"奖励: {self.episodeReward:7.2f} | "
                            f"长度: {self.episodeLength:4d}"
                        )

        except Exception as e:
            logger.error(f"工作线程 {self.workerId} 发生错误: {e}")
        finally:
            if self.preprocessedEnvironment:
                self.preprocessedEnvironment.close()


# =============================== A3C训练器 ===============================
class A3CTrainer:
    """A3C训练器"""

    def __init__(self, config: A3CConfig):
        self.config = config
        self.globalAgent = None
        self.workers = []

    def initializeEnvironment(self) -> Tuple[Tuple[int, int, int], int]:
        """初始化环境并返回状态形状和动作数量"""
        try:
            environment = gym.make(self.config.environmentName, render_mode='rgb_array')
            preprocessedEnvironment = AtariEnvironmentPreprocessor(
                environment,
                frameSkip=self.config.frameSkip,
                screenSize=self.config.screenSize
            )

            numActions = preprocessedEnvironment.actionSpace.n
            stateShape = (4, self.config.screenSize, self.config.screenSize)

            preprocessedEnvironment.close()
            environment.close()

            logger.info(f"环境初始化完成: {self.config.environmentName}")
            return stateShape, numActions

        except Exception as e:
            logger.error(f"环境初始化失败: {e}")
            raise

    def train(self) -> Tuple[A3CAgent, List[float], List[float]]:
        """训练A3C智能体"""
        # 初始化环境
        stateShape, numActions = self.initializeEnvironment()

        # 创建全局智能体
        self.globalAgent = A3CAgent(stateShape, numActions, self.config)

        # 创建工作线程
        self.workers = []
        for i in range(self.config.numProcesses):
            worker = A3CWorker(i, self.globalAgent, self.config, stateShape, numActions)
            self.workers.append(worker)

        # 训练统计
        episodeRewards = []
        movingAverageRewards = []

        bestAverageReward = -float('inf')
        updateCount = 0

        logger.info(
            f"开始训练 {self.config.environmentName}, 使用A3C算法, "
            f"目标时间步数: {self.config.trainingTimesteps}, "
            f"工作线程数量: {self.config.numProcesses}"
        )

        # 启动所有工作线程
        for worker in self.workers:
            worker.start()

        try:
            # 主训练循环 - 监控训练进度
            while (self.globalAgent.globalStep < self.config.trainingTimesteps and
                   any(worker.is_alive() for worker in self.workers)):

                time.sleep(10)  # 每10秒检查一次
                updateCount += 1

                # 定期保存检查点
                if updateCount % self.config.saveInterval == 0 and updateCount > 0:
                    self.globalAgent.saveCheckpoint(
                        f"./A3C_models/a3cCheckpoint_{self.config.environmentName.replace('/', '_')}_step_{self.globalAgent.globalStep}.pth"
                    )

                # 检查停止条件
                if (len(movingAverageRewards) >= 30 and
                        movingAverageRewards[-1] >= self.config.targetAverageReward):
                    logger.info(f"达到目标性能! 在时间步 {self.globalAgent.globalStep}")
                    self.globalAgent.saveCheckpoint(
                        f"./A3C_models/a3cFinal_{self.config.environmentName.replace('/', '_')}.pth"
                    )
                    break

        except KeyboardInterrupt:
            logger.info("训练被用户中断")
        except Exception as e:
            logger.error(f"训练过程中发生错误: {e}")
        finally:
            # 等待所有工作线程结束
            for worker in self.workers:
                worker.join()

            # 保存最终模型
            self.globalAgent.saveCheckpoint(
                f"./A3C_models/a3cFinal_{self.config.environmentName.replace('/', '_')}.pth"
            )

        return self.globalAgent, episodeRewards, movingAverageRewards

    def close(self) -> None:
        """关闭训练器"""
        MemoryManager.clearMemory()


# =============================== 主函数 ===============================
def main():
    """主函数"""
    try:
        # 设置内存优化
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True

        # 创建必要的目录
        os.makedirs('./log', exist_ok=True)
        os.makedirs('./A3C_models', exist_ok=True)

        # A3C训练配置 - 按照论文中的超参数
        config = A3CConfig()

        logger.info("开始A3C训练!")

        # 创建训练器并开始训练
        trainer = A3CTrainer(config)
        trainedAgent, rewards, movingAverages = trainer.train()

        # 输出训练结果
        logger.info("A3C训练完成!")
        if hasattr(trainedAgent, 'globalStep'):
            logger.info(f"最终全局步数: {trainedAgent.globalStep}")
        if hasattr(trainedAgent, 'episodesCompleted'):
            logger.info(f"完成episodes数量: {trainedAgent.episodesCompleted}")

        trainer.close()

    except KeyboardInterrupt:
        logger.info("训练被用户中断")
    except Exception as e:
        logger.error(f"训练过程中发生错误: {e}")
    finally:
        # 最终清理
        MemoryManager.clearMemory()
        logger.info("程序执行完毕")


if __name__ == "__main__":
    main()