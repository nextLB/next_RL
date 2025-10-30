"""
PPO模型推理与可视化程序
用于加载训练好的PPO模型并进行实时游戏演示
"""

import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import patches
from collections import deque
import time
import os
from typing import Tuple, List, Dict, Any, Optional
import torch.nn.functional as F
from dataclasses import dataclass

# =============================== 配置类 ===============================
@dataclass
class PPOConfig:
    """PPO训练配置参数"""
    environmentName: str = "PongNoFrameskip-v4"
    learningRate: float = 0.00025
    clipEpsilon: float = 0.1
    discountFactor: float = 0.99
    gaeLambda: float = 0.95
    valueLossCoeff: float = 0.5
    entropyCoeff: float = 0.01
    ppoEpochs: int = 3
    batchSize: int = 32
    horizon: int = 128  # 每个actor的时间步数
    numActors: int = 8  # 并行actor数量
    trainingTimesteps: int = 10000000
    targetAverageReward: float = 15.0
    frameSkip: int = 4
    screenSize: int = 84
    saveImages: bool = True
    imageSaveDir: str = "recordedPongPPO"
    numEpisodesToRecord: int = 3
    useAdam: bool = True
    adamEpsilon: float = 1e-5
    maxGradNorm: float = 0.5
    bufferOnGpu: bool = True  # 经验缓冲区是否放在GPU上
    logInterval: int = 5  # 日志输出间隔
    saveInterval: int = 50  # 模型保存间隔


# =============================== 网络结构 ===============================
class ResidualBlock(nn.Module):
    """残差块"""

    def __init__(self, inChannels: int, outChannels: int, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(inChannels, outChannels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(outChannels)
        self.conv2 = nn.Conv2d(outChannels, outChannels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outChannels)

        # 快捷连接
        self.shortcut = nn.Sequential()
        if stride != 1 or inChannels != outChannels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inChannels, outChannels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(outChannels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        residual = self.shortcut(residual)
        out += residual
        out = F.relu(out)
        return out


class PPONetwork(nn.Module):
    """PPO网络 - 包含策略网络和价值网络"""

    def __init__(self, inputShape: Tuple[int, int, int], numActions: int):
        super().__init__()

        self.inChannels = 64
        self.numActions = numActions

        # 共享的特征提取层
        self.conv1 = nn.Conv2d(inputShape[0], 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # ResNet层
        self.layer1 = self._makeLayer(64, 64, 2, stride=1)
        self.layer2 = self._makeLayer(64, 128, 2, stride=2)
        self.layer3 = self._makeLayer(128, 256, 2, stride=2)
        self.layer4 = self._makeLayer(256, 512, 2, stride=2)

        # 自适应平均池化
        self.adaptiveAvgPool = nn.AdaptiveAvgPool2d((1, 1))

        # 策略头
        self.policyHead = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, numActions)
        )

        # 价值头
        self.valueHead = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def _makeLayer(self, inChannels: int, outChannels: int,
                   numBlocks: int, stride: int) -> nn.Sequential:
        """创建ResNet层"""
        strides = [stride] + [1] * (numBlocks - 1)
        layers = []
        for currentStride in strides:
            layers.append(ResidualBlock(inChannels, outChannels, currentStride))
            inChannels = outChannels
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """前向传播"""
        # 共享特征提取
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.adaptiveAvgPool(x)
        x = x.view(x.size(0), -1)

        # 策略头
        policyLogits = self.policyHead(x)
        actionProbs = F.softmax(policyLogits, dim=-1)

        # 价值头
        stateValue = self.valueHead(x)

        return actionProbs, stateValue.squeeze(-1), policyLogits

    def getAction(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor]:
        """选择动作并返回相关信息"""
        with torch.no_grad():
            actionProbs, stateValue, policyLogits = self.forward(state)
            dist = torch.distributions.Categorical(actionProbs)
            action = dist.sample()
            logProb = dist.log_prob(action)
            return action, logProb, stateValue, actionProbs


# =============================== 环境预处理 ===============================
class AtariEnvironmentPreprocessor:
    """Atari环境预处理包装器"""

    def __init__(self, environment, frameSkip: int = 4, screenSize: int = 84):
        self.environment = environment
        self.frameSkip = frameSkip
        self.screenSize = screenSize
        self.frameBuffer = deque(maxlen=4)

    def reset(self) -> Tuple[torch.Tensor, dict]:
        """重置环境并返回预处理后的初始状态"""
        state, info = self.environment.reset()
        processedState = self._preprocessFrame(state)

        # 用相同的帧填充初始缓冲区
        self.frameBuffer.extend([processedState] * 4)

        stateTensor = torch.tensor(np.stack(self.frameBuffer), dtype=torch.float32)
        return stateTensor, info

    def step(self, action: int) -> Tuple[torch.Tensor, float, bool, dict]:
        """执行动作并返回预处理后的结果"""
        totalReward = 0.0
        terminated = False
        truncated = False
        info = {}

        # 使用帧跳过
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

    def render(self) -> np.ndarray:
        """渲染当前环境状态"""
        return self.environment.render()

    @property
    def actionSpace(self):
        return self.environment.action_space

    def close(self) -> None:
        """关闭环境"""
        self.environment.close()


# =============================== 可视化器 ===============================
class GameVisualizer:
    """游戏可视化器"""

    def __init__(self, environmentName: str, actionNames: List[str]):
        self.environmentName = environmentName
        self.actionNames = actionNames
        self.setupPlot()

    def setupPlot(self):
        """设置绘图区域"""
        plt.ion()  # 开启交互模式
        self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        self.fig.suptitle(f'PPO Agent - {self.environmentName}', fontsize=16, fontweight='bold')

        # 设置子图
        self.setupGameDisplay(self.ax1)
        self.setupActionProbabilities(self.ax2)
        self.setupStatisticsDisplay(self.ax3)
        self.setupValueDisplay(self.ax4)

        plt.tight_layout()
        plt.subplots_adjust(top=0.92)

    def setupGameDisplay(self, ax):
        """设置游戏显示区域"""
        ax.set_title('Game Screen')
        ax.set_xticks([])
        ax.set_yticks([])
        self.gameImage = ax.imshow(np.zeros((210, 160, 3), dtype=np.uint8))

    def setupActionProbabilities(self, ax):
        """设置动作概率显示区域"""
        ax.set_title('Action Probabilities')
        ax.set_xlabel('Actions')
        ax.set_ylabel('Probability')
        ax.set_ylim(0, 1)
        self.actionBars = ax.bar(range(len(self.actionNames)),
                                 [0] * len(self.actionNames),
                                 color='skyblue', alpha=0.7)
        ax.set_xticks(range(len(self.actionNames)))
        ax.set_xticklabels(self.actionNames, rotation=45)

    def setupStatisticsDisplay(self, ax):
        """设置统计信息显示区域"""
        ax.set_title('Game Statistics')
        ax.set_xticks([])
        ax.set_yticks([])
        self.statsText = ax.text(0.05, 0.95, '', transform=ax.transAxes,
                                 verticalalignment='top', fontfamily='monospace',
                                 fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    def setupValueDisplay(self, ax):
        """设置价值函数显示区域"""
        ax.set_title('State Value and Confidence')
        ax.set_xticks([])
        ax.set_yticks([])
        self.valueText = ax.text(0.05, 0.95, '', transform=ax.transAxes,
                                 verticalalignment='top', fontfamily='monospace',
                                 fontsize=10, bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))

    def updateDisplay(self, gameFrame: np.ndarray, actionProbs: np.ndarray,
                      stateValue: float, step: int, totalReward: float,
                      selectedAction: int, fps: float):
        """更新显示"""
        # 更新游戏画面
        self.gameImage.set_data(gameFrame)
        self.ax1.set_title(f'Game Screen - Step: {step}')

        # 更新动作概率
        for bar, prob in zip(self.actionBars, actionProbs):
            bar.set_height(prob)
        # 高亮选中的动作
        for i, bar in enumerate(self.actionBars):
            if i == selectedAction:
                bar.set_color('red')
                bar.set_alpha(1.0)
            else:
                bar.set_color('skyblue')
                bar.set_alpha(0.7)
        self.ax2.set_ylim(0, max(actionProbs) * 1.1 if max(actionProbs) > 0 else 1)

        # 更新统计信息
        statsInfo = (
            f"Step: {step}\n"
            f"Total Reward: {totalReward:.2f}\n"
            f"Selected Action: {self.actionNames[selectedAction]}\n"
            f"FPS: {fps:.1f}\n"
            f"Max Probability: {max(actionProbs):.3f}\n"
            f"Entropy: {-np.sum(actionProbs * np.log(actionProbs + 1e-8)):.3f}"
        )
        self.statsText.set_text(statsInfo)

        # 更新价值信息
        valueInfo = (
            f"State Value: {stateValue:.3f}\n"
            f"Confidence: {actionProbs[selectedAction]:.3f}\n"
            f"Action Certainty: {'HIGH' if actionProbs[selectedAction] > 0.7 else 'MEDIUM' if actionProbs[selectedAction] > 0.3 else 'LOW'}\n"
            f"Value Range: [{min(actionProbs):.3f}, {max(actionProbs):.3f}]"
        )
        self.valueText.set_text(valueInfo)

        # 刷新显示
        plt.draw()
        plt.pause(0.001)

    def close(self):
        """关闭可视化"""
        plt.ioff()
        plt.close()


# =============================== PPO推理器 ===============================
class PPOInference:
    """PPO模型推理器"""

    def __init__(self, modelPath: str, environmentName: str = "PongNoFrameskip-v4",
                 useCuda: bool = True):
        self.device = torch.device("cuda" if useCuda and torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.environmentName = environmentName
        self.modelPath = modelPath
        self.loadModel()

    def loadModel(self):
        """加载训练好的模型"""
        try:
            # 首先创建环境以获取状态和动作空间信息
            env = gym.make(self.environmentName)
            self.numActions = env.action_space.n
            self.stateShape = (4, 84, 84)  # 标准的Atari预处理形状

            # 创建网络结构
            self.model = PPONetwork(self.stateShape, self.numActions).to(self.device)

            # 加载训练权重 - 修复PyTorch 2.6的weights_only问题
            try:
                # 尝试使用weights_only=True加载
                checkpoint = torch.load(self.modelPath, map_location=self.device, weights_only=True)
            except:
                # 如果失败，使用weights_only=False（需要信任模型来源）
                print("Warning: Using weights_only=False to load model. Only use this if you trust the model source.")
                checkpoint = torch.load(self.modelPath, map_location=self.device, weights_only=False)

            # 检查checkpoint结构并加载权重
            if 'networkState' in checkpoint:
                self.model.load_state_dict(checkpoint['networkState'])
            elif 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                # 如果checkpoint本身就是状态字典
                self.model.load_state_dict(checkpoint)

            print(f"Model loaded successfully from {self.modelPath}")
            print(f"Action space: {self.numActions} actions")
            print(f"State shape: {self.stateShape}")

            env.close()

        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def createEnvironment(self, renderMode: str = 'rgb_array') -> AtariEnvironmentPreprocessor:
        """创建预处理环境"""
        env = gym.make(self.environmentName, render_mode=renderMode)
        preprocessedEnv = AtariEnvironmentPreprocessor(env, frameSkip=4, screenSize=84)
        return preprocessedEnv

    def getActionNames(self) -> List[str]:
        """获取动作名称列表"""
        # 对于Pong环境的标准动作映射
        if "Pong" in self.environmentName:
            return ["NOOP", "FIRE", "RIGHT", "LEFT", "RIGHTFIRE", "LEFTFIRE"]
        else:
            return [f"Action_{i}" for i in range(self.numActions)]

    def runDemo(self, numEpisodes: int = 3, maxSteps: int = 1000,
                render: bool = True, delay: float = 0.01):
        """运行演示"""
        print(f"Starting PPO Demo for {numEpisodes} episodes...")

        # 创建环境和可视化器
        env = self.createEnvironment()
        actionNames = self.getActionNames()
        visualizer = GameVisualizer(self.environmentName, actionNames) if render else None

        episodeRewards = []

        for episode in range(numEpisodes):
            print(f"\n--- Episode {episode + 1}/{numEpisodes} ---")

            state, _ = env.reset()
            state = state.unsqueeze(0).to(self.device)  # 添加batch维度

            totalReward = 0
            stepCount = 0
            startTime = time.time()

            for step in range(maxSteps):
                stepStart = time.time()

                # 模型推理
                with torch.no_grad():
                    action, logProb, stateValue, actionProbs = self.model.getAction(state)

                actionIdx = action.item()
                actionProbsNp = actionProbs.cpu().numpy()[0]
                stateValueNp = stateValue.cpu().numpy()[0]

                # 执行动作
                nextState, reward, done, info = env.step(actionIdx)
                nextState = nextState.unsqueeze(0).to(self.device)

                totalReward += reward
                stepCount += 1

                # 计算FPS
                stepTime = time.time() - stepStart
                fps = 1.0 / stepTime if stepTime > 0 else 0

                # 更新显示
                if render and visualizer:
                    gameFrame = env.render()
                    visualizer.updateDisplay(
                        gameFrame=gameFrame,
                        actionProbs=actionProbsNp,
                        stateValue=stateValueNp,
                        step=stepCount,
                        totalReward=totalReward,
                        selectedAction=actionIdx,
                        fps=fps
                    )

                # 更新状态
                state = nextState

                # 检查是否结束
                if done:
                    print(f"Episode finished after {stepCount} steps, Total reward: {totalReward:.2f}")
                    break

                # 控制速度
                if delay > 0:
                    time.sleep(delay)

            episodeTime = time.time() - startTime
            print(
                f"Episode {episode + 1} completed: {stepCount} steps, Reward: {totalReward:.2f}, Time: {episodeTime:.2f}s")
            episodeRewards.append(totalReward)

            # 等待用户输入继续下一个episode
            if episode < numEpisodes - 1 and render:
                input("Press Enter to continue to next episode...")

        # 关闭环境
        env.close()
        if visualizer:
            visualizer.close()

        # 打印统计信息
        print(f"\n=== Demo Summary ===")
        print(f"Average Reward: {np.mean(episodeRewards):.2f}")
        print(f"Max Reward: {np.max(episodeRewards):.2f}")
        print(f"Min Reward: {np.min(episodeRewards):.2f}")
        print(f"Total Steps: {stepCount}")

        return episodeRewards

    def runHeadless(self, numEpisodes: int = 10, maxSteps: int = 1000):
        """无头模式运行（用于性能测试）"""
        print(f"Running headless mode for {numEpisodes} episodes...")

        env = self.createEnvironment()
        episodeRewards = []
        totalSteps = 0

        for episode in range(numEpisodes):
            state, _ = env.reset()
            state = state.unsqueeze(0).to(self.device)

            totalReward = 0
            stepCount = 0

            for step in range(maxSteps):
                with torch.no_grad():
                    action, _, _, _ = self.model.getAction(state)

                nextState, reward, done, _ = env.step(action.item())
                nextState = nextState.unsqueeze(0).to(self.device)

                totalReward += reward
                stepCount += 1
                state = nextState

                if done:
                    break

            episodeRewards.append(totalReward)
            totalSteps += stepCount

            if (episode + 1) % 5 == 0:
                print(f"Completed {episode + 1}/{numEpisodes} episodes")

        env.close()

        avgReward = np.mean(episodeRewards)
        stdReward = np.std(episodeRewards)

        print(f"\n=== Headless Mode Results ===")
        print(f"Episodes: {numEpisodes}")
        print(f"Average Reward: {avgReward:.2f} ± {stdReward:.2f}")
        print(f"Max Reward: {np.max(episodeRewards):.2f}")
        print(f"Min Reward: {np.min(episodeRewards):.2f}")
        print(f"Average Steps per Episode: {totalSteps / numEpisodes:.1f}")

        return episodeRewards


# =============================== 主函数 ===============================
def main():
    """主函数"""
    # 配置参数
    modelPath = "./PPO_V1_0_models/ppoResNetBest_PongNoFrameskip-v4.pth"  # 修改为你的模型路径
    environmentName = "PongNoFrameskip-v4"
    numEpisodes = 3
    maxSteps = 1000
    render = True
    delay = 0.02  # 控制游戏速度

    # 检查模型文件是否存在
    if not os.path.exists(modelPath):
        print(f"Model file not found: {modelPath}")
        print("Please train the model first or provide the correct path.")
        return

    try:
        # 创建推理器
        inference = PPOInference(modelPath, environmentName, useCuda=True)

        # 运行演示
        if render:
            rewards = inference.runDemo(
                numEpisodes=numEpisodes,
                maxSteps=maxSteps,
                render=True,
                delay=delay
            )
        else:
            rewards = inference.runHeadless(
                numEpisodes=numEpisodes,
                maxSteps=maxSteps
            )

        print(f"\nDemo completed successfully!")

    except Exception as e:
        print(f"Error during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()