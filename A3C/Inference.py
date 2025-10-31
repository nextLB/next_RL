"""
A3C模型推理可视化程序
用于展示训练好的A3C模型在Atari游戏中的表现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PIL import Image
import time
import os
from dataclasses import dataclass
from typing import Tuple

# 配置类
@dataclass
class InferenceConfig:
    """推理配置参数"""
    environmentName: str = "PongNoFrameskip-v4"
    modelPath: str = "./A3C_models/a3c_model_PongNoFrameskip-v4_step_100001.pth"
    frameSkip: int = 4
    screenSize: int = 84
    numEpisodes: int = 3
    maxStepsPerEpisode: int = 1000
    renderDelay: float = 0.01

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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

def createEnvironment(config: InferenceConfig, renderMode: str = 'rgb_array'):
    """创建环境函数"""
    env = gym.make(config.environmentName, render_mode=renderMode)
    preprocessedEnv = AtariEnvironmentPreprocessor(
        env,
        frameSkip=config.frameSkip,
        screenSize=config.screenSize
    )
    return preprocessedEnv, env

def getNumActions(environmentName: str) -> int:
    """获取动作空间大小"""
    env = gym.make(environmentName)
    numActions = env.action_space.n
    env.close()
    return numActions

def loadTrainedModel(config: InferenceConfig) -> ActorCriticNetwork:
    """加载训练好的模型"""
    if not os.path.exists(config.modelPath):
        raise FileNotFoundError(f"Model file not found: {config.modelPath}")

    numActions = getNumActions(config.environmentName)
    model = ActorCriticNetwork(4, numActions).to(device)

    checkpoint = torch.load(config.modelPath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Model loaded successfully from: {config.modelPath}")
    return model

class A3CVisualizer:
    """A3C模型可视化类"""

    def __init__(self, config: InferenceConfig):
        self.config = config
        self.model = loadTrainedModel(config)
        self.preprocessedEnv, self.originalEnv = createEnvironment(config, 'rgb_array')

        # 创建图形界面
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 10))
        self.fig.suptitle(f'A3C Agent Performance - {config.environmentName}', fontsize=16, fontweight='bold')

        # 初始化图形元素
        self.initPlotElements()

    def initPlotElements(self):
        """初始化图形元素"""
        # 清除所有子图
        for ax in self.axes.flat:
            ax.clear()

        # 子图1: 实时游戏画面
        self.axes[0, 0].set_title('Game Screen')
        self.axes[0, 0].set_xticks([])
        self.axes[0, 0].set_yticks([])
        self.gameScreenImg = self.axes[0, 0].imshow(np.zeros((210, 160, 3), dtype=np.uint8))

        # 子图2: 动作概率分布
        self.axes[0, 1].set_title('Action Probability Distribution')
        self.axes[0, 1].set_xlabel('Action')
        self.axes[0, 1].set_ylabel('Probability')
        numActions = getNumActions(self.config.environmentName)
        self.actionBars = self.axes[0, 1].bar(range(numActions), [0]*numActions,
                                            color='skyblue', alpha=0.7)
        self.axes[0, 1].set_ylim(0, 1)

        # 子图3: 奖励曲线
        self.axes[1, 0].set_title('Reward History')
        self.axes[1, 0].set_xlabel('Step')
        self.axes[1, 0].set_ylabel('Reward')
        self.rewardLine, = self.axes[1, 0].plot([], [], 'b-', linewidth=2)
        self.axes[1, 0].grid(True, alpha=0.3)

        # 子图4: 状态价值估计
        self.axes[1, 1].set_title('State Value Estimation')
        self.axes[1, 1].set_xlabel('Step')
        self.axes[1, 1].set_ylabel('Value')
        self.valueLine, = self.axes[1, 1].plot([], [], 'r-', linewidth=2)
        self.axes[1, 1].grid(True, alpha=0.3)

        # 文本信息
        self.infoText = self.fig.text(0.02, 0.02, '', fontsize=10,
                                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))

        plt.tight_layout()

    def updateVisualization(self, frame, stepCount, totalReward, currentReward,
                          actionProbabilities, stateValue, gameScreen):
        """更新可视化"""
        # 更新游戏画面
        self.gameScreenImg.set_array(gameScreen)

        # 更新动作概率分布
        for bar, prob in zip(self.actionBars, actionProbabilities):
            bar.set_height(prob)

        # 更新奖励曲线
        rewardHistory = self.rewardLine.get_ydata()
        stepHistory = self.rewardLine.get_xdata()

        newRewardHistory = np.append(rewardHistory, currentReward) if len(rewardHistory) > 0 else [currentReward]
        newStepHistory = np.append(stepHistory, stepCount) if len(stepHistory) > 0 else [stepCount]

        self.rewardLine.set_data(newStepHistory, newRewardHistory)
        self.axes[1, 0].relim()
        self.axes[1, 0].autoscale_view()

        # 更新状态价值曲线
        valueHistory = self.valueLine.get_ydata()
        valueStepHistory = self.valueLine.get_xdata()

        newValueHistory = np.append(valueHistory, stateValue) if len(valueHistory) > 0 else [stateValue]
        newValueStepHistory = np.append(valueStepHistory, stepCount) if len(valueStepHistory) > 0 else [stepCount]

        self.valueLine.set_data(newValueStepHistory, newValueHistory)
        self.axes[1, 1].relim()
        self.axes[1, 1].autoscale_view()

        # 更新文本信息
        infoStr = (f"Step: {stepCount}\n"
                  f"Total Reward: {totalReward:.2f}\n"
                  f"Current Reward: {currentReward:.2f}\n"
                  f"State Value: {stateValue:.4f}\n"
                  f"Selected Action: {np.argmax(actionProbabilities)}")
        self.infoText.set_text(infoStr)

        # 刷新图形
        self.fig.canvas.draw_idle()
        plt.pause(0.001)

    def runInference(self):
        """运行模型推理并可视化"""
        print(f"Starting inference on {self.config.environmentName}")

        for episode in range(self.config.numEpisodes):
            print(f"Episode {episode + 1}/{self.config.numEpisodes}")

            # 重置环境
            state, _ = self.preprocessedEnv.reset()
            stateTensor = torch.FloatTensor(state).to(device)

            totalReward = 0
            stepCount = 0

            # 重置图形
            self.initPlotElements()

            done = False
            while not done and stepCount < self.config.maxStepsPerEpisode:
                # 获取原始环境画面用于显示
                gameScreen = self.originalEnv.render()

                # 模型推理
                action, policyLogits, value = self.model.getAction(stateTensor)
                actionProbabilities = F.softmax(policyLogits, dim=-1).cpu().numpy().flatten()
                stateValue = value.item()

                # 执行动作
                nextState, reward, done, _ = self.preprocessedEnv.step(action)
                nextStateTensor = torch.FloatTensor(nextState).to(device)

                # 更新状态
                stateTensor = nextStateTensor
                totalReward += reward
                stepCount += 1

                # 更新可视化
                self.updateVisualization(
                    frame=stepCount,
                    stepCount=stepCount,
                    totalReward=totalReward,
                    currentReward=reward,
                    actionProbabilities=actionProbabilities,
                    stateValue=stateValue,
                    gameScreen=gameScreen
                )

                # 控制显示速度
                time.sleep(self.config.renderDelay)

                if done:
                    print(f"Episode finished after {stepCount} steps. Total reward: {totalReward}")
                    break

            # 显示最终结果
            finalText = f"Episode {episode + 1} Completed!\nFinal Reward: {totalReward}\nTotal Steps: {stepCount}"
            self.fig.text(0.5, 0.95, finalText, fontsize=14, fontweight='bold',
                         ha='center', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen"))
            plt.pause(2)  # 暂停2秒显示最终结果

        # 关闭环境
        self.preprocessedEnv.close()
        self.originalEnv.close()

        print("Inference completed!")
        plt.show()

def main():
    """主函数"""
    try:
        # 配置推理参数
        config = InferenceConfig()

        # 创建可视化器并运行
        visualizer = A3CVisualizer(config)
        visualizer.runInference()

    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
