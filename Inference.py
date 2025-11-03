"""
DQN模型实时推理可视化程序 - 修复版本
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import gymnasium as gym
from PIL import Image
import time
from dataclasses import dataclass
from typing import Tuple


@dataclass
class TrainingConfig:
    """训练配置参数（与训练代码保持一致）"""
    version: str = "V1.2"
    environmentName: str = "PongNoFrameskip-v4"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    imageShape: Tuple[int, int, int] = (1, 120, 120)
    numActions: int = 6
    learningRate: float = 0.00025
    trainingEpisodes: int = 1000
    initialEpsilon: float = 1.0
    finalEpsilon: float = 0.01
    epsilonDecaySteps: int = 100000
    replayBufferCapacity: int = 10000
    discountFactor: float = 0.99
    targetUpdateFrequency: int = 300


class ResidualBlock(torch.nn.Module):
    def __init__(self, inChannels: int, outChannels: int, stride: int = 1):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(inChannels, outChannels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(outChannels)
        self.conv2 = torch.nn.Conv2d(outChannels, outChannels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(outChannels)

        self.shortcut = torch.nn.Sequential()
        if stride != 1 or inChannels != outChannels:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(inChannels, outChannels, kernel_size=1, stride=stride, bias=False),
                torch.nn.BatchNorm2d(outChannels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = torch.nn.functional.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        residual = self.shortcut(residual)
        out += residual
        out = torch.nn.functional.relu(out)
        return out


class ResNetDeepQNetwork(torch.nn.Module):
    def __init__(self, inputShape, numActions):
        super(ResNetDeepQNetwork, self).__init__()
        self.conv1 = torch.nn.Conv2d(inputShape[0], 64, 3, 1, 1)
        self.bn1 = torch.nn.BatchNorm2d(64)

        self.layer1 = self.makeLayer(64, 64, 2, stride=1)
        self.layer2 = self.makeLayer(64, 128, 2, stride=2)
        self.layer3 = self.makeLayer(128, 256, 2, stride=2)
        self.layer4 = self.makeLayer(256, 512, 2, stride=2)

        self.adaptiveAvgPool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(512, numActions)

        self.initializeWeights()

    def makeLayer(self, inChannels: int, outChannels: int, numBlocks: int, stride: int) -> torch.nn.Sequential:
        strides = [stride] + [1] * (numBlocks - 1)
        layers = []
        for currentStride in strides:
            layers.append(ResidualBlock(inChannels, outChannels, currentStride))
            inChannels = outChannels
        return torch.nn.Sequential(*layers)

    def initializeWeights(self):
        for module in self.modules():
            if isinstance(module, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(module.weight, 1)
                torch.nn.init.constant_(module.bias, 0)
            elif isinstance(module, torch.nn.Linear):
                torch.nn.init.normal_(module.weight, 0, 0.01)
                torch.nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.adaptiveAvgPool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class DQNInferenceVisualizer:
    def __init__(self, modelPath, environmentName="PongNoFrameskip-v4"):
        self.modelPath = modelPath
        self.environmentName = environmentName
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 初始化环境
        self.env = gym.make(self.environmentName, render_mode='rgb_array')
        self.numActions = self.env.action_space.n
        self.imageShape = (1, 120, 120)

        # 加载模型
        self.policyNetwork = self.loadModel()
        self.policyNetwork.eval()

        # 初始化matplotlib图形
        self.setupPlot()

        # 状态跟踪
        self.currentState = None
        self.totalReward = 0
        self.stepCount = 0
        self.episodeCount = 0

    def setupPlot(self):
        """设置matplotlib图形窗口"""
        self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        self.fig.suptitle('DQN Model Real-time Inference Visualization', fontsize=16, fontweight='bold')

        # 原始游戏画面
        self.ax1.set_title('Raw Game Screen')
        self.rawImage = self.ax1.imshow(np.zeros((210, 160, 3), dtype=np.uint8))
        self.ax1.axis('off')

        # 预处理后的灰度图
        self.ax2.set_title('Preprocessed Grayscale Frame')
        self.processedImage = self.ax2.imshow(np.zeros((120, 120), dtype=np.float32), cmap='gray')
        self.ax2.axis('off')

        # Q值分布
        self.ax3.set_title('Q-value Distribution for Each Action')
        self.qValuesBars = self.ax3.bar(range(self.numActions), np.zeros(self.numActions))
        self.ax3.set_xlabel('Action Index')
        self.ax3.set_ylabel('Q-value')
        self.ax3.set_xticks(range(self.numActions))

        # 统计信息
        self.ax4.set_title('Training Statistics')
        self.ax4.axis('off')
        self.statisticsText = self.ax4.text(0.1, 0.9, '', transform=self.ax4.transAxes, fontsize=12,
                                           verticalalignment='top', fontfamily='monospace')

        plt.tight_layout()

    def loadModel(self):
        """加载训练好的DQN模型 - 修复版本"""
        try:
            # 创建网络结构
            model = ResNetDeepQNetwork(self.imageShape, self.numActions).to(self.device)

            # 使用 weights_only=False 来加载包含自定义类的检查点
            checkpoint = torch.load(self.modelPath, map_location=self.device, weights_only=False)

            # 加载网络权重
            model.load_state_dict(checkpoint['policyNetworkState'])
            print("Model loaded successfully!")
            return model

        except Exception as e:
            print(f"Error loading model: {e}")
            print("Trying alternative loading method...")

            # 备用加载方法
            try:
                model = ResNetDeepQNetwork(self.imageShape, self.numActions).to(self.device)

                # 使用更宽松的加载方式
                checkpoint = torch.load(
                    self.modelPath,
                    map_location=self.device,
                    weights_only=False,
                    pickle_module=__import__('pickle')
                )

                model.load_state_dict(checkpoint['policyNetworkState'])
                print("Model loaded successfully with alternative method!")
                return model
            except Exception as e2:
                print(f"Alternative loading also failed: {e2}")
                raise

    def preprocessFrame(self, frame):
        """预处理游戏帧（与训练时保持一致）"""
        if len(frame.shape) == 3:
            frame = np.mean(frame, axis=2)

        img = Image.fromarray(frame.astype(np.uint8))
        img = img.resize((self.imageShape[1], self.imageShape[2]), Image.BILINEAR)
        frame = np.array(img)
        frame = frame.astype(np.float32) / 255.0

        return frame

    def selectAction(self, state):
        """选择动作（贪婪策略）"""
        with torch.no_grad():
            # 确保状态张量格式正确
            if not isinstance(state, torch.Tensor):
                state = torch.FloatTensor(state)
            if len(state.shape) == 2:  # 如果是2D，添加batch和channel维度
                state = state.unsqueeze(0).unsqueeze(0)

            state = state.to(self.device)
            qValues = self.policyNetwork(state)
            return qValues.max(1)[1].item(), qValues.cpu().numpy()[0]

    def resetEnvironment(self):
        """重置环境"""
        state, info = self.env.reset()
        processedState = self.preprocessFrame(state)
        self.currentState = processedState
        self.totalReward = 0
        self.stepCount = 0
        self.episodeCount += 1
        return state, processedState

    def stepEnvironment(self, action):
        """执行一步环境交互"""
        nextState, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        processedNextState = self.preprocessFrame(nextState)
        self.currentState = processedNextState
        self.totalReward += reward
        self.stepCount += 1
        return nextState, processedNextState, reward, done, info

    def updateVisualization(self, frame):
        """更新可视化显示"""
        # 更新原始游戏画面
        self.rawImage.set_array(frame)

        # 更新预处理后的灰度图
        self.processedImage.set_array(self.currentState)

        # 更新Q值分布
        _, qValues = self.selectAction(self.currentState)
        for bar, value in zip(self.qValuesBars, qValues):
            bar.set_height(value)

        maxQ = max(qValues) if len(qValues) > 0 else 1
        self.ax3.set_ylim(0, maxQ * 1.1 if maxQ > 0 else 1)

        # 更新统计信息
        statsText = f"""Episode: {self.episodeCount}
Step: {self.stepCount}
Total Reward: {self.totalReward:.2f}
Current Q-values:
"""
        for i, qVal in enumerate(qValues):
            statsText += f"  Action {i}: {qVal:.4f}\n"

        bestAction = np.argmax(qValues)
        statsText += f"Selected Action: {bestAction}"

        self.statisticsText.set_text(statsText)

        return self.rawImage, self.processedImage, *self.qValuesBars, self.statisticsText

    def runInference(self, maxSteps=1000):
        """运行推理并实时可视化"""
        print("Starting DQN inference visualization...")
        print("Close the visualization window to stop.")

        currentFrame, _ = self.resetEnvironment()
        done = False

        def update(frameNum):
            nonlocal currentFrame, done

            if done:
                currentFrame, _ = self.resetEnvironment()
                done = False

            # 选择动作
            action, _ = self.selectAction(self.currentState)

            # 执行动作
            currentFrame, processedFrame, reward, done, info = self.stepEnvironment(action)

            # 更新可视化
            artists = self.updateVisualization(currentFrame)

            # 如果游戏结束，打印统计信息
            if done:
                print(f"Episode {self.episodeCount} finished with total reward: {self.totalReward}")

            return artists

        # 创建动画
        self.animation = FuncAnimation(
            self.fig, update, frames=maxSteps,
            interval=100, blit=True, repeat=True, cache_frame_data=False
        )

        plt.show()

    def close(self):
        """关闭环境和资源"""
        self.env.close()
        if hasattr(self, 'animation'):
            self.animation.event_source.stop()
        plt.close('all')


def createDummyModelIfNotExists(modelPath):
    """如果模型文件不存在，创建一个虚拟模型用于测试"""
    import os
    if not os.path.exists(modelPath):
        print(f"Model file not found at {modelPath}, creating dummy model for testing...")

        # 创建目录
        os.makedirs(os.path.dirname(modelPath), exist_ok=True)

        # 创建配置
        config = TrainingConfig()

        # 创建模型
        model = ResNetDeepQNetwork(config.imageShape, config.numActions)

        # 创建检查点
        checkpoint = {
            'policyNetworkState': model.state_dict(),
            'targetNetworkState': model.state_dict(),
            'optimizerState': None,
            'stepsCompleted': 0,
            'episodesCompleted': 0,
            'config': config
        }

        # 保存模型
        torch.save(checkpoint, modelPath)
        print(f"Dummy model created at {modelPath}")


def main():
    # 模型路径
    modelPath = "./RL_models/DQN_models/best_model.pth"

    # 如果模型不存在，创建虚拟模型
    createDummyModelIfNotExists(modelPath)

    try:
        # 创建可视化器
        visualizer = DQNInferenceVisualizer(modelPath)

        # 运行推理可视化
        visualizer.runInference(maxSteps=1000)

    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'visualizer' in locals():
            visualizer.close()


if __name__ == "__main__":
    main()