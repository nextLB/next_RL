"""
A3C模型实时推理可视化程序
"""

# NumPy 2.0 兼容性修复
import numpy as np
import sys

# 处理 NumPy 2.0 不兼容问题
def fix_numpy_compatibility():
    """修复 NumPy 2.0 兼容性问题"""
    numpy_version = np.__version__
    print(f"NumPy version: {numpy_version}")

    # 为旧代码提供向后兼容
    if not hasattr(np, 'Inf'):
        np.Inf = np.inf
    if not hasattr(np, 'float128'):
        np.float128 = np.longdouble
    if not hasattr(np, 'float96'):
        np.float96 = np.longdouble
# 在导入其他库之前应用修复
fix_numpy_compatibility()


import torch
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
import time
import argparse
import os
from collections import deque
from PIL import Image
import sys

# 添加项目路径以便导入模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from A3CAgent import ActorCriticNetwork
from Config import A3CConfig



class AtariEnvironmentPreprocessor:
    """Atari环境预处理包装器（用于推理）"""

    def __init__(self, environment, frameSkip: int = 4, screenSize: int = 84):
        self.environment = environment
        self.frameSkip = frameSkip
        self.screenSize = screenSize
        self.frameBuffer = deque(maxlen=4)

    def reset(self):
        """重置环境并返回预处理后的初始状态"""
        state, info = self.environment.reset()
        processedState = self._preprocessFrame(state)

        # 用相同的帧填充初始缓冲区
        for _ in range(4):
            self.frameBuffer.append(processedState)

        stateArray = np.stack(self.frameBuffer, axis=0)  # Shape: (4, 84, 84)
        return stateArray, info

    def step(self, action: int):
        totalReward = 0.0
        terminated = False
        truncated = False

        # 执行frameSkip次动作
        for _ in range(self.frameSkip):
            nextState, reward, terminated, truncated, stepInfo = self.environment.step(action)
            totalReward += reward
            if terminated or truncated:
                break

        # 重要：对奖励进行裁剪
        totalReward = np.clip(totalReward, -1, 1)

        done = terminated or truncated
        processedNextState = self._preprocessFrame(nextState)
        self.frameBuffer.append(processedNextState)

        nextStateArray = np.stack(self.frameBuffer, axis=0)
        return nextStateArray, totalReward, done, stepInfo

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

    def render(self):
        """渲染当前环境状态"""
        return self.environment.render()

    def close(self):
        """关闭环境"""
        self.environment.close()

    @property
    def actionSpace(self):
        return self.environment.action_space

class ModelVisualizer:
    """模型可视化器"""

    def __init__(self, modelPath: str, config: A3CConfig, device: torch.device):
        self.modelPath = modelPath
        self.config = config
        self.device = device
        self.setupFonts()
        self.loadModel()
        self.createEnvironment()

    def setupFonts(self):
        """设置英文字体"""
        try:
            # 尝试使用Arial字体，如果不可用则使用默认字体
            self.font = font_manager.FontProperties(family='Arial', size=12)
            # 测试字体是否可用
            plt.rcParams['font.family'] = 'Arial'
        except:
            print("Arial字体不可用，使用默认字体")
            plt.rcParams['font.family'] = 'sans-serif'
            self.font = font_manager.FontProperties(size=12)

    def loadModel(self):
        """加载训练好的模型"""
        print(f"加载模型: {self.modelPath}")

        if not os.path.exists(self.modelPath):
            raise FileNotFoundError(f"模型文件不存在: {self.modelPath}")

        # 获取动作数量
        env = gym.make(self.config.environmentName)
        self.numActions = env.action_space.n
        env.close()

        # 创建网络并加载权重
        self.model = ActorCriticNetwork(4, self.numActions).to(self.device)
        checkpoint = torch.load(self.modelPath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        print(f"模型加载成功，动作空间: {self.numActions}")

    def createEnvironment(self):
        """创建环境"""
        self.environment = gym.make(
            self.config.environmentName,
            render_mode='rgb_array'
        )
        self.processedEnvironment = AtariEnvironmentPreprocessor(
            self.environment,
            frameSkip=self.config.frameSkip,
            screenSize=self.config.screenSize
        )

    def getActionProbabilities(self, state: torch.Tensor) -> np.ndarray:
        """获取动作概率分布"""
        with torch.no_grad():
            policyLogits, value = self.model(state)
            probabilities = torch.softmax(policyLogits, dim=-1)
            return probabilities.cpu().numpy()[0], value.item()

    def setupVisualization(self):
        """设置可视化界面"""
        self.fig, ((self.axMain, self.axActionProbs), (self.axValue, self.axStats)) = plt.subplots(2, 2, figsize=(15, 10))
        self.fig.suptitle('A3C Model Visualization - Pong', fontsize=16, fontweight='bold')

        # 初始化主游戏画面
        self.imgMain = self.axMain.imshow(np.zeros((210, 160, 3), dtype=np.uint8))
        self.axMain.set_title('Game Screen', fontproperties=self.font)
        self.axMain.axis('off')

        # 初始化动作概率条形图
        self.actionBars = self.axActionProbs.bar(range(self.numActions), np.zeros(self.numActions))
        self.axActionProbs.set_title('Action Probabilities', fontproperties=self.font)
        self.axActionProbs.set_xlabel('Action', fontproperties=self.font)
        self.axActionProbs.set_ylabel('Probability', fontproperties=self.font)
        self.axActionProbs.set_ylim(0, 1)

        # 动作标签（Pong游戏）
        actionLabels = ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE']
        self.axActionProbs.set_xticks(range(self.numActions))
        self.axActionProbs.set_xticklabels(actionLabels[:self.numActions], fontproperties=self.font)

        # 初始化价值函数显示
        self.valueText = self.axValue.text(0.5, 0.5, 'Value: 0.00',
                                          horizontalalignment='center',
                                          verticalalignment='center',
                                          fontsize=20,
                                          transform=self.axValue.transAxes)
        self.axValue.set_title('State Value', fontproperties=self.font)
        self.axValue.axis('off')

        # 初始化统计信息
        self.statsText = self.axStats.text(0.1, 0.9, '', fontproperties=self.font, transform=self.axStats.transAxes)
        self.axStats.set_title('Statistics', fontproperties=self.font)
        self.axStats.axis('off')

        plt.tight_layout()
        plt.ion()  # 开启交互模式
        plt.show()

    def updateVisualization(self, frame: np.ndarray, actionProbs: np.ndarray,
                          value: float, action: int, reward: float,
                          totalReward: float, step: int):
        """更新可视化界面"""
        # 更新主游戏画面
        self.imgMain.set_array(frame)

        # 更新动作概率条形图
        for bar, prob in zip(self.actionBars, actionProbs):
            bar.set_height(prob)
            # 高亮当前选择的动作
            if bar.get_x() == action:
                bar.set_color('red')
            else:
                bar.set_color('blue')

        # 更新价值函数显示
        self.valueText.set_text(f'Value: {value:.3f}')

        # 更新统计信息
        statsInfo = f'Step: {step}\nAction: {action}\nReward: {reward:.2f}\nTotal Reward: {totalReward:.2f}'
        self.statsText.set_text(statsInfo)

        # 刷新画面
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.01)  # 短暂暂停以更新画面

    def runInference(self, numEpisodes: int = 3, maxSteps: int = 1000):
        """运行模型推理"""
        print("开始模型推理...")

        for episode in range(numEpisodes):
            print(f"\n开始回合 {episode + 1}/{numEpisodes}")

            # 重置环境
            state, _ = self.processedEnvironment.reset()
            stateTensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

            totalReward = 0
            done = False
            step = 0

            while not done and step < maxSteps:
                # 获取动作概率和价值
                actionProbs, value = self.getActionProbabilities(stateTensor)

                # 选择动作（使用最大概率的动作，而不是采样）
                action = np.argmax(actionProbs)

                # 执行动作
                nextState, reward, done, _ = self.processedEnvironment.step(action)
                nextStateTensor = torch.FloatTensor(nextState).unsqueeze(0).to(self.device)

                # 获取当前帧用于显示
                currentFrame = self.environment.render()

                # 更新可视化
                self.updateVisualization(
                    frame=currentFrame,
                    actionProbs=actionProbs,
                    value=value,
                    action=action,
                    reward=reward,
                    totalReward=totalReward,
                    step=step
                )

                # 更新状态
                stateTensor = nextStateTensor
                totalReward += reward
                step += 1

                # 短暂延迟以控制游戏速度
                time.sleep(0.05)

            print(f"回合 {episode + 1} 结束，总奖励: {totalReward:.2f}, 步数: {step}")

            # 回合结束后暂停一下
            if episode < numEpisodes - 1:
                print("准备下一回合...")
                time.sleep(2)

    def close(self):
        """关闭环境和可视化"""
        self.processedEnvironment.close()
        plt.close('all')
        print("可视化程序已关闭")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='A3C模型实时推理可视化')
    parser.add_argument('--modelPath', type=str, required=True, help='模型文件路径')
    parser.add_argument('--episodes', type=int, default=3, help='运行回合数')
    parser.add_argument('--maxSteps', type=int, default=1000, help='每回合最大步数')

    args = parser.parse_args()

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 创建配置
    config = A3CConfig()

    try:
        # 创建可视化器
        visualizer = ModelVisualizer(args.modelPath, config, device)

        # 设置可视化界面
        visualizer.setupVisualization()

        # 运行推理
        visualizer.runInference(numEpisodes=args.episodes, maxSteps=args.maxSteps)

    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # 确保资源被正确释放
        if 'visualizer' in locals():
            visualizer.close()

        print("程序执行完毕")

if __name__ == "__main__":
    main()