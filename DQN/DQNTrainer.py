"""
DQN训练器模块
"""
import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from Config import TrainingConfig
from Environment import AtariEnvironmentPreprocessor
from DQNAgent import DeepQNAgent
from Memory import MemoryManager
import logging
import os

os.makedirs('./DQN_models', exist_ok=True)

logger = logging.getLogger(__name__)


class DQNTrainer:
    """DQN训练器 - GPU加速版本"""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.environment = None
        self.preprocessedEnvironment = None
        self.agent = None

    def initializeEnvironment(self) -> None:
        """初始化环境"""
        try:
            # 使用NoFrameskip版本
            self.environment = gym.make(self.config.environmentName, render_mode='rgb_array')
            self.preprocessedEnvironment = AtariEnvironmentPreprocessor(
                self.environment,
                frameSkip=self.config.frameSkip,
                screenSize=self.config.screenSize
            )
            logger.info(f"环境初始化完成: {self.config.environmentName}")
        except Exception as e:
            logger.error(f"环境初始化失败: {e}")
            # 回退到普通版本
            try:
                envName = self.config.environmentName.replace("NoFrameskip", "")
                self.environment = gym.make(envName, render_mode='rgb_array')
                self.preprocessedEnvironment = AtariEnvironmentPreprocessor(
                    self.environment,
                    frameSkip=self.config.frameSkip,
                    screenSize=self.config.screenSize
                )
                logger.info(f"使用回退环境: {envName}")
            except Exception as e2:
                logger.error(f"回退环境也失败: {e2}")
                raise

    def recordInitialEpisodes(self) -> None:
        """在训练之前记录原始环境数据为PNG图片"""
        if not self.config.saveImages:
            logger.info("图像保存功能已禁用，跳过环境记录")
            return

        logger.info("开始记录原始环境数据为PNG图片...")
        from Environment import EnvironmentRecorder
        recorder = EnvironmentRecorder(self.config)

        try:
            # 记录指定数量的episode
            for episode in range(self.config.numEpisodesToRecord):
                logger.info(f"记录第 {episode + 1}/{self.config.numEpisodesToRecord} 个episode...")
                recorder.recordEpisode(episode, maxSteps=500)

            logger.info(f"环境记录完成！所有帧已保存到: {recorder.saveDir}")
        except Exception as e:
            logger.error(f"环境记录过程中发生错误: {e}")
        finally:
            recorder.close()

    def train(self) -> tuple:
        """训练DQN智能体 - GPU加速版本"""
        # 首先记录原始环境数据
        self.recordInitialEpisodes()

        if self.preprocessedEnvironment is None:
            self.initializeEnvironment()

        numActions = self.preprocessedEnvironment.actionSpace.n
        stateShape = (4, self.config.screenSize, self.config.screenSize)

        self.agent = DeepQNAgent(stateShape, numActions, self.config)

        # 训练统计
        episodeRewards = []
        episodeLosses = []
        movingAverageRewards = []
        epsilonHistory = []

        bestAverageReward = -float('inf')

        logger.info(
            f"开始训练 {self.config.environmentName}, 使用GPU加速ResNet架构, 目标回合数: {self.config.trainingEpisodes}")

        for episode in range(self.config.trainingEpisodes):
            try:
                # 定期清理内存
                if episode % 10 == 0:
                    MemoryManager.clearMemory()

                state, _ = self.preprocessedEnvironment.reset()
                totalReward = 0.0
                stepsInEpisode = 0
                totalLoss = 0.0
                lossCount = 0

                while True:
                    # 确保状态张量格式正确并移动到GPU
                    if not isinstance(state, torch.Tensor):
                        state = torch.tensor(state, dtype=torch.float32)
                    state = state.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))  # 立即移动到GPU

                    action = self.agent.selectAction(state, training=True)
                    nextState, reward, done, _ = self.preprocessedEnvironment.step(action)

                    # 确保下一个状态张量格式正确并移动到GPU
                    if not isinstance(nextState, torch.Tensor):
                        nextState = torch.tensor(nextState, dtype=torch.float32)
                    nextState = nextState.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))  # 立即移动到GPU

                    self.agent.memory.push(state, action, reward, nextState, done)

                    loss = self.agent.optimizeModel()
                    if loss > 0:
                        totalLoss += loss
                        lossCount += 1

                    state = nextState
                    totalReward += reward
                    stepsInEpisode += 1

                    if done:
                        break

                self.agent.episodesCompleted += 1

                # 记录统计信息
                averageLoss = totalLoss / lossCount if lossCount > 0 else 0.0
                episodeRewards.append(totalReward)
                episodeLosses.append(averageLoss)
                epsilonHistory.append(self.agent.getCurrentEpsilon())

                # 计算移动平均奖励
                if len(episodeRewards) >= 50:
                    movingAverage = np.mean(episodeRewards[-50:])
                else:
                    movingAverage = np.mean(episodeRewards)
                movingAverageRewards.append(movingAverage)

                # 更新最佳模型
                if movingAverage > bestAverageReward and len(episodeRewards) >= 20:
                    bestAverageReward = movingAverage
                    self.agent.saveCheckpoint(
                        f"./DQN_models/gpu_resnet_dqn_best_{self.config.environmentName.replace('/', '_')}.pth"
                    )

                # 定期日志输出
                if episode % 5 == 0:
                    stats = self.agent.getTrainingStatistics()
                    logger.info(
                        f"回合 {episode:4d} | "
                        f"奖励: {totalReward:7.2f} | "
                        f"步数: {stepsInEpisode:4d} | "
                        f"移动平均: {movingAverage:7.2f} | "
                        f"平均损失: {averageLoss:7.4f} | "
                        f"Epsilon: {stats['currentEpsilon']:.3f} | "
                        f"经验回放: {stats['memoryUsage']:5.1f}% | "
                        f"GPU内存: {stats['gpuMemoryAllocated']:.2f}GB"
                    )

                    # 定期保存检查点
                    if episode % 50 == 0 and episode > 0:
                        self.agent.saveCheckpoint(
                            f"./DQN_models/gpu_resnet_dqn_checkpoint_{self.config.environmentName.replace('/', '_')}_episode_{episode}.pth"
                        )

                # 检查停止条件
                if (len(movingAverageRewards) >= 30 and
                        movingAverageRewards[-1] >= self.config.targetAverageReward):
                    logger.info(f"达到目标性能! 在回合 {episode}")
                    self.agent.saveCheckpoint(
                        f"./DQN_models/gpu_resnet_dqn_final_{self.config.environmentName.replace('/', '_')}.pth"
                    )
                    break

            except Exception as e:
                logger.error(f"回合 {episode} 训练过程中发生错误: {e}")
                MemoryManager.clearMemory()
                continue

        self._plotTrainingResults(episodeRewards, movingAverageRewards, episodeLosses, epsilonHistory)
        return self.agent, episodeRewards, movingAverageRewards

    def _plotTrainingResults(self, episodeRewards: list, movingAverageRewards: list,
                             episodeLosses: list, epsilonHistory: list) -> None:
        """绘制训练结果图表"""
        try:
            plt.figure(figsize=(15, 10))

            # 奖励曲线
            plt.subplot(2, 2, 1)
            plt.plot(episodeRewards, alpha=0.6, label='每回合奖励')
            plt.plot(movingAverageRewards, 'r-', linewidth=2, label='移动平均奖励 (50回合)')
            plt.title(f'{self.config.environmentName} - GPU加速ResNet DQN回合奖励')
            plt.xlabel('回合')
            plt.ylabel('奖励')
            plt.legend()
            plt.grid(True)

            # 损失曲线
            plt.subplot(2, 2, 2)
            plt.plot(episodeLosses)
            plt.title(f'{self.config.environmentName} - GPU加速ResNet DQN训练损失')
            plt.xlabel('回合')
            plt.ylabel('损失')
            plt.grid(True)

            # Epsilon衰减
            plt.subplot(2, 2, 3)
            plt.plot(epsilonHistory)
            plt.title(f'{self.config.environmentName} - GPU加速ResNet DQN Epsilon衰减')
            plt.xlabel('回合')
            plt.ylabel('Epsilon')
            plt.grid(True)

            # 奖励分布
            plt.subplot(2, 2, 4)
            plt.hist(episodeRewards, bins=50, alpha=0.7)
            plt.title(f'{self.config.environmentName} - GPU加速ResNet DQN奖励分布')
            plt.xlabel('奖励')
            plt.ylabel('频率')
            plt.grid(True)

            plt.tight_layout()
            plotFileName = f'gpu_resnet_training_results_{self.config.environmentName.replace("/", "_")}.png'
            plt.savefig(plotFileName, dpi=150, bbox_inches='tight')
            plt.close()
            logger.info(f"训练图表已保存到: {plotFileName}")
        except Exception as e:
            logger.error(f"绘制训练结果失败: {e}")

    def close(self) -> None:
        """关闭环境"""
        if self.preprocessedEnvironment:
            self.preprocessedEnvironment.close()
        MemoryManager.clearMemory()

