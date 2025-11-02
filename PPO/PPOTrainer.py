"""
PPO训练器
"""
import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import List, Dict, Tuple
import torch

from Config import PPOConfig, device, LunarLanderConfig
from PPOAgent import PPOAgent
from Environment import AtariEnvironmentPreprocessor, EnvironmentRecorder, LunarLanderEnvironment
from Experience import PPOBuffer
from Memory import MemoryManager

logger = logging.getLogger(__name__)


class PPOTrainer:
    """PPO训练器"""

    def __init__(self, config: PPOConfig):
        self.config = config
        self.environment = None
        self.preprocessedEnvironment = None
        self.agent = None
        self.buffer = None

    def initializeEnvironment(self) -> None:
        """初始化环境"""
        try:
            import gymnasium as gym
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

    def collectExperience(self) -> Tuple[float, int, torch.Tensor]:
        """收集经验数据 - 确保所有数据都在GPU上"""
        import torch
        totalReward = 0
        episodeCount = 0

        # 初始化状态 - 确保在GPU上
        states = []
        for _ in range(self.config.numActors):
            state, _ = self.preprocessedEnvironment.reset()
            states.append(state)
        states = torch.stack(states).to(device)

        # 收集经验
        for step in range(self.config.horizon):
            with torch.no_grad():
                actions, logProbs, values, _ = self.agent.network.getAction(states)

            # 执行动作
            nextStates = []
            rewards = []
            dones = []
            infos = []

            for i in range(self.config.numActors):
                nextState, reward, done, info = self.preprocessedEnvironment.step(actions[i].item())
                nextStates.append(nextState)
                rewards.append(reward)
                dones.append(done)
                infos.append(info)

                totalReward += reward
                if done:
                    episodeCount += 1

            # 转换为张量并确保在GPU上
            nextStates = torch.stack(nextStates).to(device)
            rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
            dones = torch.tensor(dones, dtype=torch.bool, device=device)

            # 存储经验 - 所有数据已经在GPU上
            self.buffer.push(states, actions, rewards, values, logProbs, dones)

            # 更新状态
            states = nextStates

            # 如果环境结束，重置
            for i in range(self.config.numActors):
                if dones[i]:
                    state, _ = self.preprocessedEnvironment.reset()
                    states[i] = state.to(device)

        # 计算最后一个状态的价值
        with torch.no_grad():
            _, lastValues, _ = self.agent.network(states)

        return totalReward, episodeCount, lastValues

    def train(self) -> Tuple[PPOAgent, List[float], List[float]]:
        """训练PPO智能体"""

        # 首先记录原始环境数据
        self.recordInitialEpisodes()

        if self.preprocessedEnvironment is None:
            self.initializeEnvironment()

        numActions = self.preprocessedEnvironment.actionSpace.n
        stateShape = (4, self.config.screenSize, self.config.screenSize)

        self.agent = PPOAgent(stateShape, numActions, self.config)
        self.buffer = PPOBuffer(self.config.horizon, self.config.numActors, stateShape, device)

        # 训练统计
        episodeRewards = []
        movingAverageRewards = []
        trainingStatsHistory = []

        bestAverageReward = -float('inf')
        totalTimesteps = 0
        updateCount = 0

        logger.info(
            f"开始训练 {self.config.environmentName}, 使用PPO算法, 目标时间步数: {self.config.trainingTimesteps}")

        while totalTimesteps < self.config.trainingTimesteps:
            try:
                # 收集经验
                totalReward, episodeCount, lastValues = self.collectExperience()
                totalTimesteps += self.config.horizon * self.config.numActors
                self.agent.stepsCompleted = totalTimesteps
                self.agent.episodesCompleted += episodeCount

                # 计算优势函数和回报 - 在GPU上执行
                self.buffer.computeAdvantagesAndReturns(
                    lastValues,
                    self.config.discountFactor,
                    self.config.gaeLambda
                )

                # 更新网络
                stats = self.agent.update(self.buffer)
                trainingStatsHistory.append(stats)
                updateCount += 1

                # 记录奖励
                if episodeCount > 0:
                    episodeRewards.append(totalReward / episodeCount)
                else:
                    episodeRewards.append(totalReward)

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
                        f"./PPO_models/ppoResNetBest_{self.config.environmentName.replace('/', '_')}.pth"
                    )

                # 定期日志输出
                if updateCount % self.config.logInterval == 0:
                    trainingStats = self.agent.getTrainingStatistics()
                    logger.info(
                        f"时间步 {totalTimesteps:8d} | "
                        f"平均奖励: {episodeRewards[-1]:7.2f} | "
                        f"移动平均: {movingAverage:7.2f} | "
                        f"策略损失: {stats['policyLoss']:7.4f} | "
                        f"价值损失: {stats['valueLoss']:7.4f} | "
                        f"熵: {stats['entropy']:7.4f} | "
                        f"裁剪比例: {stats['clipFraction']:7.4f}"
                    )

                # 定期保存检查点
                if updateCount % self.config.saveInterval == 0 and updateCount > 0:
                    self.agent.saveCheckpoint(
                        f"./PPO_models/ppoResNetCheckpoint_{self.config.environmentName.replace('/', '_')}_step_{totalTimesteps}.pth"
                    )

                # 检查停止条件
                if (len(movingAverageRewards) >= 30 and
                        movingAverageRewards[-1] >= self.config.targetAverageReward):
                    logger.info(f"达到目标性能! 在时间步 {totalTimesteps}")
                    self.agent.saveCheckpoint(
                        f"./PPO_models/ppoResNetFinal_{self.config.environmentName.replace('/', '_')}.pth"
                    )
                    break

                # 清空缓冲区
                self.buffer.clear()

            except Exception as e:
                logger.error(f"训练过程中发生错误: {e}")
                MemoryManager.clearMemory()
                continue

        self._plotTrainingResults(episodeRewards, movingAverageRewards, trainingStatsHistory)
        return self.agent, episodeRewards, movingAverageRewards

    def _plotTrainingResults(self, episodeRewards: List[float],
                             movingAverageRewards: List[float],
                             trainingStats: List[Dict[str, float]]) -> None:
        """绘制训练结果图表"""
        try:
            plt.figure(figsize=(15, 12))

            # 奖励曲线
            plt.subplot(3, 2, 1)
            plt.plot(episodeRewards, alpha=0.6, label='Reward per episode')
            plt.plot(movingAverageRewards, 'r-', linewidth=2, label='Moving average reward (50 episodes)')
            plt.title(f'{self.config.environmentName} - PPO Episode Rewards')
            plt.xlabel('Update Step')
            plt.ylabel('Reward')
            plt.legend()
            plt.grid(True)

            # 策略损失
            plt.subplot(3, 2, 2)
            policyLosses = [stats['policyLoss'] for stats in trainingStats]
            plt.plot(policyLosses)
            plt.title(f'{self.config.environmentName} - PPO Policy Loss')
            plt.xlabel('Update Step')
            plt.ylabel('Policy Loss')
            plt.grid(True)

            # 价值损失
            plt.subplot(3, 2, 3)
            valueLosses = [stats['valueLoss'] for stats in trainingStats]
            plt.plot(valueLosses)
            plt.title(f'{self.config.environmentName} - PPO Value Loss')
            plt.xlabel('Update Step')
            plt.ylabel('Value Loss')
            plt.grid(True)

            # 熵
            plt.subplot(3, 2, 4)
            entropies = [stats['entropy'] for stats in trainingStats]
            plt.plot(entropies)
            plt.title(f'{self.config.environmentName} - PPO Entropy')
            plt.xlabel('Update Step')
            plt.ylabel('Entropy')
            plt.grid(True)

            # 裁剪比例
            plt.subplot(3, 2, 5)
            clipFractions = [stats['clipFraction'] for stats in trainingStats]
            plt.plot(clipFractions)
            plt.title(f'{self.config.environmentName} - PPO Clip Fraction')
            plt.xlabel('Update Step')
            plt.ylabel('Clip Fraction')
            plt.grid(True)

            # KL散度
            plt.subplot(3, 2, 6)
            approxKls = [stats['approxKl'] for stats in trainingStats]
            plt.plot(approxKls)
            plt.title(f'{self.config.environmentName} - PPO Approximate KL Divergence')
            plt.xlabel('Update Step')
            plt.ylabel('KL Divergence')
            plt.grid(True)

            plt.tight_layout()
            plotFileName = f'ppoTrainingResults_{self.config.environmentName.replace("/", "_")}.png'
            plt.savefig(plotFileName, dpi=150, bbox_inches='tight')
            plt.close()
            logger.info(f"训练图表已保存到: {plotFileName}")  # 日志保持中文便于调试
        except Exception as e:
            logger.error(f"绘制训练结果失败: {e}")

    def close(self) -> None:
        """关闭环境"""
        if self.preprocessedEnvironment:
            self.preprocessedEnvironment.close()
        MemoryManager.clearMemory()




# 月球着陆器游戏的PPO算法的训练器
class LunarLanderPPOTrainer:
    def __init__(self, config: LunarLanderConfig):
        self.config = config
        self.environment = LunarLanderEnvironment(self.config)
        self.agent = None
        self.buffer = None

    def train(self):
        """训练PPO智能体"""

        # 首先可视化一下环境视频保存下来
        self.environment.save_episode_videos()













