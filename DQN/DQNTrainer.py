"""
    关于DQN模型训练的主程序
"""

import torch
import numpy as np



class V1_2_DQNTrainer:
    def __init__(self, environment, agent, experience, config):
        self.name = 'V1.2_DQNTrainer'
        self.environment = environment
        self.agent = agent
        self.config = config
        self.experience = experience

    def train(self):

        # 训练统计
        episodeRewards = []
        episodeLosses = []
        movingAverageRewards = []
        epsilonHistory = []
        bestAverageReward = -float('inf')

        for episode in range(self.config.trainingEpisodes):
            state, info = self.environment.reset()
            totalReward = 0.0
            stepsInEpisode = 0
            totalLoss = 0.0
            lossCount = 0
            loss = 0

            while True:
                if not isinstance(state, torch.Tensor):
                    state = torch.tensor(state, dtype=torch.float32)
                    state = torch.unsqueeze(state, 0)
                    state = torch.unsqueeze(state, 0)
                state = state.to(self.config.device)
                action = self.agent.selectAction(state)
                # 输入的环境中进行交互，返回信息
                nextState, reward, done, info = self.environment.step(action)
                # 添加到经验池中
                self.experience.push(state, action, reward, nextState, done)

                # 满足一定经验池的数量限制后再进行优化模型
                if len(self.experience.buffer) >= self.config.replayBufferCapacity:
                    # 优化模型
                    loss = self.agent.optimizeModel(self.experience)

                # 统计信息与数据
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

            # 计算移动平均奖励  五十个回合内的
            if len(episodeRewards) >= 50:
                movingAverage = np.mean(episodeRewards[-50:])
            else:
                movingAverage = np.mean(episodeRewards)
            movingAverageRewards.append(movingAverage)

            # 保存更新最佳模型
            if movingAverage > bestAverageReward:
                bestAverageReward = movingAverage
                self.agent.saveCheckpoint(
                    f"./RL_models/DQN_models/best_model.pth"
                )

            # 定期日志输出
            if episode % 5 == 0:
                stats = self.agent.getTrainingStatistics()
                print(
                    f"回合 {episode:4d} | "
                    f"奖励: {totalReward:7.2f} | "
                    f"步数: {stepsInEpisode:4d} | "
                    f"移动平均: {movingAverage:7.2f} | "
                    f"平均损失: {averageLoss:7.4f} | "
                    f"Epsilon: {stats['currentEpsilon']:.3f} | "
                )











