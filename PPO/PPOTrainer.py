"""
    PPO模型训练器
"""

import torch
import numpy as np
import logging

class V1_2_PPOTrainer:
    def __init__(self, environment, agent, experienceBuffer, config):
        self.name = 'V1.2_PPOTrainer'
        self.environment = environment
        self.agent = agent
        self.experienceBuffer = experienceBuffer
        self.config = config
        self.logger = logging.getLogger(__name__)

    def train(self):
        # 训练统计
        episodeRewards = []
        movingAverageRewards = []
        bestAverageReward = -float('inf')

        for episode in range(self.config.trainingEpisodes):
            state, info = self.environment.reset()
            episodeReward = 0.0
            episodeSteps = 0

            # 重置经验缓冲区
            self.experienceBuffer.clear()

            while True:
                if not isinstance(state, torch.Tensor):
                    state = torch.tensor(state, dtype=torch.float32)
                    state = torch.unsqueeze(state, 0)
                    state = torch.unsqueeze(state, 0)
                state = state.to(self.config.device)

                # 选择动作
                action, logProb, value = self.agent.select_action(state)

                # 与环境交互
                nextState, reward, done, info = self.environment.step(action)

                # 存储经验
                self.experienceBuffer.push(state.cpu(), action, reward, done, value, logProb)

                state = nextState
                episodeReward += reward
                episodeSteps += 1
                self.agent.stepsCompleted += 1

                if done:
                    # 计算最后一个状态的价值
                    if not isinstance(state, torch.Tensor):
                        state = torch.tensor(state, dtype=torch.float32)
                        state = torch.unsqueeze(state, 0)
                        state = torch.unsqueeze(state, 0)
                    state = state.to(self.config.device)
                    with torch.no_grad():
                        _, lastValue = self.agent.network(state)
                        lastValue = lastValue.item()
                    break

            # 计算回报和优势
            returns, advantages = self.experienceBuffer.compute_returns_and_advantages(
                lastValue, self.config.gamma, self.config.gaeLambda
            )

            # 标准化优势
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # 获取批次数据
            states, actions, oldLogProbs = self.experienceBuffer.get_batch_data()

            # 转移到设备
            states = states.to(self.config.device)
            actions = actions.to(self.config.device)
            oldLogProbs = oldLogProbs.to(self.config.device)
            returns = returns.to(self.config.device)
            advantages = advantages.to(self.config.device)

            # 更新网络
            actorLoss, criticLoss, entropyLoss = self.agent.update(
                states, actions, oldLogProbs, advantages, returns
            )

            self.agent.episodesCompleted += 1

            # 记录统计信息
            episodeRewards.append(episodeReward)

            # 计算移动平均奖励
            if len(episodeRewards) >= 50:
                movingAverage = np.mean(episodeRewards[-50:])
            else:
                movingAverage = np.mean(episodeRewards)
            movingAverageRewards.append(movingAverage)

            # 保存最佳模型
            if movingAverage > bestAverageReward:
                bestAverageReward = movingAverage
                self.agent.save_checkpoint(f"./RL_models/PPO_models/best_model.pth")
                self.logger.info(f"新的最佳模型已保存，平均奖励: {bestAverageReward:.2f}")

            # 定期日志输出
            if episode % 5 == 0:
                stats = self.agent.get_training_statistics()
                self.logger.info(
                    f"回合 {episode:4d} | "
                    f"奖励: {episodeReward:7.2f} | "
                    f"步数: {episodeSteps:4d} | "
                    f"移动平均: {movingAverage:7.2f} | "
                    f"Actor损失: {actorLoss:7.4f} | "
                    f"Critic损失: {criticLoss:7.4f} | "
                    f"熵损失: {entropyLoss:7.4f}"
                )

        self.logger.info("PPO训练完成！")
