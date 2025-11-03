"""
    关于DQN模型训练的主程序
"""

import torch



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

            while True:
                if not isinstance(state, torch.Tensor):
                    state = torch.tensor(state, dtype=torch.float32)
                    state = torch.unsqueeze(state, 0)
                    state = torch.unsqueeze(state, 0)
                state = state.to(self.config.device)
                action = self.agent.selectAction(state)
                nextState, reward, done, info = self.environment.step(action)
                self.experience.push(state, action, reward, nextState, done)
                # 优化模型
                loss = self.agent.optimizeModel(self.experience)
                if loss > 0:
                    totalLoss += loss
                    lossCount += 1
                print(loss)
                state = nextState
                totalReward += reward
                stepsInEpisode += 1

                if done:
                    break

            self.agent.episodesCompleted += 1














