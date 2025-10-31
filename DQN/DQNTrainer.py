"""
    DQN训练器的实现
"""
from logging import Logger
import Config
from Environment import AtariEnvironmentPreprocessor, EnvironmentRecorder
import gymnasium as gym


class DQNTrainer:
    """DQN训练器 - GPU加速版本"""

    def __init__(self, config: Config.Config):
        self.config = config
        self.environment = None
        self.preprocessedEnvironment = None
        self.agent = None

    def initializeEnvironment(self) -> None:
        """初始化环境"""
        # 使用NoFrameskip版本
        self.environment = gym.make(self.config.environmentName, render_mode='rgb_array')
        self.preprocessedEnvironment = AtariEnvironmentPreprocessor(
            self.environment,
            frameSkip=self.config.frameSkip,
            screenSize=self.config.screenSize
        )

    def recordInitialEpisodes(self) -> None:
        """在训练之前记录原始环境数据为PNG图片"""
        if not self.config.saveImages:
            return

        recorder = EnvironmentRecorder(self.config)

        try:
            print('开始预存数据集')
            # 记录指定数量的episode
            for episode in range(self.config.numEpisodesToRecord):
                print(f"记录第 {episode + 1}/{self.config.numEpisodesToRecord} 个episode...")
                recorder.recordEpisode(episode, maxSteps=500)

        except Exception as e:
            print(f"环境记录过程中发生错误: {e}")
        finally:
            recorder.close()

    def train(self, logger: Logger) -> None:
        """训练DQN智能体 - GPU加速版本"""
        # 首先记录原始环境数据
        self.recordInitialEpisodes()

