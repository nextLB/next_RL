"""
A3C训练器主模块
"""

import torch
import torch.multiprocessing as mp
import time
import os
import json
from Config import A3CConfig
from Environment import getNumActions
from A3CAgent import ActorCriticNetwork
from WorkerThread import worker
from Log import trainingMonitor, logger
import numpy as np

# 修复多进程启动问题
if mp.get_start_method() != 'spawn':
    mp.set_start_method('spawn', force=True)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

class TrainingState:
    """训练状态管理"""
    def __init__(self):
        self.bestAverageReward = -float('inf')
        self.currentStep = 0
        self.episodeRewards = []

    def update(self, reward: float, step: int):
        self.episodeRewards.append(reward)
        if len(self.episodeRewards) > 100:  # 保持最近100个回合
            self.episodeRewards.pop(0)
        self.currentStep = step

    def getAverageReward(self) -> float:
        if not self.episodeRewards:
            return -float('inf')
        return np.mean(self.episodeRewards)

    def isBestModel(self, threshold: float = -18.0) -> bool:
        currentAvg = self.getAverageReward()
        return currentAvg > self.bestAverageReward and currentAvg > threshold

def saveModel(model, optimizer, step: int, config: A3CConfig, modelType: str = "regular"):
    """保存模型"""
    if modelType == "best":
        modelPath = f"./A3CModels/a3c_model_{config.environmentName.replace('/', '_')}_best.pth"
    elif modelType == "checkpoint":
        modelPath = f"./A3CModels/a3c_model_{config.environmentName.replace('/', '_')}_checkpoint_step_{step}.pth"
    else:
        modelPath = f"./A3CModels/a3c_model_{config.environmentName.replace('/', '_')}_step_{step}.pth"

    os.makedirs(os.path.dirname(modelPath), exist_ok=True)

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': step,
        'config': config.__dict__
    }, modelPath)

    logger.info(f"模型已保存到: {modelPath}")
    return modelPath

def saveTrainingState(trainingState: TrainingState, config: A3CConfig):
    """保存训练状态"""
    statePath = f"./A3CModels/training_state_{config.environmentName.replace('/', '_')}.json"
    stateData = {
        'bestAverageReward': trainingState.bestAverageReward,
        'currentStep': trainingState.currentStep,
        'episodeRewards': trainingState.episodeRewards
    }

    with open(statePath, 'w') as f:
        json.dump(stateData, f)

    logger.info(f"训练状态已保存到: {statePath}")

def loadTrainingState(config: A3CConfig) -> TrainingState:
    """加载训练状态"""
    statePath = f"./A3CModels/training_state_{config.environmentName.replace('/', '_')}.json"
    trainingState = TrainingState()

    if os.path.exists(statePath):
        with open(statePath, 'r') as f:
            stateData = json.load(f)
            trainingState.bestAverageReward = stateData['bestAverageReward']
            trainingState.currentStep = stateData['currentStep']
            trainingState.episodeRewards = stateData['episodeRewards']
        logger.info(f"训练状态已从 {statePath} 加载")

    return trainingState

def trainA3C(config: A3CConfig, resumeFromCheckpoint: str = None):
    """训练A3C算法"""
    logger.info("开始A3C训练")

    # 获取动作数量
    numActions = getNumActions(config)
    logger.info(f"动作空间大小: {numActions}")

    # 创建共享模型
    sharedModel = ActorCriticNetwork(4, numActions).to(device)
    sharedModel.share_memory()

    # 创建优化器
    optimizer = torch.optim.RMSprop(
        sharedModel.parameters(),
        lr=config.learningRate,
        alpha=0.99,
        eps=1e-5,
        weight_decay=1e-4  # 添加权重衰减
    )

    # 加载训练状态
    trainingState = loadTrainingState(config)

    # 从检查点恢复
    startStep = 0
    if resumeFromCheckpoint and os.path.exists(resumeFromCheckpoint):
        checkpoint = torch.load(resumeFromCheckpoint)
        sharedModel.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        startStep = checkpoint['step']
        logger.info(f"从检查点恢复: {resumeFromCheckpoint}, 步数: {startStep}")

    # 全局计数器
    globalCounter = mp.Value('i', startStep)

    # 训练队列和最佳奖励共享变量
    trainingQueue = mp.Queue()
    bestReward = mp.Value('d', trainingState.bestAverageReward)

    # 启动监控进程
    monitorProcess = mp.Process(target=trainingMonitor, args=(trainingQueue, config, bestReward))
    monitorProcess.start()

    # 创建工作进程
    processes = []
    for i in range(config.numProcesses):
        process = mp.Process(
            target=worker,
            args=(i, sharedModel, optimizer, config, globalCounter, trainingQueue, device, bestReward)
        )
        process.start()
        processes.append(process)
        time.sleep(0.5)  # 避免同时创建过多进程

    logger.info(f"启动 {len(processes)} 个训练进程")

    # 等待训练完成
    try:
        lastCheckpointStep = startStep
        lastSaveStep = startStep

        while globalCounter.value < config.trainingSteps:
            time.sleep(5)
            currentStep = globalCounter.value

            # 定期日志
            if currentStep % config.logInterval == 0:
                progress = currentStep / config.trainingSteps * 100
                logger.info(f"训练进度: {currentStep}/{config.trainingSteps} ({progress:.1f}%)")

                # 更新训练状态
                trainingState.currentStep = currentStep
                saveTrainingState(trainingState, config)

            # 保存检查点
            if currentStep - lastCheckpointStep >= config.saveCheckpointFrequency:
                saveModel(sharedModel, optimizer, currentStep, config, "checkpoint")
                lastCheckpointStep = currentStep

            # 定期保存模型
            if currentStep - lastSaveStep >= config.saveModelFrequency:
                saveModel(sharedModel, optimizer, currentStep, config)
                lastSaveStep = currentStep

    except KeyboardInterrupt:
        logger.info("收到中断信号，停止训练")

    # 终止进程
    trainingQueue.put(None)  # 发送终止信号给监控进程

    for process in processes:
        process.terminate()
        process.join(timeout=5)

    monitorProcess.terminate()
    monitorProcess.join(timeout=5)

    # 保存最终模型
    saveModel(sharedModel, optimizer, globalCounter.value, config)

    logger.info("A3C训练完成")

def testTrainedModel(config: A3CConfig, modelPath: str, numEpisodes: int = 5):
    """测试训练好的模型"""
    logger.info(f"开始测试模型: {modelPath}")

    # 加载模型
    checkpoint = torch.load(modelPath, map_location='cpu')
    trainedModel = ActorCriticNetwork(4, getNumActions(config)).to(device)
    trainedModel.load_state_dict(checkpoint['model_state_dict'])
    trainedModel.eval()

    # 创建环境
    from Environment import createEnvironment
    environment = createEnvironment(config)

    episodeRewards = []

    for episode in range(numEpisodes):
        state, _ = environment.reset()
        state = torch.FloatTensor(state).to(device)

        episodeReward = 0
        done = False

        while not done:
            # 选择动作
            with torch.no_grad():
                action, _, _ = trainedModel.getAction(state)

            # 执行动作
            nextState, reward, done, _ = environment.step(action)
            nextState = torch.FloatTensor(nextState).to(device)

            # 更新状态
            state = nextState
            episodeReward += reward

        episodeRewards.append(episodeReward)
        logger.info(f"测试回合 {episode + 1}: 奖励 = {episodeReward}")

    avgReward = np.mean(episodeRewards)
    logger.info(f"平均测试奖励: {avgReward:.2f}")

    environment.close()

    return avgReward