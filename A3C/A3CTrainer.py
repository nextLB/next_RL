"""
A3C训练器主模块
"""

import torch
import torch.multiprocessing as mp
import time
import os
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

def saveModel(model, optimizer, step: int, config: A3CConfig):
    """保存模型"""
    modelPath = f"./A3C_models/a3c_model_{config.environmentName.replace('/', '_')}_step_{step}.pth"
    os.makedirs(os.path.dirname(modelPath), exist_ok=True)

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': step,
        'config': config.__dict__
    }, modelPath)

    logger.info(f"模型已保存到: {modelPath}")

def trainA3C(config: A3CConfig):
    """训练A3C算法"""
    logger.info("开始A3C训练")

    # 获取动作数量
    numActions = getNumActions(config)
    logger.info(f"动作空间大小: {numActions}")

    # 创建共享模型
    sharedModel = ActorCriticNetwork(4, numActions).to(device)
    sharedModel.share_memory()

    # 创建优化器
    optimizer = torch.optim.Adam(sharedModel.parameters(), lr=config.learningRate)

    # 全局计数器
    globalCounter = mp.Value('i', 0)

    # 训练队列
    trainingQueue = mp.Queue()

    # 启动监控进程
    monitorProcess = mp.Process(target=trainingMonitor, args=(trainingQueue, config))
    monitorProcess.start()

    # 创建工作进程
    processes = []
    for i in range(config.numProcesses):
        process = mp.Process(
            target=worker,
            args=(i, sharedModel, optimizer, config, globalCounter, trainingQueue, device)
        )
        process.start()
        processes.append(process)
        time.sleep(1)  # 避免同时创建过多进程

    logger.info(f"启动 {len(processes)} 个训练进程")

    # 等待训练完成
    try:
        while globalCounter.value < config.trainingSteps:
            time.sleep(5)
            currentStep = globalCounter.value
            if currentStep % config.logInterval == 0:
                logger.info(f"训练进度: {currentStep}/{config.trainingSteps} ({currentStep/config.trainingSteps*100:.1f}%)")

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
    checkpoint = torch.load(modelPath)
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