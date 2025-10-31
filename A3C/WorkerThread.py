"""
工作进程模块
"""

import torch
import torch.multiprocessing as mp
import numpy as np
from Config import A3CConfig
from Environment import createEnvironment
from A3CAgent import ActorCriticNetwork
from Loss import computeA3CLoss
from Log import logger


def worker(processId: int, sharedModel, optimizer,
           config: A3CConfig, globalCounter: mp.Value, trainingQueue: mp.Queue,
           device, bestReward: mp.Value = None):
    """工作进程函数"""
    try:
        logger.info(f"进程 {processId} 启动")

        # 设置随机种子
        torch.manual_seed(processId)
        np.random.seed(processId)

        # 创建本地模型
        from Environment import getNumActions
        localModel = ActorCriticNetwork(4, getNumActions(config)).to(device)

        # 创建环境
        environment = createEnvironment(config)

        # 训练状态
        episodeReward = 0
        episodeLength = 0
        totalEpisodes = 0

        # 初始化状态
        state, _ = environment.reset()
        state = torch.FloatTensor(state).to(device)

        logger.info(f"进程 {processId} 初始化完成，开始训练")

        while globalCounter.value < config.trainingSteps:
            # 同步模型参数
            localModel.load_state_dict(sharedModel.state_dict())

            states, actions, rewards = [], [], []

            # 收集经验
            for step in range(config.tMax):
                # 选择动作
                action, policyLogits, value = localModel.getAction(state)

                # 执行动作
                nextState, reward, done, _ = environment.step(action)
                nextStateTensor = torch.FloatTensor(nextState).to(device)

                # 存储经验
                states.append(state.clone())
                actions.append(action)
                rewards.append(reward)

                # 更新状态
                state = nextStateTensor
                episodeReward += reward
                episodeLength += 1

                # 更新全局计数器
                with globalCounter.get_lock():
                    globalCounter.value += 1
                    currentStep = globalCounter.value

                # 如果episode结束
                if done:
                    totalEpisodes += 1

                    # 发送训练统计信息
                    trainingData = {
                        'processId': processId,
                        'episode': totalEpisodes,
                        'reward': episodeReward,
                        'step': currentStep
                    }

                    # 检查是否是最佳模型
                    if bestReward is not None and episodeReward > bestReward.value:
                        with bestReward.get_lock():
                            if episodeReward > bestReward.value:
                                bestReward.value = episodeReward
                                trainingData['isBest'] = True

                    trainingQueue.put(trainingData)

                    # 重置环境
                    state, _ = environment.reset()
                    state = torch.FloatTensor(state).to(device)
                    episodeReward = 0
                    episodeLength = 0
                    break

            # 计算损失 - 传递nextState用于bootstrap
            nextStateForBootstrap = state if not done else torch.FloatTensor(environment.reset()[0]).to(device)
            loss = computeA3CLoss(localModel, states, actions, rewards, done, nextStateForBootstrap, config, device)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(localModel.parameters(), config.maxGradientNorm)

            # 将梯度添加到共享模型
            for localParam, sharedParam in zip(localModel.parameters(), sharedModel.parameters()):
                if localParam.grad is not None:
                    if sharedParam.grad is None:
                        sharedParam.grad = localParam.grad.clone()
                    else:
                        sharedParam.grad += localParam.grad.clone()

            # 更新全局模型
            optimizer.step()
            sharedModel.zero_grad()

        environment.close()
        logger.info(f"进程 {processId} 训练完成")

    except Exception as e:
        logger.error(f"进程 {processId} 发生错误: {e}")
        import traceback
        logger.error(traceback.format_exc())