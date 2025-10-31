"""
工作进程模块
"""

import torch
import torch.multiprocessing as mp
import numpy as np
import time
from typing import List
from Config import A3CConfig
from Environment import createEnvironment
from A3CAgent import ActorCriticNetwork
from Loss import computeA3CLoss
from Log import logger
import torch.nn.functional as F


def worker(processId: int, sharedModel, optimizer,
           config: A3CConfig, globalCounter: mp.Value, trainingQueue: mp.Queue, device):
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

        episodeCount = 0
        while globalCounter.value < config.trainingSteps:
            # 同步模型参数
            localModel.load_state_dict(sharedModel.state_dict())

            states, actions, rewards = [], [], []

            # 添加网络输出监控
            if processId == 0 and episodeCount % 50 == 0:
                with torch.no_grad():
                    testState = torch.FloatTensor(state).to(device)
                    policyLogits, value = localModel(testState.unsqueeze(0))
                    policy = F.softmax(policyLogits, dim=-1)
                    logger.info(
                        f"策略熵: {-(policy * torch.log(policy + 1e-8)).sum().item():.4f}, 价值估计: {value.item():.4f}")

            for step in range(config.tMax):
                # 选择动作
                action, _, _ = localModel.getAction(state)

                # 执行动作
                nextState, reward, done, _ = environment.step(action)
                nextState = torch.FloatTensor(nextState).to(device)

                # 存储经验
                states.append(state.clone())
                actions.append(action)
                rewards.append(reward)

                # 更新状态
                state = nextState
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
                    if processId == 0 and totalEpisodes % 5 == 0:
                        trainingQueue.put({
                            'processId': processId,
                            'episode': totalEpisodes,
                            'reward': episodeReward,
                            'step': currentStep
                        })

                    # 重置环境
                    state, _ = environment.reset()
                    state = torch.FloatTensor(state).to(device)
                    episodeReward = 0
                    episodeLength = 0
                    break

            # 计算损失
            loss = computeA3CLoss(localModel, states, actions, rewards, done, config, device)

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

            # 定期保存模型
            if processId == 0 and currentStep % config.saveModelFrequency == 0:
                from A3CTrainer import saveModel
                saveModel(sharedModel, optimizer, currentStep, config)

        environment.close()
        logger.info(f"进程 {processId} 训练完成")

    except Exception as e:
        logger.error(f"进程 {processId} 发生错误: {e}")
        import traceback
        logger.error(traceback.format_exc())