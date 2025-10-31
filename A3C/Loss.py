"""
经验缓存和损失计算模块
"""

import torch
import torch.nn.functional as F
from typing import List
from Config import A3CConfig

def computeA3CLoss(model, states: List[torch.Tensor], actions: List[int],
                   rewards: List[float], done: bool, config: A3CConfig, device) -> torch.Tensor:
    """计算A3C损失 - 修复形状问题"""
    # 确保所有状态都需要梯度
    statesTensor = torch.stack(states)

    # 重新计算策略和价值（确保有梯度）
    policyLogits, values = model(statesTensor)

    # 计算n步回报
    R = 0.0
    if not done:
        with torch.no_grad():
            # 计算最后一个状态的价值
            lastState = states[-1].unsqueeze(0)
            _, lastValue = model(lastState)
            R = lastValue.item()

    returns = []
    for r in reversed(rewards):
        R = r + config.discountFactor * R
        returns.insert(0, R)

    returnsTensor = torch.tensor(returns, dtype=torch.float32).to(device)

    # 修复形状问题 - 确保values和returnsTensor形状一致
    valuesFlat = values.squeeze()
    if valuesFlat.dim() == 0:  # 如果是标量
        valuesFlat = valuesFlat.unsqueeze(0)

    # 计算优势函数
    advantages = returnsTensor - valuesFlat.detach()

    # 策略损失
    logProbs = F.log_softmax(policyLogits, dim=-1)
    actionsTensor = torch.tensor(actions, dtype=torch.long).to(device)
    actionLogProbs = logProbs.gather(1, actionsTensor.unsqueeze(1)).squeeze()

    # 确保actionLogProbs和advantages形状一致
    if actionLogProbs.dim() == 0:
        actionLogProbs = actionLogProbs.unsqueeze(0)

    policyLoss = -(actionLogProbs * advantages).mean()

    # 价值损失 - 修复形状问题
    # 确保valuesFlat和returnsTensor形状完全一致
    if valuesFlat.shape != returnsTensor.shape:
        # 如果形状不匹配，调整valuesFlat的形状
        if valuesFlat.numel() == 1 and returnsTensor.numel() > 1:
            valuesFlat = valuesFlat.expand_as(returnsTensor)
        elif valuesFlat.numel() > 1 and returnsTensor.numel() == 1:
            returnsTensor = returnsTensor.expand_as(valuesFlat)

    valueLoss = F.mse_loss(valuesFlat, returnsTensor)

    # 熵正则化
    entropy = -(logProbs * torch.exp(logProbs)).sum(-1).mean()
    entropyLoss = -config.entropyCoefficient * entropy

    totalLoss = policyLoss + config.valueLossCoefficient * valueLoss + entropyLoss

    return totalLoss

