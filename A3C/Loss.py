"""
经验缓存和损失计算模块
"""

import torch
import torch.nn.functional as F
from typing import List
from Config import A3CConfig

def computeA3CLoss(model, states: List[torch.Tensor], actions: List[int],
                   rewards: List[float], done: bool, nextState: torch.Tensor,
                   config: A3CConfig, device) -> torch.Tensor:
    """计算A3C损失"""
    # 确保所有状态都需要梯度
    statesTensor = torch.stack(states).to(device)

    # 重新计算策略和价值（确保有梯度）
    policyLogits, values = model(statesTensor)

    # 计算n步回报
    R = 0.0
    if not done:
        with torch.no_grad():
            # 计算最后一个状态的价值
            lastValue = model.getValue(nextState.unsqueeze(0))
            R = lastValue.item()
    else:
        R = 0.0

    # 计算n步回报
    returns = []
    for r in reversed(rewards):
        R = r + config.discountFactor * R
        returns.insert(0, R)

    returnsTensor = torch.tensor(returns, dtype=torch.float32).to(device)

    # 确保values形状正确
    values = values.squeeze()
    if values.dim() == 0:
        values = values.unsqueeze(0)

    # 计算优势函数
    advantages = returnsTensor - values.detach()

    # 策略损失
    logProbs = F.log_softmax(policyLogits, dim=-1)
    actionsTensor = torch.tensor(actions, dtype=torch.long).to(device)
    actionLogProbs = logProbs[range(len(actions)), actionsTensor]

    policyLoss = -(actionLogProbs * advantages).mean()

    # 价值损失
    valueLoss = F.mse_loss(values, returnsTensor)

    # 熵正则化
    probs = F.softmax(policyLogits, dim=-1)
    entropy = -(probs * logProbs).sum(-1).mean()
    entropyLoss = -config.entropyCoefficient * entropy

    totalLoss = policyLoss + config.valueLossCoefficient * valueLoss + entropyLoss

    return totalLoss