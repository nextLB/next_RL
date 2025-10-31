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
    if not states:
        return torch.tensor(0.0, device=device, requires_grad=True)

    statesTensor = torch.stack(states).to(device)
    policyLogits, values = model(statesTensor)

    # 计算n步回报
    R = 0.0
    if not done:
        with torch.no_grad():
            lastValue = model.getValue(nextState.unsqueeze(0))
            R = lastValue.item()

    returns = []
    for r in reversed(rewards):
        R = r + config.discountFactor * R
        returns.insert(0, R)

    returnsTensor = torch.tensor(returns, dtype=torch.float32).to(device)
    values = values.squeeze()

    if values.dim() == 0:
        values = values.unsqueeze(0)

    # 计算优势函数
    advantages = returnsTensor - values.detach()

    # 修复标准差计算 - 避免自由度问题
    advantages_mean = advantages.mean()

    # 只在有多个元素时计算标准差
    if len(advantages) > 1:
        # 使用更安全的标准差计算，避免自由度警告
        advantages_std = advantages.std(unbiased=False)  # 使用有偏估计器

        # 只有在标准差足够大时才进行标准化
        if advantages_std > 1e-6 and not torch.isnan(advantages_std):
            advantages = (advantages - advantages_mean) / (advantages_std + 1e-8)
        else:
            advantages = advantages - advantages_mean
    else:
        # 只有一个元素时，只进行中心化
        advantages = advantages - advantages_mean

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

    # 检查损失是否为NaN
    if torch.isnan(totalLoss):
        return torch.tensor(0.0, device=device, requires_grad=True)

    return totalLoss