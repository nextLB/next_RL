"""
    依照本目录的paper目录下的DQNNaturePaper篇文章进行训练的，与原论文在架构方面有一些改动的
"""

# V1.0  2025.10.28      --- by next, 初步实现了使用Renet深度学习模型架构的DQN强化学习模型的搭建与训练等


import torch


# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




def main():
    print(f"使用设备: {device}")



if __name__ == '__main__':
    main()


