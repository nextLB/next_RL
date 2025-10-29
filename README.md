# next_RL

## 本项目是由next个人所实现的与强化学习相关的代码

## 环境参考

*项目的环境配置主要依赖与conda与python独立的虚拟环境*

    详细的依赖库与包的安装，可依照以下命令一步一步的进行安装

*创建独立的python虚拟环境*

    conda create -n next_test_RL python=3.11

*配置pytorch+cuda的环境*

    pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126

*配置gymnasium强化学习环境*

    pip install gymnasium==0.28.1
    pip install gym==0.26.2


*配置Atari环境支持*

    pip install AutoROM==0.4.2
    pip install "gymnasium[atari,accept-rom-license]"
    pip install "autorom[accept-rom-license]"


*配置图像处理库*

    pip install Pillow matplotlib
    pip install opencv-python
    pip install imageio
    pip install imageio-ffmpeg
    pip install moviepy

*配置系统监控库*

    pip install psutil

*安装其他必要依赖*

    pip install tqdm
    pip install pandas
    pip install requests
    pip install ale-py==0.8.1
    pip install protobuf

*tensorboard支持*

    pip install tensorboard

*安装grpcio和gym-notices*

    pip install grpcio==1.76.0
    pip install gym-notices==0.1.0




## V1.0版本

在V1.0版本中，初步实现了DQN的模型训练与搭建

### 对于DQN的详细实现，请见DQN文件夹下的程序

    V1.0.py是完整的基于"PongNoFrameskip-v4"的强化学习模型训练程序
    comprehend_V1_0.py是便于使用者逐步去理解V1.0版本的基于"PongNoFrameskip-v4"的强化学习模型训练程序


