# 初始化DQN类
import torch
from GRL_Library.agent import AVDQN_agent  # 导入编写的GRL_agent
from GRL_Net.Pytorch_GRL import torch_GRL  # 导入编写的pytorch下的网络
from GRL_Library.common import replay_buffer, explorer  # 导入编写的GRL库


def Create_DQN(num_HVs, num_AVs, param):
    # 初始化GRL网络
    N = num_HVs + num_AVs
    F = 10 + param['n_lanes']
    A = 24
    Net = torch_GRL(N, F, A)
    # 初始化优化器
    optimizer = torch.optim.Adam(Net.parameters(), lr=0.00075)  # 需要定义学习率
    # 定义replay_buffer
    replay_buffer_0 = replay_buffer.ReplayBuffer(size=10 ** 6)
    # 定义折扣因子
    gamma = 0.99
    # 定义智能体策略参数
    explorer_0 = explorer.LinearDecayEpsilonGreedy(start_epsilon=0.5, end_epsilon=0.01, decay_step=500000)

    # 初始化DQN类
    warmup = 1000  # 设置warmup步长
    GRL_DQN = AVDQN_agent.DQN(
        Net,  # 模型采用的网络
        optimizer,  # 模型采用的优化器
        explorer_0,  # 策略探索模型
        replay_buffer_0,  # 经验池
        gamma,  # 折扣率
        batch_size=128,  # 定义batch_size
        warmup_step=warmup,  # 定义开始更新的步长
        update_interval=10,  # 当前网络更新步长间隔
        target_update_interval=2000,  # 目标网络更新步长间隔
        target_update_method='soft',  # 目标网络更新方法
        soft_update_tau=0.1,  # 若soft_update，定义更新权重
        n_steps=5,  # multi-steps learning学习步长
        model_name="AVDQN_model" , # 模型命名
        num_target_values=10
    )


    return Net, GRL_DQN
