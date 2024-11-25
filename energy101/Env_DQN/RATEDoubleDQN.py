"""Contains an experiment class for running simulations."""
import datetime
import logging

# from GRL_Experiment.Exp_HighwayRamps.registry import make_create_env




# 初始化DQN类
import torch
from GRL_Library.common import replay_buffer, explorer ,prioritized_replay_buffer # 导入编写的GRL库
from GRL_Library.agent import RATEDoubleDQN_agent  # 导入编写的GRL_agent
from GRL_Net.Pytorch_GRL import torch_GRL  # 导入编写的pytorch下的网络

def Create_DQN(num_HVs, num_AVs, param):
        # 初始化GRL网络
        N = num_HVs + num_AVs
        F = 10 + param['n_lanes']
        A = 24
        Net = torch_GRL(N, F, A)
        from GRL_Net.NoisyNet import noisy_chain
        noisy_sigma = 0.2  # 定义噪声参数
        noisy_chain.to_factorized_noisy(Net, sigma_scale=noisy_sigma)
        # 初始化优化器
        optimizer = torch.optim.Adam(Net.parameters(), lr=0.00025)  # 需要定义学习率
        # 定义replay_buffer
        # replay_buffer_0 = replay_buffer.ReplayBuffer(size=10 ** 6)
        replay_buffer_0 = prioritized_replay_buffer.PrioritizedReplayBuffer(capacity=2 ** 14,
                                                                            alpha=0.6,
                                                                            beta=0.4,
                                                                            beta_step=0.0001,
                                                                            epsilon=1e-4)
        # replay_buffer_0=RankBasedReplay(size=2**15, alpha=0.6)
        # 定义折扣因子
        gamma = 0.99
        # 定义智能体策略参数
        # explorer_0 = explorer.LinearDecayEpsilonGreedy(start_epsilon=0.5, end_epsilon=0.1, decay_step=400000)
        explorer_0 = explorer.LinearDecayEpsilonGreedy(start_epsilon=0.9, end_epsilon=0.025, decay_step=150000)
        # 打印网络初始参数
        # for parameters in GRL_Net.parameters():
        #     print("param:", parameters)

        # 初始化DQN类
        warmup = 5000  # 设置warmup步长
        GRL_DQN = RATEDoubleDQN_agent.DoubleDQN(
            Net,  # 模型采用的网络
            optimizer,  # 模型采用的优化器
            explorer_0,   # 策略探索模型
            replay_buffer_0,  # 经验池
            gamma,  # 折扣率
            batch_size=64,  # 定义batch_size
            warmup_step=warmup,  # 定义开始更新的步长
            update_interval=50,  # 当前网络更新步长间隔
            target_update_interval=4000,  # 目标网络更新步长间隔
            target_update_method='soft',  # 目标网络更新方法
            soft_update_tau=0.075,  # 若soft_update，定义更新权重
            n_steps=3,  # multi-steps learning学习步长
            model_name="AVDoubleDQN_model" , # 模型命名
            num_target_values = 10,
            rate=0.9
        )


        return Net, GRL_DQN
