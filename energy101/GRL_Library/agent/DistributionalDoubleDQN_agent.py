"""
    该函数用来定义DistributionalDQN-agent
"""

import torch
import numpy as np
import torch.autograd as autograd
import torch.nn.functional as F
import copy
import collections
import GRL_Library.agent.DQN_agent as DQN

# CUDA设置
USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() \
    if USE_CUDA else autograd.Variable(*args, **kwargs)


# 通过继承的方式创建DistributionalDoubleDQN类
class DistributionalDoubleDQN(DQN.DQN):
    """
        定义DistributionalDoubleDQN类，继承DQN类的所有特性

        额外参数说明:
        --------
        V_min: 分布价值区间最小值
        V_max: 分布价值区间最大值
        n_atoms: 分布采样数量
    """

    def __init__(self, model, optimizer, explorer, replay_buffer, gamma, batch_size, warmup_step,
                 update_interval, target_update_interval, target_update_method,
                 soft_update_tau, n_steps, V_min, V_max, n_atoms, model_name):
        super().__init__(model, optimizer, explorer, replay_buffer,
                         gamma, batch_size, warmup_step, update_interval,
                         target_update_interval, target_update_method,
                         soft_update_tau, n_steps, model_name)
        # 参数赋值
        self.V_min = V_min
        self.V_max = V_max
        self.n_atoms = n_atoms

        # 计算条件分布support
        self.support = torch.linspace(self.V_min, self.V_max, self.n_atoms).to(self.device)

    def compute_loss(self, data_batch):
        """
           <损失计算函数>
           用来计算预测值和目标值的损失，为后续反向传播求导作基础

           参数说明:
           --------
           data_batch: 从经验池中采样的用来训练的数据
        """
        # 初始化loss矩阵
        loss = []
        # 初始化TD_error矩阵
        TD_error = []

        # 计算z_delta
        delta_z = float(self.V_max - self.V_min) / (self.n_atoms - 1)

        # 获取智能体数量
        num_agents = self.model.num_agents

        # 获取智能体编号索引
        index_dist = [i for i in range(num_agents)]

        # 按照数据存储顺序，从每一个sample的中提取数据
        for elem in data_batch:
            # 获取elem中的具体元素
            state, action, reward, next_state, done = elem

            # 针对DoubleDQN进行操作
            # 计算动作及分布
            next_action = self.model(next_state).argmax(1)
            next_dist = self.target_model.dist(next_state)
            # 根据动作获取distribution的具体值
            next_dist = next_dist[index_dist, next_action, :]  # 根据索引以及动作获取对应的q值分布

            # 计算distribution相关参数
            t_z = reward + self.gamma * self.support * (1 - done)
            t_z = t_z.clamp(min=self.V_min, max=self.V_max)
            b = (t_z - self.V_min) / delta_z
            l = b.floor().long()
            u = b.ceil().long()

            offset = (torch.linspace(0, (num_agents - 1) * self.n_atoms, num_agents).
                      long().unsqueeze(1).expand(num_agents, self.n_atoms).to(self.device))

            proj_dist = torch.zeros(next_dist.size(), device=self.device)
            proj_dist.view(-1).index_add_ \
                (0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1))
            proj_dist.view(-1).index_add_ \
                (0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1))

            # 计算预测q值分布
            dist = self.model.dist(state)

            # 计算KL散度作为损失
            log_p = torch.log(dist[index_dist, action, :])
            loss_sample = -(proj_dist * log_p).sum(1).mean()

            # 将当前sample的损失计入总损失
            loss.append(loss_sample)

        # 求loss的平均值
        loss = torch.stack(loss)

        return loss

    def compute_loss_multisteps(self, data_batch, n_steps):
        """
           <多步学习损失计算函数>
           用来计算预测值和目标值的损失，为后续反向传播求导作基础

           参数说明:
           --------
           data_batch: 从经验池中采样的用来训练的数据
           n_steps: 多步学习步长间隔
        """
        # 初始化loss矩阵
        loss = []
        # 初始化TD_error矩阵
        TD_error = []

        # 计算z_delta
        delta_z = float(self.V_max - self.V_min) / (self.n_atoms - 1)

        # 获取智能体数量
        num_agents = self.model.num_agents

        # 获取智能体编号索引
        index_dist = [i for i in range(num_agents)]

        # 按照数据存储顺序，从每一个sample的中提取数据
        # 对于多步学习，每一个data_batch包含若干连续的样本
        for elem in data_batch:
            # 取n_steps和elem长度的较小值，以防止n步连续采样超出索引
            n_steps = min(self.n_steps, len(elem))

            # ------计算q_target的动作及分布------ #
            # 获取第n个sample
            state, action, reward, next_state, done = elem[n_steps - 1]
            # 计算动作及分布
            next_action = self.model(next_state).argmax(1)
            next_dist = self.target_model.dist(next_state)
            # 根据动作获取distribution的具体值
            next_dist = next_dist[index_dist, next_action, :]  # 根据索引以及动作获取对应的q值分布

            # ------计算奖励------ #
            # 获取奖励值
            reward = [i[2] for i in elem]
            # 计算折扣系数
            n_step_scaling = [self.gamma ** i for i in range(n_steps)]
            # 将奖励值与折扣系数对应系数相乘，计算奖励矩阵
            R = np.multiply(reward, n_step_scaling)
            # 奖励求和
            R = np.sum(R)

            # ------计算distribution相关参数------ #
            t_z = R + (self.gamma ** self.n_steps) * self.support * (1 - done)
            t_z = t_z.clamp(min=self.V_min, max=self.V_max)
            b = (t_z - self.V_min) / delta_z
            l = b.floor().long()
            u = b.ceil().long()

            offset = (torch.linspace(0, (num_agents - 1) * self.n_atoms, num_agents).
                      long().unsqueeze(1).expand(num_agents, self.n_atoms).to(self.device))

            proj_dist = torch.zeros(next_dist.size(), device=self.device)
            proj_dist.view(-1).index_add_ \
                (0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1))
            proj_dist.view(-1).index_add_ \
                (0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1))

            # ------计算q_predict分布------ #
            state, action, reward, next_state, done = elem[0]
            dist = self.model.dist(state)

            # ------计算loss------ #
            # 计算KL散度作为损失
            log_p = torch.log(dist[index_dist, action, :])
            loss_sample = -(proj_dist * log_p).sum(1).mean()
            # 将当前sample的损失计入总损失
            loss.append(loss_sample)

        # 求loss的平均值
        loss = torch.stack(loss)

        return loss
