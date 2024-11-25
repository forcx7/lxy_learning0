"""
    该函数用来定义DoubleDQN-agent
"""

import torch
import numpy as np
import torch.autograd as autograd
import torch.nn.functional as F
import copy
import collections
import GRL_Library.agent.RATEDQN_agent as RATEDQN

# CUDA设置
USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() \
    if USE_CUDA else autograd.Variable(*args, **kwargs)


# 通过继承的方式创建DoubleDQN类
class DoubleDQN(RATEDQN.DQN):
    """
        定义DoubleDQN类，继承DQN类的所有特性
    """

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

        # 按照数据存储顺序，从每一个sample的中提取数据
        for elem in data_batch:
            state, action, reward, next_state, done = elem
            action = torch.as_tensor(action, dtype=torch.long, device=self.device)

            # 计算当前网络预测值
            q_predict = self.model(state)  # 计算状态S的Q表
            q_predict = q_predict.gather(1, action.unsqueeze(1)).squeeze(1)  # 根据action选择Q预测值

            # 储存q_predict
            q_predict_save = q_predict.detach().cpu().numpy().reshape(len(q_predict), 1)
            data_useful = np.any(q_predict_save, axis=1)
            self.q_record.append(q_predict_save / (data_useful.sum() + 1))

            # 计算当前网络在状态S+1的动作值
            q_evaluation = self.model(next_state)
            action_evaluation = torch.argmax(q_evaluation, dim=1)

            # ------目标值计算------ #
            # 计算目标网络在状态S+1的q值
            q_a_values_sum = torch.FloatTensor(24).zero_()
            q_a_values_sum = q_a_values_sum.cuda()

            # print q_a_values_sum

            for i in range(self.num_active_target):
                # print target_q_values[0](obs_tp1_batch)
                q_a_values_sum = torch.add(q_a_values_sum, self.target_q_values[i](next_state).data* self.rate**(i+1))

            q_a_values_sum = Variable(q_a_values_sum)
            q_a_vales_tp1 = q_a_values_sum.detach().max(1)[0]
            # q_next = self.target_model(next_state)
            # q_next = q_next.max(dim=1)[0]
            q_target = reward + self.gamma / self.num_active_target * q_a_vales_tp1 * (1 - done)

            # 计算TD_error(时间差分)
            q_next = self.target_model(next_state)
            # 根据评估的action选择动作
            # q_next = q_next.gather(1, action_evaluation.unsqueeze(1)).squeeze(1)
            # # 计算目标值
            # q_target = reward + self.gamma * q_next * (1 - done)

            # 计算损失
            loss_sample = F.smooth_l1_loss(q_predict, q_target)

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

        # 按照数据存储顺序，从每一个sample的中提取数据
        # 对于多步学习，每一个data_batch包含若干连续的样本
        for elem in data_batch:
            # ------计算q_predict的值------ #
            # 获取第一个sample
            n_steps = min(self.n_steps, len(elem))
            state, action, reward, next_state, done = elem[0]
            action = torch.as_tensor(action, dtype=torch.long, device=self.device)
            # 计算当前网络预测值
            q_predict = self.model(state)  # 计算状态S的Q表
            q_predict = q_predict.gather(1, action.unsqueeze(1)).squeeze(1)  # 根据action选择Q预测值
            # 储存q_predict
            q_predict_save = q_predict.detach().cpu().numpy().reshape(len(q_predict), 1)
            data_useful = np.any(q_predict_save, axis=1)
            self.q_record.append(q_predict_save / (data_useful.sum() + 1))

            # ------计算奖励------ #
            # 获取奖励值
            reward = [i[2] for i in elem]
            # 计算折扣系数
            n_step_scaling = [self.gamma ** i for i in range(n_steps)]
            # 将奖励值与折扣系数对应系数相乘，计算奖励矩阵
            R = np.multiply(reward, n_step_scaling)
            # 奖励求和
            R = np.sum(R)

            # ------计算q_target------ #
            state, action, reward, next_state, done = elem[n_steps - 1]
            # 计算当前网络在n_steps状态S+1的q值，以及最大动作
            q_evaluation = self.model(next_state)
            action_evaluation = torch.argmax(q_evaluation, dim=1)
            # 计算目标网络在n_steps状态S+1的q值
            q_a_values_sum = torch.FloatTensor(24).zero_()
            q_a_values_sum = q_a_values_sum.cuda()

            # print q_a_values_sum

            for i in range(self.num_active_target):
                # print target_q_values[0](obs_tp1_batch)
                q_a_values_sum = torch.add(q_a_values_sum, self.target_q_values[i](next_state).data* self.rate**(i+1))

            q_a_values_sum = Variable(q_a_values_sum)
            q_a_vales_tp1 = q_a_values_sum.detach().max(1)[0]
            # q_next = self.target_model(next_state)
            # q_next = q_next.max(dim=1)[0]
            # q_target = reward + self.gamma / self.num_active_target * q_a_vales_tp1 * (1 - done)

            # 计算TD_error(时间差分)
            # q_next = self.target_model(next_state)
            # # 根据评估的action选择动作
            # q_next = q_next.gather(1, action_evaluation.unsqueeze(1)).squeeze(1)
            # # 计算目标值
            q_target = R + (self.gamma ** n_steps)/ self.num_active_target * q_a_vales_tp1 * (1 - done)

            # ------计算n步TD_error------ #
            TD_error_sample = torch.abs(q_target - q_predict)
            TD_error_sample = torch.mean(TD_error_sample)
            # 将当前sample的TD_error计入总TD_error
            TD_error.append(TD_error_sample)

            # ------计算loss------ #
            loss_sample = F.smooth_l1_loss(q_predict, q_target)
            # 将当前sample的损失计入总损失
            loss.append(loss_sample)

        # 进一步处理TD_error
        TD_error = torch.stack(TD_error)

        # 求loss的平均值
        loss = torch.stack(loss)

        return loss
