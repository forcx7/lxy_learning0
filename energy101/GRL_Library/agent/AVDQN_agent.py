"""
    该函数用来定义DQN-agent
"""

import torch
import numpy as np
import torch.autograd as autograd
import torch.nn.functional as F
import copy
import collections
from GRL_Library.common.prioritized_replay_buffer import PrioritizedReplayBuffer

# CUDA设置
USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() \
    if USE_CUDA else autograd.Variable(*args, **kwargs)
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


class DQN(object):
    """
        定义DQN类

        参数说明:
        --------
        model: agent中采用的神经网络模型
        optimizer: 训练模型的优化器
        explorer: 探索及动作选择策略
        replay_buffer: 经验回放池
        gamma: 折扣系数
        batch_size: batch存储长度
        warmup_step: 随机探索步长
        update_interval: 当前网络更新间隔
        target_update_interval: 目标网络更新间隔
        target_update_method: 目标网络更新方式(hard or soft)
        soft_update_tau: 目标网络soft更新参数
        n_steps: Time Difference更新步长(整数，1为单步更新，其余为Multi-step learning)
        model_name: 模型名称(用来保存和读取)
    """

    def __init__(self,
                 model,
                 optimizer,
                 explorer,
                 replay_buffer,
                 gamma,
                 batch_size,
                 warmup_step,
                 update_interval,
                 target_update_interval,
                 target_update_method,
                 soft_update_tau,
                 n_steps,
                 model_name,
                 num_target_values ):
        # 参数赋值
        self.model = model  # 当前网络
        self.optimizer = optimizer
        self.explorer = explorer
        self.replay_buffer = replay_buffer
        self.gamma = gamma
        self.batch_size = batch_size
        self.warmup_step = warmup_step
        self.update_interval = update_interval
        self.target_update_interval = target_update_interval
        self.target_update_method = target_update_method
        self.soft_update_tau = soft_update_tau
        self.n_steps = n_steps
        self.model_name = model_name
        self.num_target_values=num_target_values
        # GPU设置
        if USE_CUDA:
            GPU_num = torch.cuda.current_device()
            self.device = torch.device("cuda:{}".format(GPU_num))
            self.model.to(self.device)

        # 设置目标网络
        self.target_model = copy.deepcopy(model)
        self.target_q_values = {}
        for i in range(self.num_target_values):
            self.target_q_values[i] = self.target_model.type(dtype)
            # print(self.target_q_values)
        self.num_active_target = 1
        # 设置当前仿真步长计数器
        self.time_counter = 0

        # 设置训练数据记录矩阵
        self.loss_record = collections.deque(maxlen=100)
        self.q_record = collections.deque(maxlen=100)

    def store_transition(self, state, action, reward, next_state, done):
        """
           <经验存储函数>
           用来存储agent学习过程中的经验数据

           参数说明:
           --------
           state: 当前时刻状态
           action: 当前时刻动作
           reward：执行当前动作后获得的奖励
           next_state: 执行当前动作后的下一个状态
           done: 是否终止
        """
        # 调用replay_buffer中保存数据的函数
        self.replay_buffer.add(state, action, reward, next_state, done)

    def sample_memory(self):
        """
           <经验采样函数>
           用来从agent学习过程中的经验数据中进行采样
        """
        # 调用replay_buffer中的采样函数
        data_sample = self.replay_buffer.sample(self.batch_size, self.n_steps)
        return data_sample

    def choose_action(self, observation):
        """
           <训练动作选择函数>
           针对训练过程，根据环境观测生成agent的动作

           参数说明:
           --------
           observation: 智能体所在环境观测
        """
        # 生成动作
        action = self.model(observation)
        action = torch.argmax(action, dim=1)  # 取每一列最大q值对应的动作(greedy_action)
        action = self.explorer.generate_action(action)
        return action

    def test_action(self, observation):
        """
           <测试动作选择函数>
           针对测试过程，根据环境观测生成agent的动作，直接选择得分最高动作

           参数说明:
           --------
           observation: 智能体所在环境观测
        """
        # 生成动作
        action = self.model(observation)
        action = torch.argmax(action, dim=1)  # 取每一列最大q值对应的动作
        return action

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

            # 计算目标网络目标值
            q_a_values_sum = torch.FloatTensor(24).zero_()
            q_a_values_sum = q_a_values_sum.cuda()

            # print q_a_values_sum

            for i in range(self.num_active_target):
                # print target_q_values[0](obs_tp1_batch)
                q_a_values_sum = torch.add(q_a_values_sum, self.target_q_values[i](next_state).data)

            q_a_values_sum = Variable(q_a_values_sum)
            q_a_vales_tp1 = q_a_values_sum.detach().max(1)[0]
            # q_next = self.target_model(next_state)
            # q_next = q_next.max(dim=1)[0]
            q_target = reward + self.gamma / self.num_active_target * q_a_vales_tp1 * (1 - done)

            # 计算TD_error(时间差分)
            TD_error_sample = torch.abs(q_target - q_predict)
            TD_error_sample = torch.mean(TD_error_sample)
            # 将当前sample的TD_error计入总TD_error
            TD_error.append(TD_error_sample)

            # 计算损失
            loss_sample = F.smooth_l1_loss(q_predict, q_target)
            # 将当前sample的损失计入总损失
            loss.append(loss_sample)

        # 进一步处理TD_error
        TD_error = torch.stack(TD_error)

        # 将sample中不同样本的loss合并为tensor
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
            # 取n_steps和elem长度的较小值，以防止n步连续采样超出索引
            n_steps = min(self.n_steps, len(elem))

            # ------计算q_predict的值------ #
            # 获取第一个sample
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
            # 计算n步max_Q的值
            q_a_values_sum = torch.FloatTensor(24).zero_()
            q_a_values_sum = q_a_values_sum.cuda()

            # print q_a_values_sum

            for i in range(self.num_active_target):
                # print target_q_values[0](obs_tp1_batch)
                q_a_values_sum = torch.add(q_a_values_sum, self.target_q_values[i](next_state).data)

            q_a_values_sum = Variable(q_a_values_sum)
            q_a_vales_tp1 = q_a_values_sum.detach().max(1)[0]
            # q_next = self.target_model(next_state)
            # q_next = q_next.max(dim=1)[0]
            # q_target = reward + self.gamma / self.num_active_target * q_a_vales_tp1 * (1 - done)

            # 计算TD_error(时间差分)
            # q_next = self.target_model(next_state)
            # q_next = q_next.max(dim=1)[0]
            # 计算目标值
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

        # 进一步处理TD_error，可根据需要决定是否return这一项
        TD_error = torch.stack(TD_error)

        # 将sample中不同样本的loss合并为tensor
        loss = torch.stack(loss)

        return loss

    def loss_process(self, loss, weight):
        """
           <损失后处理函数>
           不同算法对损失数据的维度需求不同，故编写此函数进行统一处理

           参数说明:
           --------
           loss: 通过sample计算所得的损失[1, self.batch_size]
        """
        # 根据权重对loss进行计算
        weight = torch.as_tensor(weight, dtype=torch.float32, device=self.device)
        loss = torch.mean(loss * weight)

        return loss

    def synchronize_target(self):
        """
           <目标网络同步函数>
           用来同步目标网络（target_network）
        """
        if self.target_update_method == "hard":
            self.hard_update()
        elif self.target_update_method == "soft":
            self.soft_update()
        else:
            raise ValueError("Unknown target update method")

    def hard_update(self):
        """
           <目标网络hard更新函数>
           采用hard_update的方法同步目标网络（target_network）
        """
        if self.num_active_target >= self.num_target_values:
            self.num_active_target = self.num_target_values
        print
        "Update Q Values : Active {} Q values".format(self.num_active_target)
        for i in range(self.num_active_target - 1, 0, -1):
            self.target_q_values[i].load_state_dict(self.target_q_values[i - 1].state_dict())
        self.target_q_values[0].load_state_dict(self.model.state_dict())
        self.target_model.load_state_dict(self.model.state_dict())

    def soft_update(self):
        """
           <目标网络soft更新函数>
           采用soft_update的方法同步目标网络（target_network）
        """
        # 必须要定义正确的soft更新参数
        assert 0.0 < self.soft_update_tau < 1.0
        if self.num_active_target >= self.num_target_values:
            self.num_active_target = self.num_target_values
        print
        "Update Q Values : Active {} Q values".format(self.num_active_target)
        # 进行参数更新
        for i in range(self.num_active_target - 1, 0, -1):
            self.target_q_values[i].load_state_dict(self.target_q_values[i - 1].state_dict())

        for target_param, target_param1, source_param in zip(self.target_q_values[0].parameters(),
                                              self.target_model.parameters(),
                                              self.model.parameters()):
            # self.target_q_values[0].load_state_dict(self.model.state_dict())
            # self.target_model.load_state_dict(self.model.state_dict())
            target_param1.data.copy_((1 - self.soft_update_tau) *
                                    target_param1.data + self.soft_update_tau * source_param.data)
            target_param.data.copy_((1 - self.soft_update_tau) *
                                    target_param.data + self.soft_update_tau * source_param.data)

    def learn(self):
        """
           <策略更新函数>
           用来实现agent的学习过程
        """
        # ------如果处于warmup阶段，或没达到更新步长时，则直接return------ #
        if (self.time_counter <= self.warmup_step) or \
                (self.time_counter % self.update_interval != 0):
            self.time_counter += 1
            return

        # ------计算损失------ #
        # 经验池采样，samples包括权重和索引，data_sample为具体的采样数据
        samples, data_sample = self.sample_memory()

        # loss矩阵计算
        if self.n_steps == 1:  # 单步学习
            elementwise_loss = self.compute_loss(data_sample)
        else:  # 多步学习
            elementwise_loss = self.compute_loss_multisteps(data_sample, self.n_steps)

        # 如果是PrioritizedReplayBuffer，则在计算总损失之前更新priority
        if isinstance(self.replay_buffer, PrioritizedReplayBuffer):
            self.replay_buffer.update_priority(samples['indexes'], elementwise_loss)

        # 计算总损失
        loss = self.loss_process(elementwise_loss, samples['weights'])

        # 存储loss
        self.loss_record.append(float(loss.detach().cpu().numpy()))

        # 优化器操作
        self.optimizer.zero_grad()

        # 反向传播求导
        loss.backward()

        # 参数更新
        self.optimizer.step()

        # 判断是否更新目标网络
        if self.time_counter % self.target_update_interval == 0:
            self.num_active_target += 1
            self.synchronize_target()

        # 计数器加1
        self.time_counter += 1

    def get_statistics(self):
        """
           <训练数据获取函数>
           用来获取训练过程中的相关数据
        """
        loss_statistics = np.mean(self.loss_record) if self.loss_record else np.nan
        q_statistics = np.mean(np.absolute(self.q_record)) if self.q_record else np.nan
        return [loss_statistics, q_statistics]

    def save_model(self, save_path):
        """
           <模型保存函数>
           用来保存训练的模型
        """
        save_path = save_path + "/" + self.model_name + ".pt"
        torch.save(self.model, save_path)

    def load_model(self, load_path):
        """
           <模型读取函数>
           用来读取训练的模型
        """
        load_path = load_path + "/" + self.model_name + ".pt"
        self.model = torch.load(load_path)
