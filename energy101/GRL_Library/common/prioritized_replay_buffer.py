"""
    该函数用来定义DRL中的prioritized_replay_buffer
"""
import random
import numpy as np


class PrioritizedReplayBuffer(object):
    def __init__(self, capacity, alpha, beta, beta_step, epsilon):
        """
            <构造函数>
            定义priority_replay_buffer类

            参数说明：
            ------
            capacity: replay_buffer的最大容量，当超过容量时，新的
                      数据会替换掉旧的数据
            alpha: 抽样概率误差指数
            beta: 重要性采样指数
            beta_step: 每次采样beta增加值(beta不超过1，且要控制更新速率)
            epsilon: 很小的值防止零priority
        """
        # capacity设置为2的n次方，以方便代码编写和调试
        assert capacity & (capacity - 1) == 0

        # 获取各项参数输入
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_step = beta_step
        self.epsilon = epsilon

        # 设置最大priority
        self.max_priority = 1.

        # 线段二叉树以求和并在一个范围内找到最小值
        self.priority_sum = [0 for _ in range(2 * self.capacity)]
        self.priority_min = [float('inf') for _ in range(2 * self.capacity)]

        self.next_index = 0  # 定义索引
        self.size = 0  # 定义buffer的大小
        self.buffer = []  # 定义buffer

    def add(self, state, action, reward, next_state, done):
        """
            <数据存储函数>
            在replay_buffer中存储数据

            参数说明：
            ------
            state: 当前时刻状态
            action: 当前时刻动作
            reward：执行当前动作后获得的奖励
            next_state: 执行当前动作后的下一个状态
            done: 是否终止
        """
        # 将上述数据合并存储至data中
        data = (state, action, reward, next_state, done)

        # 设置索引
        idx = self.next_index

        # 进行数据存储
        if idx >= len(self.buffer):
            self.buffer.append(data)
        else:
            self.buffer[idx] = data

        # 索引更新
        self.next_index = (idx + 1) % self.capacity
        # 长度更新
        self.size = min(self.capacity, self.size + 1)

        # 计算Pi^a，新样本获得最大的priority
        priority_alpha = self.max_priority ** self.alpha

        # priority更新
        self._set_priority_min(idx, priority_alpha)
        self._set_priority_sum(idx, priority_alpha)

    def _set_priority_min(self, idx, priority_alpha):
        """
            <最小priority函数>
            在二叉线段树中设置最小priority

            参数说明：
            ------
            idx: 当前transition的索引
            priority_alpha：优先级取值
        """
        # 叶子节点
        idx += self.capacity
        self.priority_min[idx] = priority_alpha

        # 沿父节点遍历，实现对树的更新，直到树的根节点
        while idx >= 2:
            idx //= 2
            self.priority_min[idx] = min(self.priority_min[2 * idx],
                                         self.priority_min[2 * idx + 1])

    def _set_priority_sum(self, idx, priority):
        """
            <求和priority函数>
            在二叉线段树中设置priority求和

            参数说明：
            ------
            idx: 当前transition的索引
            priority：优先级取值
        """
        # 叶子节点
        idx += self.capacity
        self.priority_sum[idx] = priority

        # 沿父节点遍历，实现对树的更新，直到树的根节点
        while idx >= 2:
            idx //= 2
            self.priority_sum[idx] = self.priority_sum[2 * idx] + \
                                     self.priority_sum[2 * idx + 1]

    def _sum(self):
        """
            <priority总和函数>
            在二叉线段树中对priority求总和，具体为：
            ∑k(Pk)^alpha
        """
        return self.priority_sum[1]

    def _min(self):
        """
            <最小priority函数>
            在二叉线段树中搜索最小priority，具体为：
            min_k (Pk)^alpha
        """
        return self.priority_min[1]

    def find_prefix_sum_idx(self, prefix_sum):
        """
            <最大priority搜索函数>
            在二叉线段树中搜索最大priority
        """
        # 从根节点检索
        idx = 1
        while idx < self.capacity:
            if self.priority_sum[idx * 2] > prefix_sum:  # 左节点
                idx = 2 * idx
            else:
                prefix_sum -= self.priority_sum[idx * 2]  # 右节点
                idx = 2 * idx + 1
        return idx - self.capacity

    def sample(self, batch_size, n_steps):
        """
            <数据采样函数>
            在replay_buffer中采样数据

            参数说明：
            ------
            batch_size: 需要从replay_buffer中采样的数据数量
            n_steps: multi-steps learning步长，影响连续采样的样本数量
        """
        # 获取beta值
        beta = self.beta

        # samples初始化
        samples = {
            'weights': np.zeros(shape=batch_size, dtype=np.float32),
            'indexes': np.zeros(shape=batch_size, dtype=np.int32)
        }

        # 获取索引
        for i in range(batch_size):
            p = random.random() * self._sum()
            idx = self.find_prefix_sum_idx(p)
            samples['indexes'][i] = idx

        # 计算min_i Pi
        probability_min = self._min() / self._sum()

        # 计算max_i wi
        max_weight = (probability_min * self.size) ** (-beta)

        # 计算样本权重
        for i in range(batch_size):
            idx = samples['indexes'][i]
            probability = self.priority_sum[idx + self.capacity] / self._sum()  # 计算Pi
            weight = (probability * self.size) ** (-beta)  # 计算权重
            samples['weights'][i] = weight / max_weight  # 为样本赋予权重

        # beta更新
        self.beta = min((beta + self.beta_step), 1)

        # 获取sample
        sample_data = []
        if n_steps == 1:  # 单步学习
            for idx in samples['indexes']:
                data_i = self.buffer[idx]  # 这里data的索引要和priority的索引对应
                sample_data.append(data_i)
        else:  # 多步学习
            for idx in samples['indexes']:
                data_i = self.buffer[idx: idx + n_steps]
                sample_data.append(data_i)

        # 这里除了要返回采样的样本数据，也要返回索引以及对应的权重，以进行priority更新和loss计算
        return samples, sample_data

    def update_priority(self, indexes, priorities):
        """
            <优先级更新函数>
            更新priority

            参数说明：
            ------
            indexes: sample产生的索引
            priorities：优先级具体取值
        """
        # 添加小的epsilon防止零priority
        priorities = priorities.detach().cpu().numpy()
        priorities = priorities + self.epsilon

        # priority更新
        for idx, priority in zip(indexes, priorities):
            self.max_priority = max(self.max_priority, priority)
            priority_alpha = priority ** self.alpha
            self._set_priority_min(idx, priority_alpha)
            self._set_priority_sum(idx, priority_alpha)
