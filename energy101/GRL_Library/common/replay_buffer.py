"""
    该函数用来定义DRL中的replay_buffer
"""

import numpy as np


class ReplayBuffer(object):
    def __init__(self, size):
        """
            <构造函数>
            定义replay_buffer类，以创建replay_buffer

            参数说明：
            ------
            size: replay_buffer的最大容量，当超过容量时，新的
                  数据会替换掉旧的数据
        """
        self.size = size  # 定义replay_buffer的最大容量
        self.buffer = []  # 定义replay_buffer的储存list(储存核心)
        self.index = 0  # 定义replay_buffer索引
        self.length = 0  # 定义replay_buffer当前长度(运行步长)

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

        # 进行数据存储
        if self.index >= len(self.buffer):
            self.buffer.append(data)
        else:
            self.buffer[self.index] = data

        # 索引更新
        self.index = (self.index + 1) % self.size

        # 长度更新
        self.length = min(self.length + 1, self.size)

    def sample(self, batch_size, n_steps):
        """
            <数据采样函数>
            在replay_buffer中采样数据

            参数说明：
            ------
            batch_size: 需要从replay_buffer中采样的数据数量
            n_steps: multi-steps learning步长，影响连续采样的样本数量
        """
        # samples初始化，与PER形式统一，以记录权重和索引
        # 普通replay_buffer，索引随机生成，权重为全1矩阵
        samples = {'weights': np.ones(shape=batch_size, dtype=np.float32),
                   'indexes': np.random.choice(self.length - n_steps + 1, batch_size, replace=False)}

        # 数据采样
        sample_data = []
        if n_steps == 1:  # 单步学习
            for i in samples['indexes']:
                data_i = self.buffer[i]
                sample_data.append(data_i)
        else:  # 多步学习
            for i in samples['indexes']:
                data_i = self.buffer[i: i + n_steps]
                sample_data.append(data_i)

        return samples, sample_data

