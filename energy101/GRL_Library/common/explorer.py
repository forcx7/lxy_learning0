"""
    该函数用来定义DRL中的explorer strategy，具体包括：
    1.常数epsilon贪婪策略
    2.线性衰减epsilon贪婪策略
    3.指数衰减epsilon贪婪策略
"""

import numpy as np


def random_action(original_action):
    """
        <随机动作函数>
        根据GRL模型生成的动作的特征进行动作选择

        参数说明：
        ------
        original_action: GRL神经网络模型生成的原始动作

        注意!!! IMPORTANT!!!
        这个函数需要根据动作空间的特征（取值范围，维度等进行调整）
    """
    action = np.random.choice(np.arange(3), len(original_action))
    return action


class Greedy(object):
    """
        Greedy Strategy
    """

    def __init__(self):
        """
            <构造函数>
        """

    def generate_action(self, original_action):
        """
            <完全贪婪动作函数>
            根据GRL模型生成的动作的特征选择贪婪动作

            参数说明：
            ------
            original_action: GRL神经网络模型生成的原始动作
        """
        return original_action


class ConstantEpsilonGreedy(object):
    """
        Epsilon-greedy with constant epsilon.
    """

    def __init__(self, epsilon):
        """
            <构造函数>

            参数说明：
            ------
            epsilon: 探索率[0,1];
            若epsilon=0: 完全随机策略
            若epsilon=1: 完全贪婪策略
        """
        assert 0 <= epsilon <= 1
        self.epsilon = epsilon

    def generate_action(self, original_action):
        """
            <动作生成函数>
            保证agent按照既定策略选择GRL模型生成的动作

            参数说明：
            ------
            original_action: GRL神经网络模型生成的原始动作
        """
        if np.random.random() > self.epsilon:  # 执行贪婪策略
            action = original_action
        else:  # 执行随机策略（这里要根据动作空间进行调整）
            action = random_action(original_action)
        return action


class LinearDecayEpsilonGreedy(object):
    """
        Epsilon-greedy with linearly decayed epsilon.
    """

    def __init__(self, start_epsilon, end_epsilon, decay_step):
        """
            <构造函数>

            参数说明：
            ------
            start_epsilon: 起始时的epsilon
            end_epsilon: 结束时的epsilon
            decay_step: epsilon取值延迟下降步长

        """
        assert 0 <= start_epsilon <= 1
        assert 0 <= end_epsilon <= 1
        assert decay_step >= 0
        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.decay_step = decay_step
        self.counters = 0  # 定义步长计数器
        self.epsilon = start_epsilon  # 类内定义epsilon取值

    def compute_epsilon(self):
        """
            <epsilon计算函数>
            该函数用来计算不同时刻的epsilon
        """
        if self.counters > self.decay_step:
            epsilon = self.end_epsilon
            self.counters += 1
            return epsilon
        else:
            epsilon_diff = self.end_epsilon - self.start_epsilon
            epsilon = self.start_epsilon + epsilon_diff * (self.counters / self.decay_step)
            self.counters += 1
            return epsilon

    def generate_action(self, original_action):
        """
            <动作生成函数>
            保证agent按照既定策略选择GRL模型生成的动作

            参数说明：
            ------
            original_action: GRL神经网络模型生成的原始动作
        """
        self.epsilon = self.compute_epsilon()
        if np.random.random() > self.epsilon:  # 执行贪婪策略
            action = original_action
        else:  # 执行随机策略（这里要根据动作空间进行调整）
            action = random_action(original_action)
        return action


class ExponentialDecayEpsilonGreedy(object):
    """
        Epsilon-greedy with exponential decayed epsilon.
    """

    def __init__(self, start_epsilon, end_epsilon, decay):
        """
            <构造函数>

            参数说明：
            ------
            start_epsilon: 起始时的epsilon
            end_epsilon: 结束时的epsilon
            decay: epsilon延迟系数
        """
        assert 0 <= start_epsilon <= 1
        assert 0 <= end_epsilon <= 1
        assert 0 < decay < 1
        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.decay = decay
        self.counters = 0  # 定义步长计数器
        self.epsilon = start_epsilon  # 类内定义epsilon取值

    def compute_epsilon(self):
        """
            <epsilon计算函数>
            该函数用来计算不同时刻的epsilon
        """
        epsilon = self.start_epsilon * (self.decay ** self.counters)
        self.counters += 1
        return max(epsilon, self.end_epsilon)

    def generate_action(self, original_action):
        """
            <动作生成函数>
            保证agent按照既定策略选择GRL模型生成的动作

            参数说明：
            ------
            original_action: GRL神经网络模型生成的原始动作
        """
        self.epsilon = self.compute_epsilon()
        if np.random.random() > self.epsilon:  # 执行贪婪策略
            action = original_action
        else:  # 执行随机策略（这里要根据动作空间进行调整）
            action = random_action(original_action)
        return action
