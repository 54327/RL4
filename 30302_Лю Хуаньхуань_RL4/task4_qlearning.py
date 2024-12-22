import sys, os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import random
import math
import gym

# Настройте виртуальный дисплей, подходящий для сред без физических дисплеев.;设置虚拟显示，适用于没有物理显示器的环境
if type(os.environ.get("DISPLAY")) is not str or len(os.environ.get("DISPLAY")) == 0:
    os.environ['DISPLAY'] = ':1'

# Реализация агента Q-Learning;Q-Learning Agent 实现
class QLearningAgent:
    def __init__(self, alpha, epsilon, discount, get_legal_actions):
        """
        Q-Learning Agent
        :param alpha: 学习率
        :param epsilon: 探索率
        :param discount: 折扣因子
        :param get_legal_actions: 获取合法动作的方法
        """
        self.get_legal_actions = get_legal_actions
        self._qvalues = defaultdict(lambda: defaultdict(lambda: 0))
        self.alpha = alpha
        self.epsilon = epsilon
        self.discount = discount

    def get_qvalue(self, state, action):
        """ 返回 Q(s, a) 值 """
        return self._qvalues[tuple(state)][action]

    def set_qvalue(self, state, action, value):
        """ 设置 Q(s, a) 值 """
        self._qvalues[state][action] = value

    def get_value(self, state):
        """
        计算状态的价值 V(s) = max_over_action Q(s, a)
        如果没有合法动作，返回 0.0
        """
        possible_actions = self.get_legal_actions(state)
        if len(possible_actions) == 0:
            return 0.0
        value = max(self.get_qvalue(state, action) for action in possible_actions)
        return value

    def update(self, state, action, reward, next_state):
        """
        更新 Q 值：
        Q(s,a) := (1 - alpha) * Q(s,a) + alpha * (r + gamma * V(s'))
        """
        gamma = self.discount
        learning_rate = self.alpha

        current_q = self.get_qvalue(state, action)
        next_value = self.get_value(next_state)
        updated_q = (1 - learning_rate) * current_q + learning_rate * (reward + gamma * next_value)
        self.set_qvalue(tuple(state), action, updated_q)

    def get_best_action(self, state):
        """
        返回状态的最佳动作
        如果没有合法动作，返回 None
        """
        possible_actions = self.get_legal_actions(state)
        if len(possible_actions) == 0:
            return None
        best_action = max(possible_actions, key=lambda action: self.get_qvalue(state, action))
        return best_action

    def get_action(self, state):
        """
        根据 epsilon-greedy 策略选择动作
        以 epsilon 概率选择随机动作，否则选择最佳动作
        """
        possible_actions = self.get_legal_actions(state)
        if len(possible_actions) == 0:
            return None

        if random.random() < self.epsilon:
            return random.choice(possible_actions)
        else:
            return self.get_best_action(state)

# Эксперимент в среде Taxi-v3;Taxi-v3 环境实验
def play_and_train(env, agent, t_max=10**4):
    """
    执行完整游戏并训练代理
    返回总奖励
    """
    total_reward = 0.0
    state = env.reset()
    if isinstance(state, dict):  # Убедитесь, что оператор if находится внутри тела функции.;确保这个 if 语句在函数体内部
        state = tuple(state.values())
    for t in range(t_max):
        action = agent.get_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.update(state, action, reward, next_state)
        state = next_state
        total_reward += reward
        if done:
            break
    return total_reward  # Убедитесь, что оператор return находится внутри тела функции.;确保 return 语句在函数体内部


# Инициализировать среду Taxi-v3;初始化 Taxi-v3 环境
env = gym.make("Taxi-v3")
n_actions = env.action_space.n
agent = QLearningAgent(alpha=0.5, epsilon=0.25, discount=0.99, get_legal_actions=lambda s: range(n_actions))

rewards = []
for i in range(1000):
    rewards.append(play_and_train(env, agent))
    agent.epsilon *= 0.99  # Снизить скорость разведки;减少探索率
    if i % 100 == 0:
        plt.title(f'eps = {agent.epsilon:.2e}, mean reward = {np.mean(rewards[-10:]):.1f}')
        plt.plot(rewards)
        plt.show()

# Бинаризация состояния среды CartPole-v0;CartPole-v0 环境状态二值化
from gym.core import ObservationWrapper

class Binarizer(ObservationWrapper):
    def observation(self, state):
        """
        将状态二值化
        """
        state = np.round(state, decimals=1)
        if isinstance(state, np.ndarray):
            state = tuple(state)
        return tuple(state)  # Исправить отступ;修正缩进

# Инициализируйте среду CartPole-v0;初始化 CartPole-v0 环境
env2 = Binarizer(gym.make("CartPole-v0")).env
seen_observations = []

for _ in range(1000):
    state = env2.reset()
    seen_observations.append(state)
    done = False
    while not done:
        action = env2.action_space.sample()
        state, _, done, _ = env2.step(action)
        seen_observations.append(state)

seen_observations = np.array(seen_observations)
for obs_i in range(env2.observation_space.shape[0]):
    plt.hist(seen_observations[:, obs_i], bins=20)
    plt.show()