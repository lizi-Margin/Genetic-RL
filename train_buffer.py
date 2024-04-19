import random
from collections import deque
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return ReplayBuffer.toNumpy(states), ReplayBuffer.toNumpy(actions), ReplayBuffer.toNumpy(rewards), ReplayBuffer.toNumpy(next_states), ReplayBuffer.toNumpy(dones)

    @ staticmethod
    def toList(tuple):
        return [i for i in tuple]
    @ staticmethod
    def toNumpy(tuple):
        return np.array(ReplayBuffer.toList(tuple))
    def __len__(self):
        return len(self.buffer)

# 使用示例：
# 创建一个容量为10000的ReplayBuffer
# replay_buffer = ReplayBuffer(capacity=10000)

# 将经验添加到缓冲区中
# replay_buffer.add(state, action, reward, next_state, done)

# 从缓冲区中随机采样一批经验
# states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
