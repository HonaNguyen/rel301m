import gym
import gym_tetris
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

# Define the Q-network
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Hyperparameters
EPISODES = 1000
GAMMA = 0.99
LR = 0.001
BATCH_SIZE = 64
TARGET_UPDATE = 10
MEMORY_SIZE = 10000

# Initialize environment
env = gym.make('Tetris-v0')
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n

# Initialize Q-network and target network
q_network = DQN(input_dim, output_dim)
target_network = DQN(input_dim, output_dim)
target_network.load_state_dict(q_network.state_dict())
optimizer = optim.Adam(q_network.parameters(), lr=LR)
memory = deque(maxlen=MEMORY_SIZE)

def select_action(state, epsilon):
    if random.random() > epsilon:
        with torch.no_grad():
            return q_network(torch.FloatTensor(state)).argmax().item()
    else:
        return random.choice(range(output_dim))

def train():
    if len(memory) < BATCH_SIZE:
        return

    batch = random.sample(memory, BATCH_SIZE)
    states, actions, rewards, next_states, dones = zip(*batch)
    
    states = torch.FloatTensor(states)
    actions = torch.LongTensor(actions)
    rewards = torch.FloatTensor(rewards)
    next_states = torch.FloatTensor(next_states)
    dones = torch.FloatTensor(dones)

    q_values = q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    next_q_values = target_network(next_states).max(1)[0]
    target_q_values = rewards + GAMMA * next_q_values * (1 - dones)

    loss = nn.MSELoss()(q_values, target_q_values.detach())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01

for episode in range(EPISODES):
    state = env.reset()
    total_reward = 0

    while True:
        action = select_action(state, epsilon)
        next_state, reward, done, _ = env.step(action)
        memory.append((state, action, reward, next_state, done))

        state = next_state
        total_reward += reward
        train()

        if done:
            break

    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    if episode % TARGET_UPDATE == 0:
        target_network.load_state_dict(q_network.state_dict())

    print(f"Episode: {episode}, Total Reward: {total_reward}")

# Save the trained model
torch.save(q_network.state_dict(), 'tetris_dqn.pth')

# To load the model for inference
# q_network.load_state_dict(torch.load('tetris_dqn.pth'))
