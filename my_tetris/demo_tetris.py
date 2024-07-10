import gym
import gym_tetris
import torch
import torch.nn as nn

# Define the same Q-network architecture
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

# Function to select action based on the trained model
def select_action(state, model):
    with torch.no_grad():
        return model(torch.FloatTensor(state)).argmax().item()

# Main function to run the demo
def run_demo():
    # Initialize environment
    env = gym.make('Tetris-v0')
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n

    # Initialize Q-network and load trained weights
    q_network = DQN(input_dim, output_dim)
    q_network.load_state_dict(torch.load('tetris_dqn.pth'))

    # Run the game
    state = env.reset()
    total_reward = 0

    while True:
        env.render()  # Display the game
        action = select_action(state, q_network)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        total_reward += reward

        if done:
            break

    print(f"Total Reward: {total_reward}")
    env.close()

if __name__ == '__main__':
    run_demo()
