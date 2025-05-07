import numpy as np
import torch

# Define the QNetwork class (same as in the training script)
class QNetwork(torch.nn.Module):
    def __init__(self, state_size, num_actions):
        super(QNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(state_size, 128)
        self.fc2 = torch.nn.Linear(128, 128)
        self.fc3 = torch.nn.Linear(128, 64)
        self.fc4 = torch.nn.Linear(64, num_actions)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)

# Define the policy_action function
def policy_action(policy, observation, device="cpu"):
    # Initialize the Q-network
    state_size = len(observation)
    num_actions = 4  # LunarLander-v3 has 4 actions
    q_network = QNetwork(state_size, num_actions).to(device)

    # Reconstruct the state_dict from the structured array
    state_dict = {key: torch.tensor(policy[key], dtype=torch.float32).to(device) for key in policy.dtype.names}

    # Load the state_dict into the Q-network
    q_network.load_state_dict(state_dict)
    q_network.eval()

    # Convert observation to a tensor and move to the correct device
    observation_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(device)

    # Get the Q-values for the current observation
    with torch.no_grad():
        q_values = q_network(observation_tensor)

    # Select the action with the highest Q-value
    action = torch.argmax(q_values).item()

    return action