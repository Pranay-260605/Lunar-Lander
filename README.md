# Lunar Lander (Deep Q-Network - DQN)

This project implements a reinforcement learning agent for the **LunarLander-v2** environment from **OpenAI Gym**, using the **Deep Q-Network (DQN)** algorithm. The agent’s goal is to safely land a lunar lander on the moon's surface by controlling its altitude, rotation, and orientation using thrusters. The environment challenges the agent with a physics-based simulation where it must optimize fuel usage while avoiding crashes.

## Key Features

- **Environment Setup**: The agent interacts with OpenAI Gym’s LunarLander-v2, where it takes actions such as thrusting, rotating, and controlling altitude to navigate and land the module.
- **Deep Q-Network (DQN)**: The agent uses a DQN to approximate Q-values and make decisions. It learns from past experiences using a replay buffer and stabilizes training with a target network.
- **Reward Function**: The agent is rewarded for safe landings, penalized for crashes, and encouraged to conserve fuel.

## Libraries Used

- **Python**: The primary programming language.
- **OpenAI Gym**: For creating the LunarLander-v2 environment.
- **TensorFlow / PyTorch**: Deep learning framework for training the DQN model.
- **NumPy**: For numerical operations and array manipulations.
