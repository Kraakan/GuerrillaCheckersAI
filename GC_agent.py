import torch
import torch.nn as nn
import torch.optim as optim

# Run the code in guerilla-checkers.py - This worked with the checkers agent
get_ipython().run_line_magic('run', 'guerilla_checkers')

# Define the neural network that will be used to predict action probabilities
class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(PolicyNetwork, self).__init__()
        # Define the first fully connected layer
        self.fc1 = nn.Linear(state_size, 128)
        # Define the second fully connected layer
        self.fc2 = nn.Linear(128, action_size)
    
    # In PyTorch, the `forward` method is a special function that defines the forward pass of a neural network. It's where you specify how your inputs (in this case, the `state`) get transformed into the outputs (in this case, the `action_probs`).
    def forward(self, state):
        # Pass the state through the first fully connected layer and apply ReLU activation function
        x = torch.relu(self.fc1(state))
        # Pass the result through the second fully connected layer and apply softmax to get action probabilities
        action_probs = torch.softmax(self.fc2(x), dim=-1)
        return action_probs

# Define the agent
class DRLAgent:
    def __init__(self, state_size, action_size, learning_rate):
        # Initialize the policy network
        self.network = PolicyNetwork(state_size, action_size)
        # Initialize the optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
    
    def choose_action(self, state, valid_actions):
       # Convert the state to a PyTorch tensor
       state = torch.tensor(state, dtype=torch.float32)
       # Get the action probabilities from the policy network
       action_probs = self.network(state)
       # Zero out the probabilities for invalid actions
       # `valid_actions` is a Boolean tensor of the same shape as `action_probs` that indicates which actions are valid in the current state (with `True` for valid actions and `False` for invalid actions). The function zeros out the probabilities for invalid actions and then renormalizes the action probabilities so they still sum to 1. It then samples an action according to these modified action probabilities.
       action_probs[~valid_actions] = 0
       # Renormalize the action probabilities
       action_probs /= action_probs.sum()
       # Sample an action according to the action probabilities
       action = torch.multinomial(action_probs, 1).item()
       return action

    def update_knowledge(self, state, action, reward, next_state):
        # Convert the state, reward, and next state to PyTorch tensors
        state = torch.tensor(state, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        reward = torch.tensor(reward, dtype=torch.float32)

        # Get the probability of the taken action from the policy network
        current_action_prob = self.network(state)[action]
        # Get the maximum action value for the next state
        next_action_prob = self.network(next_state).max().item()

        # Calculate the loss
        loss = -torch.log(current_action_prob) * (reward + next_action_prob)
        # Zero the gradients
        self.optimizer.zero_grad()
        # Backpropagate the loss
        loss.backward()
        # Update the weights of the policy network
        self.optimizer.step()