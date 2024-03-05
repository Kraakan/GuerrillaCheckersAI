import torch
import torch.nn as nn
import torch.optim as optim
import guerilla_checkers

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

game = guerilla_checkers.game()
state, current_player = game.get_current_state()

# Initialize the agents
# state_size, action_size, learning_rate
COIN_action_size = len(guerilla_checkers.rules["all COIN moves"])
guerilla_action_size = len(guerilla_checkers.rules["all guerilla moves"])
learning_rate = 0.9 #?
coin_agent = DRLAgent(len(state), COIN_action_size, learning_rate)
guerrilla_agent = DRLAgent(len(state), guerilla_action_size, learning_rate)

# Number of games to play for training
num_games = 10

# Loop over the games
for i in range(num_games):
    # Reset the game state
    game.reset()

    # Play the game until it's over
    while not game.is_game_over():
        # Get the current state and determine the current player
        state, current_player = game.get_current_state()

        # Let the appropriate agent choose an action
        if current_player == 0: # COIN
            action = coin_agent.choose_action(state)
        else:
            action = guerrilla_agent.choose_action(state)

        # Perform the action and get the new state
        new_state, reward, done = game.take_action(current_player, action)

        # Let the appropriate agent observe the result
        if current_player == 0: #'COIN':
            coin_agent.observe((state, action, reward, new_state, done))
        else:
            guerrilla_agent.observe((state, action, reward, new_state, done))

        # If the game is over, let the agents know
        if done:
            coin_agent.end_episode()
            guerrilla_agent.end_episode()

    # Train the agents using the observations from this game
    coin_agent.network.train()
    guerrilla_agent.network.train()

    # Occasionally save the agents' weights
    if i % 1000 == 0:
        coin_agent.save_weights(f'coin_weights_{i}.h5')
        guerrilla_agent.save_weights(f'guerrilla_weights_{i}.h5')