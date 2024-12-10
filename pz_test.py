from pettingzoo_env import PettingZoo
import guerrilla_checkers
import math
import random
import copy
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import datetime
import json
from pathlib import Path

# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

# set up matplotlib 
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('device:', device)


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()

        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

BATCH_SIZE = 66 # 128 Excessive? I think it should be set to max n of turns in a game
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

# 0 for COIN, 1 for guerrilla
action_list_0 = list(guerrilla_checkers.rules['all COIN moves'].keys())
n_actions_0 = len(action_list_0)
action_list_1 = list(guerrilla_checkers.rules['all guerrilla moves'].keys())
n_actions_1 = len(action_list_1)

env = PettingZoo()

# Get the number of state observations
state = env.reset()
n_observations = len(state)

steps_done = 0

class dqn_Agent():
    
    def __init__(self, player):
        self.player = player
        if player == 0:
            self.action_list = action_list_0
            self.n_actions =  n_actions_0
        else:
            self.action_list = action_list_1
            self.n_actions =  n_actions_1
        self.policy_net = DQN(n_observations, self.n_actions).to(device)
        self.target_net = DQN(n_observations, self.n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=LR, amsgrad=True)
        self.memory = ReplayMemory(10000)
    
    def select_action(self, state):
        global steps_done
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * steps_done / EPS_DECAY)
        steps_done += 1
        valid_action_indexes = env.game.get_valid_action_indexes(self.player)
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.

                # TODO: Apply invalid action masking
                # https://github.com/vwxyzjn/invalid-action-masking/blob/master/test.py
                policy = self.policy_net(state)
                mask_tensor = torch.zeros(self.n_actions, device=device, dtype=torch.bool)
                for i in valid_action_indexes:
                    mask_tensor[i] = True
                masked_policy = copy.copy(policy)
                masked_policy = torch.where(mask_tensor, masked_policy, torch.tensor(-1e+8))
                max_i = masked_policy.max(1).indices.view(1, 1)
                return torch.tensor([[max_i]], device=device, dtype=torch.long)
        else:
            #if len(valid_action_indexes) < 1:
                #breakpoint()
            random_action = random.choice(valid_action_indexes)
            return torch.tensor([[random_action]], device=device, dtype=torch.long)
    
    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        # https://stackoverflow.com/questions/63493193/index-tensor-must-have-the-same-number-of-dimensions-as-input-tensor-error-encou
        #try:
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        #except:
        #    breakpoint()

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def push_memory(self, state, action, next_state, reward):
        self.memory.push(state, action, next_state, reward)

if torch.cuda.is_available():
    num_episodes = 5000
else:
    num_episodes = 55

wins = []

def plot_wins(show_result=False):
    plt.figure(1)
    wins_t = torch.tensor(wins, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('1 = COIN -1 = guerrilla')
    plt.plot(wins_t.numpy())
    # Take 100 episode averages and plot them too
    if len(wins_t) >= 100:
        means = wins_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

# 0 for COIN, 1 for guerrilla
# These will hopefully be easy to replace with other types of agent
COIN = dqn_Agent(0)
guerrilla = dqn_Agent(1)

# Player designators correstonds to list indexes
players = [COIN, guerrilla]

for i_episode in range(num_episodes):
    # Initialize the environment and get its state
    prev_action = None
    prev_player = 1
    state = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    if i_episode % 50 == 0:
        print("Running episode", i_episode+1)
    terminated = False
    while not terminated:
        acting_player = env.get_acting_player()
        

        if len(env.game.get_valid_action_indexes(acting_player)) < 1: # Seems like this will never happen....
            # Might happen if guerrilla doesn't have 2 adjacent spaces to play at,
            # but the game should test for that.
            terminated = True
            next_state = None
            loser = acting_player
            # Other player = abs(acting_player -1)
            winner = abs(loser -1)
            #TODO: distribute rewards
            loss_reward = torch.tensor([-1.], dtype=torch.float32, device=device)
            players[loser].push_memory(state, action, next_state, loss_reward)
            #The winner's previous action should be used here
            #It's not possible for COIN to lose on chain jumps, is it?
            win_reward = torch.tensor([1.], dtype=torch.float32, device=device)
            breakpoint()
            players[winner].push_memory(state, prev_action, next_state, win_reward)
        else:
            if prev_player != acting_player:
                prev_player = abs(prev_player -1)
                prev_action = copy.deepcopy(action)
            action = players[acting_player].select_action(state)
            action_to_pass = players[acting_player].action_list[action.item()]
            observation, reward, terminated, truncated, _ = env.step(action_to_pass, acting_player)
            reward = torch.tensor([reward], dtype=torch.float32, device=device)
            
            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
            
            # Store the transition in memory
            players[acting_player].push_memory(state, action, next_state, reward)

 

            # Perform one step of the optimization (on the policy network)
            players[acting_player].optimize_model()

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = players[acting_player].target_net.state_dict()
            policy_net_state_dict = players[acting_player].policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            players[acting_player].target_net.load_state_dict(target_net_state_dict)
        if terminated:
            #TODO: Try punishing loser
            loser = env.get_acting_player()
            loss_reward = torch.tensor([-1.], dtype=torch.float32, device=device)
            
            # Store the transition in memory
            players[loser].push_memory(state, prev_action, next_state, loss_reward)
            #breakpoint()

            # Perform one step of the optimization (on the policy network)
            players[loser].optimize_model()

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = players[loser].target_net.state_dict()
            policy_net_state_dict = players[loser].policy_net.state_dict()
            
            who_won = env.game.get_game_result()
            wins.append(who_won)
            plot_wins()
            break
        
        # Move to the next state
        state = next_state

plot_wins(show_result=True)
plt.ioff()
plt.savefig('pettingzoo' + "_".join(str(datetime.datetime.now()).split())+ '.png')
plt.show()