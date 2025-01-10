import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple, deque

import copy
import random
import math

import guerrilla_checkers

# 0 for COIN, 1 for guerrilla
action_list_0 = list(guerrilla_checkers.rules['all COIN moves'].keys())
n_actions_0 = len(action_list_0)
action_list_1 = list(guerrilla_checkers.rules['all guerrilla moves'].keys())
n_actions_1 = len(action_list_1)

n_observations = len(guerrilla_checkers.rules["starting board"])

# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Defaults
BATCH_SIZE = 128# number of transitions sampled from the replay buffer
GAMMA = 0.99    # discount factor
EPS_START = 0.9 # starting value of epsilon
EPS_END = 0.05  # final value of epsilon
EPS_DECAY = 1000# controls the rate of exponential decay of epsilon, higher means a slower decay
TAU = 0.005     # update rate of the target network
LR = 1e-4       # learning rate of the ``AdamW`` optimizer

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

class basic(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(basic, self).__init__()

        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    
class deep2(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(deep2, self).__init__()

        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 256)
        self.layer3 = nn.Linear(256, 256)
        self.layer4 = nn.Linear(256, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return self.layer4(x)

class Agent():
    
    def __init__(self, player, game, device, network="basic"):
        self.device = device
        self.game = game
        
        self.steps_done = 0

        self.player = player
        if player == 0:
            self.action_list = action_list_0
            self.n_actions =  n_actions_0
        else:
            self.action_list = action_list_1
            self.n_actions =  n_actions_1
        # TODO Choose network structure based on agenda
        if network == "2deep":
            self.policy_net = deep2(n_observations, self.n_actions).to(self.device)
            self.target_net = deep2(n_observations, self.n_actions).to(self.device)
        else:
            self.policy_net = basic(n_observations, self.n_actions).to(self.device)
            self.target_net = basic(n_observations, self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=LR, amsgrad=True)
        self.memory = ReplayMemory(10000) # TODO: Try different memory sizes?
    
    def select_action(self, state):
        #global steps_done
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        valid_action_indexes = self.game.get_valid_action_indexes(self.player)
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.

                # Apply invalid action masking
                # https://github.com/vwxyzjn/invalid-action-masking/blob/master/test.py
                policy = self.policy_net(state)
                mask_tensor = torch.zeros(self.n_actions, device=self.device, dtype=torch.bool)
                for i in valid_action_indexes:
                    mask_tensor[i] = True
                masked_policy = copy.copy(policy)
                masked_policy = torch.where(mask_tensor, masked_policy, torch.tensor(-1e+8))
                if len((masked_policy == torch.max(masked_policy)).nonzero(as_tuple=True)) > 1:
                    # Select one of the max values
                    try:
                        max_i = random.choice((masked_policy == torch.max(masked_policy)).nonzero(as_tuple=True)).view(1,1)
                    except:
                        breakpoint()
                else:
                    max_i = masked_policy.max(1).indices.view(1, 1)
                return torch.tensor([[max_i]], device=self.device, dtype=torch.long)
        else:
            #if len(valid_action_indexes) < 1:
                #breakpoint()
            random_action = random.choice(valid_action_indexes)
            return torch.tensor([[random_action]], device=self.device, dtype=torch.long)
    
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
                                            batch.next_state)), device=self.device, dtype=torch.bool)
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
        next_state_values = torch.zeros(BATCH_SIZE, device=self.device)
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

class AI():

    def __init__(self, model_path, player, game, device, network_type="basic DQN"):
        #self.model = model
        self.player = player
        self.game = game
        self.device = device
        if self.player == 1:
            self.n_ai_actions = len(guerrilla_checkers.rules['all guerrilla moves'])
        else:
            self.n_ai_actions = len(guerrilla_checkers.rules['all COIN moves'])
        if network_type == "2deep DQN":
            self.model = deep2(n_observations, self.n_ai_actions).to(device)
        else:
            self.model = basic(n_observations, self.n_ai_actions).to(device)
        self.model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
        self.model.eval()

    def select_action(self, state):
        valid_action_indexes = self.game.get_valid_action_indexes(self.player)
        policy = self.model(torch.tensor(state, device=self.device))
        mask_tensor = torch.zeros(self.n_ai_actions, device=self.device, dtype=torch.bool)
        for i in valid_action_indexes:
            mask_tensor[i] = True
        masked_policy = copy.copy(policy)
        masked_policy = torch.where(mask_tensor, masked_policy, torch.tensor(-1e+8))
        if masked_policy.dim() > 1:
            max_i = masked_policy.max(1).indices.view(1, 1)
        else:
            max_i = masked_policy.argmax()
            #max_i = max_i.to(dtype=torch.long, device=self.device)
        if len((masked_policy == torch.max(masked_policy)).nonzero(as_tuple=True)) > 1:
            breakpoint()
        return torch.tensor([[max_i]], device=self.device, dtype=torch.long)