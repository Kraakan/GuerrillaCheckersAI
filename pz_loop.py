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

import csv

import argparse

# Parse terminal arguments
parser = argparse.ArgumentParser(description="Control training. Training agenda set in training_agenda.json")
parser.add_argument(
    "--loop",
    type=int,
    default=1,
    help="Number of training sessions to do in this run (each session runs num_episodes games)"
)

args = parser.parse_args()

num_loops = args.loop

agenda_file = open("training_agenda.json", "r")
agenda = json.load(agenda_file)

# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

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

# TODO: get network structure from agenda
class DQN_basic(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN_basic, self).__init__()

        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    
class DQN_2deep(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN_2deep, self).__init__()

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

# Defaults
BATCH_SIZE = 128# number of transitions sampled from the replay buffer
GAMMA = 0.99    # discount factor
EPS_START = 0.9 # starting value of epsilon
EPS_END = 0.05  # final value of epsilon
EPS_DECAY = 1000# controls the rate of exponential decay of epsilon, higher means a slower decay
TAU = 0.005     # update rate of the target network
LR = 1e-4       # learning rate of the ``AdamW`` optimizer

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
    
    def __init__(self, player, network):
        self.player = player
        if player == 0:
            self.action_list = action_list_0
            self.n_actions =  n_actions_0
        else:
            self.action_list = action_list_1
            self.n_actions =  n_actions_1
        # TODO Choose network structure based on agenda
        if network == "2deep":
            self.policy_net = DQN_2deep(n_observations, self.n_actions).to(device)
            self.target_net = DQN_2deep(n_observations, self.n_actions).to(device)
        else:
            self.policy_net = DQN_basic(n_observations, self.n_actions).to(device)
            self.target_net = DQN_basic(n_observations, self.n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=LR, amsgrad=True)
        self.memory = ReplayMemory(10000) # TODO: Try different memory sizes?
    
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

                # Apply invalid action masking
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


wins = []
game_lengths = []

def plot_wins(show_result=False):
    plt.figure(1)
    wins_t = torch.tensor(wins, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('-1 = guerrilla 1 = COIN')
    plt.plot(wins_t.numpy())
    # Take 100 episode averages and plot them too
    if len(wins_t) >= 100:
        means = wins_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

def save_training_data(target_dir, name, wins, lengths):
    
    with open(target_dir + "/"+ name + ".csv", "w", newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(wins)
        csvwriter.writerow(lengths)

def save_models(target_dir, g_target_net , c_target_net):
    # Create unique names by combining adjectives and names from long lists 
    # (duplicates will be unlikely, and won't cause big problems anyway)
    adjectives = open("names/english-adjectives.txt", "r").read().split(sep="\n")
    girl_names = [s.split(sep=";")[0] for s in open("names/names-women.csv", "r").read().split(sep="\n")]
    boy_names = [s.split(sep=";")[0] for s in open("names/names-men.csv", "r").read().split(sep="\n")]
    adj = random.choice(adjectives)

    c_model_path = target_dir  + 'coin_model_weights.pth'
    c_name = adj + " " + random.choice(boy_names)
    c_model_info = {"index": str(new_index),
                  "player": "1",
                  "type": "Basic DQN",
                  "path": c_model_path,
                  "name": c_name
                  }
    g_model_path = target_dir + 'guerrilla_model_weights.pth'
    g_name = adj + " " + random.choice(girl_names)
    g_model_info = {"index": str(new_index + 1),
                  "player": "0",
                  "type": "Basic DQN",
                  "path": g_model_path,
                  "name": g_name
                  }
    #breakpoint()
    # Adding "training history", might be useful at a later point
    training_params = {
                  "batch_size": BATCH_SIZE,
                  "gamma": GAMMA,
                  "eps_start": EPS_START,
                  "eps_end": EPS_END,
                  "eps_decay": EPS_DECAY,
                  "tau": TAU,
                  "lr": LR
                  }
    
    training_info = {
        "description" : str(num_episodes) + " games against twin",
        "opponent id": None
    }
    training_info.update(training_params)

    c_model_info["history"] = [training_info]
    c_model_info["history"][-1]["opponent id"] = str(new_index)
    g_model_info["history"] = [training_info]
    g_model_info["history"][-1]["opponent id"] = str(new_index + 1)

    model_info[str(new_index)] = c_model_info
    model_info[str(new_index + 1)] = g_model_info
    #Save both models
    print("Saving guerrilla model to:", g_model_path)
    torch.save(g_target_net.state_dict(), g_model_path)
    print("Saving COIN model to:", c_model_path)
    torch.save(c_target_net.state_dict(), c_model_path)

    with open('models/model_info.json', 'w') as f:
        json.dump(model_info, f, indent=4) # Will this make my json pretty?
    save_training_data(new_dir, "training-data", wins, game_lengths)

i_loop = 0
i_agenda = 0

while i_loop < num_loops:

    while i_agenda < len(agenda):
        if agenda[i_agenda]["status"] == "done":
            i_agenda += 1
        else:
            params = agenda[i_agenda]
            BATCH_SIZE = params["BATCH_SIZE"]
            GAMMA = params["GAMMA"]
            EPS_START = params["EPS_START"]
            EPS_END = params["EPS_END"]
            EPS_DECAY = params["EPS_DECAY"]
            TAU = params["TAU"]
            LR = params["LR"]
            network = params["network"]
            print(params)
            break

    if torch.cuda.is_available():
        num_episodes = 10000
    else: # Don't train with cpu!
        num_episodes = 2
        print(num_loops)
    
    # Ugly, but I need to track this file in att least two places
    try:
        model_info_file = open("models/model_info.json", "r")
        model_info = json.load(model_info_file)
    except FileNotFoundError:
        print("No model info found")
        model_info = {}
    new_index = len(model_info.items())
    new_dir = 'models/' + str(new_index) + "-" + str(new_index + 1) + '/' # I decided "twins" should share a dir
    print("Creating dir", new_dir)
    Path(new_dir).mkdir()

    # 0 for COIN, 1 for guerrilla
    # These will hopefully be easy to replace with other types of agent
    COIN = dqn_Agent(0, network)
    guerrilla = dqn_Agent(1, network)
    # Player designators correstonds to list indexes
    players = [COIN, guerrilla]

    wins = []
    game_lengths = []
    start_time = str(datetime.datetime.now())

    for i_episode in range(num_episodes):
        # Initialize the environment and get its state
        prev_action = None
        prev_player = 1
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        if i_episode % 100 == 0:
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
                # Game length is inferred from the number of stones left to play, since guerrilla always plays exacly 2/turn
                game_lengths.append((66 - env.game.board[0])//2)
                plot_wins()
                if i_episode % 500 == 100: #TODO: adjust
                    # Save game record
                    record = env.game.game_record
                    with open(new_dir + str(i_episode) + ".csv", "w", newline='') as csvfile:
                        csvwriter = csv.writer(csvfile)
                        for row in record:
                            csvwriter.writerow(row)
                break
            
            # Move to the next state
            state = next_state
    end_time = str(datetime.datetime.now())
    i_loop += 1
    agenda[i_agenda]["status"] = "done"
    with open('training_agenda.json', 'w') as f:
        json.dump(agenda, f, indent=4)
    with open(new_dir + 'time.txt', 'w') as f:
        f.write(start_time + "\n" + end_time)
        f.close()
    save_models(new_dir, players[0].target_net, players[1].target_net)
    plot_wins(show_result=False)
    plt.savefig(new_dir + 'pettingzoo' + "_".join(str(datetime.datetime.now()).split())+ '.png')











