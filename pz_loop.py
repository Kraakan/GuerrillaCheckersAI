from pettingzoo_env import PettingZoo
import random
import copy
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import DQN
# Some of the imports below will be redundant

import torch

import datetime
import json
from pathlib import Path

import csv

import argparse

# Parse terminal arguments
parser = argparse.ArgumentParser(description="Runs oppositional training")
parser.add_argument(
    "--loop",
    type=int,
    default=1,
    help="Number of training sessions to do in this run (each session runs num_episodes games). Training agenda set in training_agenda.json"
)
parser.add_argument(
    "--num_episodes",
    type=int,
    default=10000,
    help="Number of games to run per training session. This number won't be used if torch.cuda.is_available() == False"
)
parser.add_argument(
    "--num_checkers",
    type=int,
    default=6,
    help="Number of checkers to place on the starting board. This is to give the guerrilla AI an easier challenge. Will have no effect if < 1 or > 5."
)
parser.add_argument(
    "--hardcoded_c",
    action='store_true',
    help="This will train guerilla models only, against an opponent that is hardcoded to always chose the first in a list of available moves"
)
parser.add_argument(
    "--random_c",
    action='store_true',
    help="This will train guerilla models only, against an opponent that selects moves at random."
)
parser.add_argument(
    "--no_punish",
    action='store_true',
    help="Remove extra punishment (-1) given to the loser of each game."
)

args = parser.parse_args()

num_loops = args.loop
num_checkers = args.num_checkers

agenda_file = open("training_agenda.json", "r")
agenda = json.load(agenda_file)



# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('device:', device)

env = PettingZoo(num_checkers=num_checkers)

# Get the number of state observations
state = env.reset()
n_observations = len(state)

steps_done = 0

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

def save_models(target_dir, c_target_net, g_target_net, network_type, new_index): # This is where I reversed the players - reorder to fix, but is it worth it?
    # Create unique names by combining adjectives and names from long lists 
    # (duplicates will be unlikely, and won't cause big problems anyway)
    adjectives = open("names/english-adjectives.txt", "r").read().split(sep="\n")
    girl_names = [s.split(sep=";")[0] for s in open("names/names-women.csv", "r").read().split(sep="\n")]
    boy_names = [s.split(sep=";")[0] for s in open("names/names-men.csv", "r").read().split(sep="\n")]
    adj = random.choice(adjectives)
    # Adding "training history", might be useful at a later point
    training_params = {
                  "batch_size": DQN.BATCH_SIZE,
                  "gamma": DQN.GAMMA,
                  "eps_start": DQN.EPS_START,
                  "eps_end": DQN.EPS_END,
                  "eps_decay": DQN.EPS_DECAY,
                  "tau": DQN.TAU,
                  "lr": DQN.LR,
                  "small_reward": small_reward_factor,
                  "big_reward": big_reward_factor,
                  "punish_loser": int(not args.no_punish)
                  }
    if g_target_net ==  None or c_target_net == None:
        if args.hardcoded_c:
            training_info = {
                "description" : str(num_episodes) + " games against hardcoded opponent"
            }
        if args.random_c:
            training_info = {
                "description" : str(num_episodes) + " games against randomly moving opponent"
            }
    else:
        training_info = {
            "description" : str(num_episodes) + " games against twin",
            "opponent id": None
        }
    if num_checkers < 6 and num_checkers > 0:
        training_info["description"] = training_info["description"] + " starting with " + str(num_checkers) + " COIN checkers."
    training_info.update(training_params)
    if c_target_net != None:
        c_model_path = target_dir  + 'coin_model_weights.pth'
        c_name = adj + " " + random.choice(boy_names)
        c_model_info = {"index": str(new_index),
                    "player": "0",
                    "type": network_type,
                    "path": c_model_path,
                    "name": c_name
                    }
        c_model_info["history"] = [training_info]
        model_info[str(new_index)] = c_model_info
        new_index = new_index + 1

        print('Saving COIN model', '"' + c_name + '"' ,'to:', c_model_path)
        torch.save(c_target_net.state_dict(), c_model_path)

    if g_target_net != None:
        g_model_path = target_dir + 'guerrilla_model_weights.pth'
        g_name = adj + " " + random.choice(girl_names)
        g_model_info = {"index": str(new_index),
                    "player": "1",
                    "type": network_type,
                    "path": g_model_path,
                    "name": g_name
                    }
        g_model_info["history"] = [training_info]
        model_info[str(new_index)] = g_model_info

        print('Saving guerrilla model', '"' + g_name + '"' ,'to:', g_model_path)
        torch.save(g_target_net.state_dict(), g_model_path)

    with open('models/model_info.json', 'w') as f:
        json.dump(model_info, f, indent=4) # Will this make my json pretty?
    save_training_data(new_dir, "training-data", wins, game_lengths)

i_loop = 0
i_agenda = 0

while i_loop < num_loops:
    print("Running training loop", i_loop + 1, "of", num_loops, ":")
    small_reward_factor = 1
    big_reward_factor = 1
    while i_agenda < len(agenda):
        if agenda[i_agenda]["status"] == "done":
            i_agenda += 1
        else:
            params = agenda[i_agenda]
            DQN.BATCH_SIZE = params["BATCH_SIZE"]
            DQN.GAMMA = params["GAMMA"]
            DQN.EPS_START = params["EPS_START"]
            DQN.EPS_END = params["EPS_END"]
            DQN.EPS_DECAY = params["EPS_DECAY"]
            DQN.TAU = params["TAU"]
            DQN.LR = params["LR"]
            network = params["network"]
            if "small_reward" in params:
                small_reward_factor = params["small_reward"]
            if "big_reward" in params:
                big_reward_factor = params["big_reward"]
            print(params)
            break
    
    env.game.set_small_reward_factor(small_reward_factor)
    env.game.set_big_reward_factor(big_reward_factor)
    print("Small reward factor:", small_reward_factor, " Big reward factor:", big_reward_factor)
    if torch.cuda.is_available():
        num_episodes = args.num_episodes
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
    if args.hardcoded_c or args.random_c:
        new_dir = 'models/' + str(new_index) + '/'
    else:
        new_dir = 'models/' + str(new_index) + "-" + str(new_index + 1) + '/' # I decided "twins" should share a dir
    print("Creating dir", new_dir)
    Path(new_dir).mkdir()

    # 0 for COIN, 1 for guerrilla
    # These will hopefully be easy to replace with other types of agent
    if args.hardcoded_c:
        COIN = DQN.HardCoded(0, env.game, device)
    elif args.random_c:
        COIN = DQN.Random(0, env.game, device)
    else:
        COIN = DQN.Agent(0, env.game, device, network)
    guerrilla = DQN.Agent(1, env.game, device, network)
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
            next_state, acting_player = env._get_obs()
            

            if len(env.game.get_valid_action_indexes(acting_player)) < 1: # Seems like this will never happen....
                # Might happen if guerrilla doesn't have 2 adjacent spaces to play at,
                # but the game should test for that.
                terminated = True
                #next_state = None
                
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
                if i_episode % 100 == 0:
                    if winner == 0:
                        print("No moves, COIN wins! Reward:" , win_reward, "Punishment:", loss_reward)
                    if winner == 1:
                        print("No moves, guerrilla wins! Reward:" , loss_reward, "Punishment:", loss_reward)
            else:
                if prev_player != acting_player:
                    prev_player = abs(prev_player -1)
                    prev_action = copy.deepcopy(action)
                action = players[acting_player].select_action(state)
                action_to_pass = players[acting_player].action_list[action.item()]
                observation, reward, terminated, truncated, _ = env.step(action_to_pass, acting_player)
                reward = torch.tensor([reward], dtype=torch.float32, device=device)
                
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
                    target_net_state_dict[key] = policy_net_state_dict[key]*DQN.TAU + target_net_state_dict[key]*(1-DQN.TAU)
                players[acting_player].target_net.load_state_dict(target_net_state_dict)
            if terminated and not args.no_punish:
                # Try punishing loser
                # TODO: Add option to turn this off
                result = env.game.get_game_result() # Result code:
                                                    # -1 = guerrilla wins
                                                    # 1 = COIN wins
                                                    # 0 = game isn't over
                                                    # Player index:
                                                    # 0 = COIN
                                                    # 1 =  guerrilla
                if result == 1:
                    loser = 1
                else:
                    loser = 0
                loss_reward = torch.tensor([-1. * big_reward_factor], dtype=torch.float32, device=device)
                if i_episode % 100 == 0:
                    if loser == 0:
                        print("COIN loses! Punishment:" , loss_reward, "Acting player:", acting_player, "Reward:", reward)
                    if loser == 1:
                        print("Guerrilla loses! Punishment:" , loss_reward, "Acting player:", acting_player, "Reward:", reward)
                # Store the transition in memory
                players[loser].push_memory(state, prev_action, next_state, loss_reward)

                # Perform one step of the optimization (on the policy network)
                players[loser].optimize_model()

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = players[loser].target_net.state_dict()
                policy_net_state_dict = players[loser].policy_net.state_dict()
                # I removed the rest of the soft update code because it crashed (I think),
                # but then does anything happen here?
                # As I recall, it couldn't handle next_state == None
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*DQN.TAU + target_net_state_dict[key]*(1-DQN.TAU)
                players[acting_player].target_net.load_state_dict(target_net_state_dict)
            if terminated:
                result = env.game.get_game_result()
                wins.append(result)
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
    if args.hardcoded_c or args.random_c:
        save_models(new_dir, None, players[1].target_net, network + " DQN", new_index)
    else:
        save_models(new_dir, players[0].target_net, players[1].target_net, network + " DQN", new_index)
    plot_wins(show_result=False)
    plt.savefig(new_dir + 'pettingzoo' + "_".join(str(datetime.datetime.now()).split())+ '.png')











