import json
import guerrilla_checkers
import torch
import DQN
import statistics
import pandas as pd
import numpy as np
import random
import argparse

parser = argparse.ArgumentParser(description="Test all available models against randomly moving opponents")
parser.add_argument(
    "--num_checkers",
    type=int,
    default=6,
    help="Number of checkers to place on the starting board. This is to give the guerrilla AI an easier challenge. Will have no effect if < 1 or > 5."
)
args = parser.parse_args()
num_checkers = args.num_checkers

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_COIN_actions = len(guerrilla_checkers.rules['all COIN moves'])
n_guerrilla_actions = len(guerrilla_checkers.rules['all guerrilla moves'])
action_lists = [list(guerrilla_checkers.rules['all COIN moves'].keys()),
                list(guerrilla_checkers.rules['all guerrilla moves'].keys())]

model_info_file = open("models/model_info.json", "r")
model_info = json.load(model_info_file)

def play(game, AI, AI_side):
    game.reset()
    while not game.is_game_over():
        state, player = game.get_current_state()
        whose_turn_it_is = int(player)
        if AI_side == whose_turn_it_is:
            action = AI.select_action(state)
            selected_move = action_lists[whose_turn_it_is][action.item()]
        else:
            valid_actions = game.get_valid_actions(player)
            valid_actions_list = [k for k, v in valid_actions.items() if v == True]
            selected_move = random.choice(valid_actions_list)
        game.take_action(whose_turn_it_is, selected_move)
    winner = game.get_game_result()
    game_length = (66 - game.board[0])//2
    return winner, game_length

g_indexes = []
c_indexes = []
# TODO: Go through results and only test untested models, to make adding models easier
for key, item in model_info.items():
    if item["player"] == "1":
        g_indexes.append(key)
    if item["player"] == "0":
        c_indexes.append(key)

num_games = 1000
precentage_denominator = num_games/100.0

game = guerrilla_checkers.game(num_checkers=num_checkers)
g_results_array = np.zeros((len(g_indexes), 2))
c_results_array = np.zeros((len(c_indexes), 2))
prev_type = ""
print("Testing guerrilla agents against random moves:")
for i, g_index in enumerate(g_indexes):
    g_info = model_info[g_index]
    network_type = g_info["type"]
    if i == 0:
        print("Networks of type ", network_type,":", sep="")
        prev_type = network_type
    elif network_type != prev_type:
        print("\nNetworks of type ", network_type,":", sep="")
        prev_type = network_type
    # Get model networks from info
    g_AI = DQN.AI(g_info["path"], 1, game, device, network_type=network_type)
    
    results = []
    lengths = []

    for k in range(num_games):
            score, length = play(game, g_AI, g_info["player"])
            results.append(score)
            lengths.append(length)
        
        
    win_rate = results.count(-1)
    win_rate = win_rate/precentage_denominator
    print(g_info["name"], " won ", win_rate, "%", sep="")

    avg_length = statistics.mean(lengths)
    g_results_array[i] = [win_rate, avg_length]

prev_type = ""
print("Testing COIN agents against random moves:")
for i, c_index in enumerate(c_indexes):
    c_info = model_info[c_index]
    network_type = c_info["type"]
    if i == 0:
        print("Networks of type ", network_type,":", sep="")
        prev_type = network_type
    elif network_type != prev_type:
        print("\nNetworks of type ", network_type,":", sep="")
        prev_type = network_type
    # Get model networks from info
    c_AI = DQN.AI(c_info["path"], 0, game, device, network_type=network_type)
    
    results = []
    lengths = []
    for k in range(num_games):
            score, length = play(game, c_AI, c_info["player"])
            results.append(score)
            lengths.append(length)
        
        
    win_rate = results.count(1)
    win_rate = win_rate/precentage_denominator
    print(c_info["name"], " won ", win_rate, "%", sep="")

    avg_length = statistics.mean(lengths)
    c_results_array[i] = [win_rate, avg_length]

g_results_df = pd.DataFrame(data=g_results_array,
                          index=g_indexes,
                          columns=["Win rate", "Avg. game length"])

if num_checkers == 6:
    g_results_df.to_excel('data/g_vs_random.xlsx', sheet_name='guerrilla vs. random')
else:
    g_results_df.to_excel('data/g_vs_random_' + str(num_checkers) + '_checkers.xlsx', sheet_name='g vs. random, ' + str(num_checkers) + ' checkers')

c_results_df = pd.DataFrame(data=c_results_array,
                          index=c_indexes,
                          columns=["Win rate", "Avg. game length"])
if num_checkers == 6:
    c_results_df.to_excel('data/c_vs_random.xlsx', sheet_name='COIN vs. random')
else:
    c_results_df.to_excel('data/c_vs_random' + str(num_checkers) + '_checkers.xlsx', sheet_name='C vs. random, ' + str(num_checkers) + ' checkers')

