import json
import guerrilla_checkers
import torch
import DQN
import statistics
import pandas as pd
import numpy as np
import random
import argparse
import sys

parser = argparse.ArgumentParser(description="Test all available models against randomly moving opponents")
parser.add_argument(
    "--num_checkers",
    type=int,
    default=6,
    help="Number of checkers to place on the starting board. This is to give the guerrilla AI an easier challenge. Will have no effect if < 1 or > 5."
)
parser.add_argument( #TODO: Implement!
    "--rerun",
    action="store_true",
    help="Re-run and overwrite old results."
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
    game_length = (game.starting_stones_num - game.board[0])//2
    return winner, game_length

# TODO: Go through results and only test untested models, to make adding models easier
if num_checkers == 6:
    g_file_name = 'data/g_vs_random.xlsx'
    g_sheet_name = 'guerrilla vs. random'
    c_file_name = 'data/c_vs_random.xlsx'
    c_sheet_name = 'COIN vs. random'
else:
    g_file_name = 'data/g_vs_random_' + str(num_checkers) + '_checkers.xlsx'
    g_sheet_name = 'g vs. random, ' + str(num_checkers) + ' checkers'
    c_file_name = 'data/c_vs_random' + str(num_checkers) + '_checkers.xlsx'
    c_sheet_name = 'C vs. random, ' + str(num_checkers) + ' checkers'

try:
    g_results_df = pd.read_excel(g_file_name)
    new_g_file = False
    print("Loaded data file", g_file_name)
except(FileNotFoundError):
    print("No old data found!")
    new_g_file = True

try:
    c_results_df = pd.read_excel(c_file_name)
    new_c_file = False
    print("Loaded data file", c_file_name)
except(FileNotFoundError):
    print("No old data found!")
    new_c_file = True


g_indexes = []
c_indexes = []

for key, item in model_info.items():
    if item["player"] == "1":
        g_indexes.append(key)
    if item["player"] == "0":
        c_indexes.append(key)

num_games = 1000
precentage_denominator = num_games/100.0

# Possibly: Remove existing indexes, then add new results to old ones
if not new_g_file:
    num_removals = 0
    for index in g_indexes:
        if int(index) in g_results_df.index:
            g_indexes.remove(index)
            num_removals += 1
    print(num_removals, "guerrilla indexes with data found and removed.")

if not new_c_file:
    num_removals = 0
    for index in c_indexes:
        if int(index) in c_results_df.index:
            c_indexes.remove(index)
            num_removals += 1
    print(num_removals, "COIN indexes with data found and removed.")

if len(c_indexes) == 0 and len(g_indexes) == 0:
    print("Nothing to test :)")
    sys.exit()

game = guerrilla_checkers.game(num_checkers=num_checkers)
g_results_array = np.zeros((len(g_indexes), 3))
c_results_array = np.zeros((len(c_indexes), 3))
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
    g_results_array[i] = [g_index, win_rate, avg_length]

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
    c_results_array[i] = [c_index, win_rate, avg_length]


new_g_results_df = pd.DataFrame(data=g_results_array,
                            columns=["Model index", "Win rate", "Avg. game length"])

new_g_results_df = new_g_results_df.set_index("Model index")

if new_g_file:
    g_results_df = new_g_results_df
else:
    pd.concat([g_results_df, new_g_results_df])
g_results_df.to_excel(g_file_name, sheet_name=g_sheet_name)

new_c_results_df = pd.DataFrame(data=c_results_array,
                        index=c_indexes,
                        columns=["Model index", "Win rate", "Avg. game length"])

new_c_results_df = new_c_results_df.set_index("Model index")

if new_c_file:
    c_results_df = new_c_results_df
else:
    pd.concat([c_results_df, new_c_results_df])
c_results_df.to_excel(c_file_name, sheet_name=c_sheet_name)

