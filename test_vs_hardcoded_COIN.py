import json
import guerrilla_checkers
import torch
import DQN
import statistics
import pandas as pd
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_COIN_actions = len(guerrilla_checkers.rules['all COIN moves'])
n_guerrilla_actions = len(guerrilla_checkers.rules['all guerrilla moves'])
action_lists = [list(guerrilla_checkers.rules['all COIN moves'].keys()),
                list(guerrilla_checkers.rules['all guerrilla moves'].keys())]

model_info_file = open("models/model_info.json", "r")
model_info = json.load(model_info_file)


def play(game, AI):
    game.reset()
    while not game.is_game_over():
        state, player = game.get_current_state()
        whose_turn_it_is = int(player)
        if whose_turn_it_is == 0:
            action = COIN.select_action(state)
            action = COIN.action_list[action.item()]
            selected_move = tuple(action)
        else:
            action = AI.select_action(state)
            selected_move = action_lists[whose_turn_it_is][action.item()]
        game.take_action(whose_turn_it_is, selected_move)
    winner = game.get_game_result()
    game_length = (66 - game.board[0])//2
    return winner, game_length

g_indexes = []
for key, item in model_info.items():
    if item["player"] == "1":
        g_indexes.append(key)

num_games = 1000
precentage_denominator = num_games/100.0

game = guerrilla_checkers.game()
COIN = DQN.HardCoded(0, game, device)

g_results_array = np.zeros((len(g_indexes), 2))

for i, g_index in enumerate(g_indexes):
    g_info = model_info[g_index]
    network_type = g_info["type"]
    g_AI = DQN.AI(g_info["path"], 1, game, device, network_type=network_type)
    
    results = []
    lengths = []
    for k in range(num_games):
            score, length = play(game, g_AI)
            results.append(score)
            lengths.append(length)

    win_rate = results.count(-1)
    win_rate = win_rate/precentage_denominator
    print(g_info["name"], " won ", win_rate, "%", sep="")

    avg_length = statistics.mean(lengths)
    g_results_array[i] = [win_rate, avg_length]

g_results_df = pd.DataFrame(data=g_results_array,
                          index=g_indexes,
                          columns=["Win rate", "Avg. game length"])

g_results_df.to_excel('data/g_vs_hardcoded_COIN.xlsx', sheet_name='guerrilla vs. hardcoded COIN')