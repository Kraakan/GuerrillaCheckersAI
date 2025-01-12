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

def play(g_info, c_info):
    game = guerrilla_checkers.game()
    # Get model networks from info
    g_AI = DQN.AI(g_info["path"], 1, game, device, network_type=g_info["type"])
    c_AI = DQN.AI(c_info["path"], 0, game, device, network_type=c_info["type"])
    players = [c_AI, g_AI]
    while not game.is_game_over():
        state, player = game.get_current_state()
        whose_turn_it_is = int(player)
        action = players[whose_turn_it_is].select_action(state)
        selected_move = action_lists[whose_turn_it_is][action.item()]
        game.take_action(whose_turn_it_is, selected_move)
    winner = game.get_game_result()
    game_length = (66 - game.board[0])//2
    return winner, game_length

# Play every guerrilla player against every COIN player
g_indexes = []
c_indexes = []
for key, item in model_info.items():
    if item["player"] == "1":
        g_indexes.append(key)
    if item["player"] == "0":
        c_indexes.append(key)

wins = {}
results_array = np.zeros((len(g_indexes), len(c_indexes)))
for i, g_index in enumerate(g_indexes):
    wins[g_index] = []
    g_info = model_info[g_index]
    for j, c_index in enumerate(c_indexes):
        try:
            len(wins[c_index])
        except:
            wins[c_index] = []
        c_info = model_info[c_index]
        results = []
        lengths = []
        for k in range(1):
            score, length = play(g_info, c_info)
            results.append(score)
            lengths.append(length)
        results_array[i,j] = statistics.mean(results)
        print(g_info["name"], "vs.", c_info["name"])
        #print("Guerrilla wins:", results.count(-1))
        #print("COIN wins:", results.count(1))
        if results.count(-1) > len(results)/2:
            wins[g_index].append(c_index)
            print("guerrilla wins!")
        else:
            wins[c_index].append(g_index)
            print("COIN wins!")
        print("Avg. game length:", statistics.mean(lengths))

results_df = pd.DataFrame(data=results_array,
                          index=g_indexes,
                          columns=c_indexes)

results_df.to_excel('tournament_results.xlsx', sheet_name='tournament')

for key, item in wins.items():
    print("nr.", key, model_info[key]["name"], "-", len(item), "wins!")