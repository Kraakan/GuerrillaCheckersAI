import guerrilla_checkers
import random
import statistics
import DQN
import torch

try:
    random_vs_random_baseline = open("data/random_vs_random_baseline", "r")
except FileNotFoundError:
    print("Running random vs. random games:")
    results = []
    lengths = []
    game = guerrilla_checkers.game()
    for i in range(1000):
        print(str(i), "games run.", end="\r")
        game.reset()
        while not game.is_game_over():
            state, player = game.get_current_state()
            whose_turn_it_is = int(player)
            valid_actions = game.get_valid_actions(player)
            valid_actions_list = [k for k, v in valid_actions.items() if v == True]
            selected_move = random.choice(valid_actions_list)
            game.take_action(whose_turn_it_is, selected_move)
        winner = game.get_game_result()
        game_length = (66 - game.board[0])//2
        results.append(winner)
        lengths.append(game_length)
    print("\n")
    avg_length = statistics.mean(lengths)
    g_win_rate = results.count(-1)/10.0
    c_win_rate = results.count(1)/10.0
    with open("data/random_vs_random_baseline", "w") as f:
        f.write("Guerrilla wins: " + str(g_win_rate) + " %\n" + "COIN wins: " + str(c_win_rate) + " %\n")
        f.write("Avg. game length: " + str(avg_length))
        f.close()

try:
    random_vs_random_baseline = open("data/random_guerrilla_vs_hardcoded_COIN_baseline", "r")
except FileNotFoundError:
    print("Running random guerrilla vs. hardcoded COIN games:")
    results = []
    lengths = []
    game = guerrilla_checkers.game()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    COIN = DQN.HardCoded(0, game, device)
    for i in range(1000):
        print(str(i), "games run.", end="\r")
        game.reset()
        while not game.is_game_over():
            state, player = game.get_current_state()
            whose_turn_it_is = int(player)
            if whose_turn_it_is == 0:
                action = COIN.select_action(state)
                action = COIN.action_list[action.item()]
                selected_move = tuple(action)
            else:
                valid_actions = game.get_valid_actions(player)
                valid_actions_list = [k for k, v in valid_actions.items() if v == True]
                selected_move = random.choice(valid_actions_list)
            game.take_action(whose_turn_it_is, selected_move)
        winner = game.get_game_result()
        game_length = (66 - game.board[0])//2
        results.append(winner)
        lengths.append(game_length)
    print("\n")
    avg_length = statistics.mean(lengths)
    g_win_rate = results.count(-1)/10.0
    c_win_rate = results.count(1)/10.0
    with open("data/random_guerrilla_vs_hardcoded_COIN_baseline", "w") as f:
        f.write("Guerrilla wins: " + str(g_win_rate) + " %\n" + "COIN wins: " + str(c_win_rate) + " %\n")
        f.write("Avg. game length: " + str(avg_length))
        f.close()

try:
    random_vs_1_checker_baseline = open("data/random_g_vs_1_checker_baseline", "r")
except FileNotFoundError:
    print("Running random guerrilla vs. random COIN with 1 starting checker:")
    results = []
    lengths = []
    game = guerrilla_checkers.game(num_checkers=1)
    for i in range(1000):
        print(str(i), "games run.", end="\r")
        game.reset()
        while not game.is_game_over():
            state, player = game.get_current_state()
            whose_turn_it_is = int(player)
            valid_actions = game.get_valid_actions(player)
            valid_actions_list = [k for k, v in valid_actions.items() if v == True]
            selected_move = random.choice(valid_actions_list)
            game.take_action(whose_turn_it_is, selected_move)
        winner = game.get_game_result()
        game_length = (66 - game.board[0])//2
        results.append(winner)
        lengths.append(game_length)
    print("\n")
    avg_length = statistics.mean(lengths)
    g_win_rate = results.count(-1)/10.0
    c_win_rate = results.count(1)/10.0
    with open("data/random_g_vs_1_checker_baseline", "w") as f:
        f.write("Guerrilla wins: " + str(g_win_rate) + " %\n" + "COIN wins: " + str(c_win_rate) + " %\n")
        f.write("Avg. game length: " + str(avg_length))
        f.close()