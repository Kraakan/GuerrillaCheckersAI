import guerrilla_checkers
import random
import statistics
import DQN
import torch
import argparse

parser = argparse.ArgumentParser(description="Get baselines for various tests. Tests will not be run if results are already saved")
parser.add_argument(
    "--num_games",
    type=int,
    default=10000, # 1000 games seems to have been insufficient to get a solid average
    help="Number of games to run for random tests. This exists because I'm bad at statistics"
)
args = parser.parse_args()
num_games = args.num_games

precentage_denominator = num_games/100.0

try:
    random_vs_random_baseline = open("data/random_vs_random_baseline.txt", "r")
except FileNotFoundError:
    print("Running random vs. random games:")
    results = []
    lengths = []
    game = guerrilla_checkers.game()
    for i in range(num_games):
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
    g_win_rate = results.count(-1)/precentage_denominator
    c_win_rate = results.count(1)/precentage_denominator
    with open("data/random_vs_random_baseline.txt", "w") as f:
        f.write("Guerrilla wins: " + str(g_win_rate) + " %\n" + "COIN wins: " + str(c_win_rate) + " %\n")
        f.write("Avg. game length: " + str(avg_length))
        f.close()

try:
    random_vs_hardcoded_baseline = open("data/random_guerrilla_vs_hardcoded_COIN_baseline.txt", "r")
except FileNotFoundError:
    print("Running random guerrilla vs. hardcoded COIN games:")
    results = []
    lengths = []
    game = guerrilla_checkers.game()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    COIN = DQN.HardCoded(0, game, device)
    for i in range(num_games):
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
    g_win_rate = results.count(-1)/precentage_denominator
    c_win_rate = results.count(1)/precentage_denominator
    with open("data/random_guerrilla_vs_hardcoded_COIN_baseline.txt", "w") as f:
        f.write("Guerrilla wins: " + str(g_win_rate) + " %\n" + "COIN wins: " + str(c_win_rate) + " %\n")
        f.write("Avg. game length: " + str(avg_length))
        f.close()

for c in range(1,7):
    try:
        existing_file = open("data/random_g_vs_" + str(c) + "_checker_baseline.txt", "r")
        existing_file.close()
    except FileNotFoundError:
        print("Running random guerrilla vs. random COIN with " + str(c) + " starting checker:")
        results = []
        lengths = []
        game = guerrilla_checkers.game(num_checkers=c)
        for i in range(num_games):
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
        g_win_rate = results.count(-1)/precentage_denominator
        c_win_rate = results.count(1)/precentage_denominator
        with open("data/random_g_vs_" + str(c) + "_checker_baseline.txt", "w") as f:
            f.write("Guerrilla wins: " + str(g_win_rate) + " %\n" + "COIN wins: " + str(c_win_rate) + " %\n")
            f.write("Avg. game length: " + str(avg_length))
            f.close()

    try:
        existing_file = open("data/random_g_vs_hardcoded_" + str(c) + "_checker_baseline.txt", "r")
        existing_file.close()
    except FileNotFoundError:
        print("Running random guerrilla vs. hardcoded COIN with " + str(c) + " starting checker:")
        results = []
        lengths = []
        game = guerrilla_checkers.game(num_checkers=c)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        COIN = DQN.HardCoded(0, game, device)
        for i in range(num_games):
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
        g_win_rate = results.count(-1)/precentage_denominator
        c_win_rate = results.count(1)/precentage_denominator
        with open("data/random_g_vs_hardcoded_" + str(c) + "_checker_baseline.txt", "w") as f:
            f.write("Guerrilla wins: " + str(g_win_rate) + " %\n" + "COIN wins: " + str(c_win_rate) + " %\n")
            f.write("Avg. game length: " + str(avg_length))
            f.close()