import guerrilla_checkers
import random
from gym_env import gym_env

def hello():
    print("Hello")

def randomized_game(draw=False):
    random_game = guerrilla_checkers.game()
    player = 1
    move_history = []
    
    while not random_game.is_game_over():
        player = int(random_game.guerrillas_turn)
        valid_actions = random_game.get_valid_actions(player)
        valid_actions_list = [k for k, v in valid_actions.items() if v == True]
        if len(valid_actions_list) > 0:
            selected_move = random.choice(valid_actions_list)
            move_history.append(selected_move)
            random_game.take_action(player, selected_move)
        else:
            guerrilla_checkers.draw_board(random_game.board)
            breakpoint()
            break
    winner = random_game.get_game_result()
    if winner == None:
        print("No winner")
    if winner == -1:
        print("guerrilla wins")
    if winner == 1:
        print("COIN wins")
    #breakpoint()
    return random_game.game_record, move_history

#game_record, move_history = randomized_game()
#print("Board size:", len(game_record[0]))
#print(game_record, move_history)

env = gym_env(guerrilla_checkers.game(), 1)

# Number of actions assuming player is guerrilla
n_actions = len(env.action_space)
# Get the number of state observations
state = env.reset()
n_observations = len(state)

# print("Action space size:", n_actions, "Observation space size:", n_observations, "Reset returns:", state)

valid_actions = env.get_valid_sample()
print(valid_actions)
print(type(valid_actions))

print(env.game.board)

print(len(env.game.board))

empty_board = [0] * 82

def draw_board(board):
    stones, squares, grid = guerrilla_checkers.decompress_board(env.game.board)
    for i, row in enumerate(squares):
        print(i, row)
        if i%2 == 0:
            for j, square in enumerate(row):
                print('0', square, end=' ')
            print('')
            if i<7:
                print(grid[i, :])
        else:
            for j, square in enumerate(row):
                print(square, '0', end=' ')
            print('')
            if i<7:
                print(grid[i, :])

draw_board(env.game.board)

guerrilla_checkers.draw_board(env.game.board)