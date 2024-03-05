import guerilla_checkers
import random

def hello():
    print("Hello")

def randomized_game(draw=False):
    random_game = guerilla_checkers.game()
    player = 1
    move_history = []
    
    while not random_game.is_game_over():
        player = int(random_game.guerillas_turn)
        valid_actions = random_game.get_valid_actions(player)
        valid_actions_list = [k for k, v in valid_actions.items() if v == True]
        if len(valid_actions_list) > 0:
            selected_move = random.choice(valid_actions_list)
            move_history.append(selected_move)
            random_game.take_action(player, selected_move)
        else:
            guerilla_checkers.draw_board(random_game.board)
            breakpoint()
            break
    winner = random_game.get_game_result()
    if winner == None:
        print("No winner")
    if winner == -1:
        print("Guerilla wins")
    if winner == 1:
        print("COIN wins")
    #breakpoint()
    return random_game.game_record, move_history

game_record, move_history = randomized_game()
print("Board size:", len(game_record[0]))
print(game_record, move_history)