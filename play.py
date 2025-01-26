import guerrilla_checkers
import random
import copy
import curses
from curses import wrapper
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import DQN
import argparse
# Input:
# c = stdscr.getch()
parser = argparse.ArgumentParser(description="Starts a human-playable in the terminal using curses")
parser.add_argument(
    "--num_checkers",
    type=int,
    default=6,
    help="Number of checkers to place on the starting board. This is to give the guerrilla AI an easier challenge. Will have no effect if < 1 or > 5."
)
args = parser.parse_args()
num_checkers = args.num_checkers
#breakpoint()
def start(stdscr):
    
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_MAGENTA, -1)
    curses.init_pair(2, 0, curses.COLOR_MAGENTA)
    curses.init_pair(3, curses.COLOR_CYAN, -1)
    curses.init_pair(4, 0, curses.COLOR_CYAN)
    curses.init_pair(5, 0, curses.COLOR_RED)

    while True:
        stdscr.clear()
        stdscr.addstr("There is no AI yet, just random choice.\n")
        stdscr.addstr("Do you want to play with 0, 1 or 2 players? (q to quit) (0/1/2/q)")
        player_choice = stdscr.getch()
        if player_choice == ord('0'): # 0
            game_record, move_history = randomized_game(stdscr)
            review_game(stdscr, game_record, move_history)
            break
        
        if player_choice == ord('1'):
            
            while True:
                stdscr.clear()
                stdscr.addstr("Will you play as guerrilla or COIN? (g/c)")
                player_side = stdscr.getch()
                if player_side == ord('g'):
                    human_player = 1
                    ai_player = 0
                    break
                if player_side == ord('c'):
                    human_player = 0
                    ai_player = 1
                    break
                stdscr.addstr("You have to type 'g' or 'c'!")
            # Chose model
            try:
                model_info_file = open("models/model_info.json", "r")
                stdscr.addstr("Loading models...")
                model_info = json.load(model_info_file)
                available_models = []
                for key, item in model_info.items():
                    if item["player"] == str(ai_player):
                        available_models.append(key)
                if len(available_models) < 1:
                    stdscr.addstr("Sorry, no appropriate models were found :(")
                    break
                highlited = 0
                rows, cols = stdscr.getmaxyx()
                jmin = 0
                jmax = len(available_models)
                while True:
                    stdscr.clear()
                    stdscr.addstr("Select AI model for opponent\n")
                    i = 0
                    j = jmin
                    while i < (rows - 2):
                        #for m in available_models: # Need to limit the number of options so they fit on the screen
                        if j < len(available_models):
                            if j == highlited:
                                # highlight row
                                stdscr.addstr(available_models[j] + \
                                " type: " + model_info[available_models[j]]["type"] + \
                                " name: " + model_info[available_models[j]]["name"] + \
                                    "\n", curses.A_REVERSE)
                            else:
                                stdscr.addstr(available_models[j] + " type: " + model_info[available_models[j]]["type"] + " name: " + model_info[available_models[j]]["name"] + "\n")
                            jmax = j
                            i += 1
                            if i < (rows - 2):
                                strings_to_print = ""
                                for key, item in model_info[available_models[j]]["history"][0].items():
                                    if i < (rows - 2):
                                        if len(strings_to_print) + 4 + len(key) + 2 + len(str(item)) + 1 >= cols:
                                            stdscr.addstr(strings_to_print + "\n")
                                            strings_to_print = "    " + key + ": " + str(item)
                                            i += 1
                                        else:
                                            strings_to_print += "    " + key + ": " + str(item)
                                if i < (rows - 2):
                                    stdscr.addstr(strings_to_print + "\n\n")
                                    i += 2
                            j += 1

                            #stdscr.getyx()
                        #if "history" in model_info[m]:
                        #    for h in model_info[m]["history"]:
                        #        stdscr.addstr("    " + h["description"] + "\n")
                    # TODO: Select model with curses
                    c = stdscr.getch()
                    if c == ord('q'):
                        break
                    elif c == curses.KEY_MOUSE:
                        mouse_event = curses.getmouse()
                        stdscr.addstr(str(mouse_event))
                    elif c == ord('w') or c == curses.KEY_UP:
                        if highlited == 0:
                            highlited = len(available_models) - 1
                            jmin = len(available_models) - jmax - 1
                        else:
                            highlited = highlited - 1
                            if highlited < jmin:
                                jmin = jmin - 1
                    elif c == ord('s') or c == curses.KEY_DOWN:
                        if highlited == len(available_models) - 1:
                            highlited = 0
                            jmin = 0
                        else:
                            highlited = highlited + 1
                            if highlited > jmax:
                                jmin = jmin + 1
                    elif c == curses.KEY_ENTER or c == 10 or c == 13:
                        opponent_selection = available_models[highlited]
                        selected_opponent = model_info[opponent_selection]
                        # TODO: apply network choice
                        ai_model_path = selected_opponent["path"]
                        one_player_game(human_player, ai_model_path, stdscr, network=selected_opponent["type"])                        
                        break
                    else:
                        stdscr.addstr("Select a model with arrow keys + Enter!")
            except FileNotFoundError:
                stdscr.addstr("ERROR! No model info found!")
            # Display list of applicable models

            # Create guerrilla/COIN AI

            break
        
        if player_choice == ord('2'):
            two_player_game(stdscr)
            break
        
        if player_choice == ord('q'):
            stdscr.addstr("Bye!")
            break
        stdscr.addstr("Incorrect input")

def get_highlight(move, player):
    # Convert move coords to board space on curses screen (1-17)
    y1 = 0
    x1 = 0
    y2 = 0
    x2 = 0

    if move == None or player == None:
        return (y1, x1, y2, x2)
    
    if player == 0:
        # COIN
        move_from = move[:2]
        move_to = move[2:]

        y1 = 2 * (move_from[0] + 1)

        if move_from[0] % 2 == 0:
            x1 = 4 * (move_from[1] + 1)
        else:
            x1 = 4 * (move_from[1] + 1) - 2

        y2 = 2 * (move_to[0] + 1)

        if move_to[0] % 2 == 0:
            x2 = 4 * (move_to[1] + 1)
        else:
            x2 = 4 * (move_to[1] + 1) - 2

        # Odd rows are offset to the left
        #if move_from[1] % 2 != 0:
        #    x1 -= 2
        #if move_to[1] % 2 != 0:
        #    x2 -= 2

    if player == 1:
        # guerrilla
        #first_stone = move[:2]
        #second_stone = move[2:]
        y1, x1, y2, x2 = move

        y1, x1, y2, x2 = (x * 2 + 3 for x in (y1, x1, y2, x2))
    
    return (y1, x1, y2, x2)

def select_move(player, y, x, move_start = None):
    
    if player == 0: # COIN
        move_y = y // 2 - 1
        move_x = x // 4 - 1

        if move_y % 2 != 0:
           move_x = (x + 2) // 4 - 1

        if move_y < 0:
            move_y = 0
        elif move_y > 7:
            move_y = 7
        
        if move_x < 0:
            move_x = 0
        elif move_x > 3:
             move_x = 3   

    else: # guerrilla
        move_y = (y - 3) // 2
        move_x = (x - 3) // 2
        if move_y < 0:
            move_y = 0
        elif move_y > 6:
            move_y = 6
        if move_x < 0:
            move_x = 0
        elif move_x > 6:
            move_x = 6

    if move_start == None:
        return (move_y, move_x, move_y, move_x)
    else:
        return (move_start[0], move_start[1], move_y, move_x)


def draw_board_with_curses(board, stdscr, yy, xx, move = None, player = None):
    stdscr.move(0, 0)

    stones, squares, grid = guerrilla_checkers.decompress_board(board)
    cross_glyph = u"\u253c"
    horizontal_line = u"\u2500"
    vertical_line = u"\u2502"
    black_left = u"\u2590"
    black_right = u"\u258c"
    black_top = u"\u2584"
    black_bottom = u"\u2580"
    black_middle = u"\u2588"
    black_corners_a = u"\u259E"
    black_corners_b = u"\u259A"
    copyright = u"\u00A9"
    fisheye = u"\u25C9"
    inverse_circle = u"\u25D9"
    blackstone = u"\u25CF"
    bullet = u"\u2022"
    stdscr.addstr("  A B C D E F G H\n")
    stdscr.addstr(u" \u2588\u2588\u259B\u2580\u259C\u2588\u259B\u2580\u259C\u2588\u259B\u2580\u259C\u2588\u259B\u2580\u259C")
    if move != None:
        stdscr.addstr(str(move))
    
    for i, row in enumerate(squares):
        stdscr.addstr("\n")
        stdscr.addstr(str(8-i))
        string = ' '
        if i%2 == 0:
            stdscr.addstr(black_middle)
            for j, square in enumerate(row):
                if j > 2:
                    if square == 1:
                        string = copyright
                    else: string = " "
                    stdscr.addstr(black_middle + black_right)
                    stdscr.addstr(string, curses.color_pair(1))
                else:
                    if square == 1:
                        string = copyright
                    else: string = " "
                    stdscr.addstr(black_middle + black_right)
                    stdscr.addstr(string, curses.color_pair(1))
                    stdscr.addstr(black_left)
            # Move cursor to next row
            stdscr.addstr(black_left + "\n ")
            if i<7:
                stdscr.addstr(u"\u259B")
                #stdscr.addstr(grid[i, :])
                stdscr.addstr(black_bottom)
                for j, cross in enumerate(grid[i, :]):
                    if j%2 == 0:
                        if cross == 0:
                            stdscr.addstr(black_corners_b + black_top)
                        else:
                            stdscr.addstr(bullet, curses.color_pair(3))
                            stdscr.addstr(black_top)                        
                    else:
                        if cross == 0:
                            stdscr.addstr(black_corners_a + black_bottom)
                        else:
                            stdscr.addstr(bullet, curses.color_pair(3))
                            stdscr.addstr(black_bottom)
                stdscr.addstr(u"\u259F")
        else:
            stdscr.addstr(black_right)
            for j, square in enumerate(row):
                if j > 2:
                    if square == 1:
                        string = copyright
                    else: string = " "
                    stdscr.addstr(string, curses.color_pair(1))
                    stdscr.addstr(black_left + black_middle + black_middle)
                else:
                    if square == 1:
                        string = copyright
                    else: string = " "
                    stdscr.addstr(string, curses.color_pair(1))
                    stdscr.addstr(black_left + black_middle + black_right)
            if i<7:
                # Move cursor to next row
                stdscr.addstr("\n ")
                stdscr.addstr(u"\u2599")
                #stdscr.addstr(grid[i, :])
                stdscr.addstr(black_top)
                for j, cross in enumerate(grid[i, :]):
                    if j%2 == 0:
                        if cross == 0:
                            stdscr.addstr(black_corners_a + black_bottom)
                        else:
                            stdscr.addstr(bullet, curses.color_pair(3))
                            stdscr.addstr(black_bottom)                    
                    else:
                        if cross == 0:
                            stdscr.addstr(black_corners_b + black_top)
                        else:
                            stdscr.addstr(bullet, curses.color_pair(3))
                            stdscr.addstr(black_top)    
                stdscr.addstr(u"\u259C")
    # Move cursor to next row?
    stdscr.addstr("\n ")
    stdscr.addstr(u"\u2599\u2584\u259F\u2588\u2599\u2584\u259F\u2588\u2599\u2584\u259F\u2588\u2599\u2584\u259F\u2588\u2588")
    stdscr.addstr("y:" + str(yy) + " x:" + str(xx))
    # curses.color_pair(5) curses.A_REVERSE)
    stdscr.chgat(yy, xx, 1, curses.color_pair(5) | curses.A_BLINK)
    if move != None:
        highlight = get_highlight(move, player)
        stdscr.move(16, 18)
        stdscr.addstr(str(highlight))
        y1, x1, y2, x2 = highlight
        if player == 0:
            stdscr.chgat(y1, x1, 1, curses.color_pair(2))
            stdscr.chgat(y2, x2, 1, curses.color_pair(2))
        else:
            stdscr.chgat(y1, x1, 1, curses.color_pair(4))
            stdscr.chgat(y2, x2, 1, curses.color_pair(4))
    stdscr.move(0, 0)

def two_player_game(stdscr):
    twoplayergame = guerrilla_checkers.game()
    player = 1
    yy = 1
    xx = 1
    min_y = 1
    min_x = 1
    max_y = 17
    max_x = 17
    while not twoplayergame.is_game_over():
        valid_actions = twoplayergame.get_valid_actions(player)
        valid_actions_list = [k for k, v in valid_actions.items() if v == True]
        turn_over = False
        move_start = None
        move = None
        while not turn_over:
            stdscr.clear()
            if player == 1:
                stdscr.addstr('Turn' + str(len(twoplayergame.game_record)) + ': guerrilla has' + str(twoplayergame.board[0]) + 'stones. guerrillas move')
            if player == 0:
                stdscr.addstr('Turn' + str(len(twoplayergame.game_record)) + ': guerrilla has' + str(twoplayergame.board[0]) + 'stones. COINs move')
            
            draw_board_with_curses(twoplayergame.board, stdscr, yy, xx, move = move_start, player = player)
            # FOR TESTING:
            #move_start = None
            #player = 0
            ###
            c = stdscr.getch()
            if c == ord('q'):
                break
            elif c == curses.KEY_MOUSE:
                mouse_event = curses.getmouse()
                stdscr.addstr(str(mouse_event))
            elif c == ord('a') or c == curses.KEY_LEFT:
                xx-= 1
            elif c == ord('d') or c == curses.KEY_RIGHT:
                xx+= 1
            elif c == ord('w') or c == curses.KEY_UP:
                yy-= 1
            elif c == ord('s') or c == curses.KEY_DOWN:
                yy+= 1
            if xx> max_x:
                xx= max_x
            if yy> max_y:
                yy= max_y
            if xx< min_x:
                xx= min_x
            if yy< min_y:
                yy= min_y

            # https://stackoverflow.com/questions/32252733/interpreting-enter-keypress-in-stdscr-curses-module-in-python
            if c == curses.KEY_ENTER or c == 10 or c == 13:
                selected_y = yy
                selected_x = xx
                move = select_move(player, selected_y, selected_x, move_start = move_start)

            if move_start == None:
                move_start = move
                move = None
                draw_board_with_curses(twoplayergame.board, stdscr, yy, xx, move = move_start, player = player)
            else:
                draw_board_with_curses(twoplayergame.board, stdscr, yy, xx, move = move, player = player)

            stdscr.move(15, 18)
            if move != None:
                if move in valid_actions_list:
                    turn_over = True
                    twoplayergame.take_action(player, move)
                    player = int(twoplayergame.guerrillas_turn)
                else:
                    stdscr.addstr("Invalid move!")
                    move_start = None
                    move = None
                    confirm = stdscr.getch()
    winner = twoplayergame.get_game_result()
    if winner == None:
        stdscr.addstr("No winner")
    if winner == -1:
        stdscr.addstr("guerrilla wins")
    if winner == 1:
        stdscr.addstr("COIN wins")
    stdscr.getch()
    return twoplayergame.game_record

def one_player_game(human, ai_model_path, stdscr, network="basic DQN"):
    oneplayergame = guerrilla_checkers.game()
    state, player = oneplayergame.get_current_state()
    n_observations = len(state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if human == 1:
        ai_player = 0
        n_ai_actions = len(guerrilla_checkers.rules['all COIN moves'])
        ai_action_list = list(guerrilla_checkers.rules['all COIN moves'].keys())
    else:
        ai_player = 1
        n_ai_actions = len(guerrilla_checkers.rules['all guerrilla moves'])
        ai_action_list = list(guerrilla_checkers.rules['all guerrilla moves'].keys())
    ai = DQN.AI(ai_model_path, ai_player, oneplayergame, device, network_type=network)
    player = 1
    yy = 1
    xx = 1
    min_y = 1
    min_x = 1
    max_y = 17
    max_x = 17
    while not oneplayergame.is_game_over():
        player = int(oneplayergame.guerrillas_turn)
        # I shouldn't need to check if there are no valid moves at this point
        valid_actions = oneplayergame.get_valid_actions(player)
        valid_actions_list = [k for k, v in valid_actions.items() if v == True]
        
        move_start = None
        move = None
        if player == human:
            turn_over = False
            while not turn_over:
                stdscr.clear()
                if player == 1:
                    stdscr.addstr('Turn' + str(len(oneplayergame.game_record)) + ': guerrilla has' + str(oneplayergame.board[0]) + 'stones. guerrillas move')
                if player == 0:
                    stdscr.addstr('Turn' + str(len(oneplayergame.game_record)) + ': guerrilla has' + str(oneplayergame.board[0]) + 'stones. COINs move')
                
                draw_board_with_curses(oneplayergame.board, stdscr, yy, xx, move = move_start, player = player)

                c = stdscr.getch()
                if c == ord('q'):
                    break
                elif c == curses.KEY_MOUSE:
                    mouse_event = curses.getmouse()
                    stdscr.addstr(str(mouse_event))
                elif c == ord('a') or c == curses.KEY_LEFT:
                    xx-= 1
                elif c == ord('d') or c == curses.KEY_RIGHT:
                    xx+= 1
                elif c == ord('w') or c == curses.KEY_UP:
                    yy-= 1
                elif c == ord('s') or c == curses.KEY_DOWN:
                    yy+= 1
                if xx> max_x:
                    xx= max_x
                if yy> max_y:
                    yy= max_y
                if xx< min_x:
                    xx= min_x
                if yy< min_y:
                    yy= min_y

                # https://stackoverflow.com/questions/32252733/interpreting-enter-keypress-in-stdscr-curses-module-in-python
                if c == curses.KEY_ENTER or c == 10 or c == 13:
                    selected_y = yy
                    selected_x = xx
                    move = select_move(player, selected_y, selected_x, move_start = move_start)

                if move_start == None:
                    move_start = move
                    move = None
                    draw_board_with_curses(oneplayergame.board, stdscr, yy, xx, move = move_start, player = player)
                else:
                    draw_board_with_curses(oneplayergame.board, stdscr, yy, xx, move = move, player = player)

                stdscr.move(15, 18)
                if move != None:
                    if move in valid_actions_list:
                        turn_over = True
                        oneplayergame.take_action(player, move)
                    else:
                        stdscr.addstr("Invalid move!")
                        move_start = None
                        move = None
                        confirm = stdscr.getch()
        else:
            # TODO: AI moves here
            action = ai.select_action(state)
            selected_move = ai_action_list[action.item()]
            oneplayergame.take_action(player, selected_move)
        
    winner = oneplayergame.get_game_result()
    if winner == None:
        stdscr.addstr("No winner")
    if winner == -1:
        stdscr.addstr("guerrilla wins")
    if winner == 1:
        stdscr.addstr("COIN wins")
    stdscr.getch()
    return oneplayergame.game_record

def randomized_game(stdscr, draw=False):
    random_game = guerrilla_checkers.game(num_checkers=num_checkers)
    player = 1
    move_history = []
    if draw:
        draw_board_with_curses(random_game.board, stdscr, 0, 0)
    while not random_game.is_game_over():
        player = int(random_game.guerrillas_turn)
        valid_actions = random_game.get_valid_actions(player)
        valid_actions_list = [k for k, v in valid_actions.items() if v == True]
        if len(valid_actions_list) > 0:
            selected_move = random.choice(valid_actions_list)
            move_history.append((player, selected_move))
            random_game.take_action(player, selected_move)
        player = int(random_game.guerrillas_turn)
        if draw:
            if random_game.guerrillas_turn:
                stdscr.addstr('guerrilla', end='')
            else:
                stdscr.addstr('COIN', end='')
            stdscr.addstr(' turn ' + str(len(random_game.game_record)))
            stdscr.addstr('guerrilla has', random_game.board[0], 'stones.')
            draw_board_with_curses(random_game.board, stdscr, 0, 0)
    winner = random_game.get_game_result()
    if winner == None:
        stdscr.addstr("No winner")
    if winner == -1:
        stdscr.addstr("guerrilla wins")
    if winner == 1:
        stdscr.addstr("COIN wins")
    return random_game.game_record, move_history

def review_game(stdscr, game_record, move_history):
    stdscr.clear()
    stdscr.addstr("RESULTS")
    turn = 0
    player = 1
    while True:
        if turn > 0:
            player, previous_move = move_history[turn - 1]
        else:
            previous_move = None
        draw_board_with_curses(game_record[turn], stdscr, 0, 0, move = previous_move, player = player)
        stdscr.addstr("Flip through turns with arrow keys, press q to quit\n")
        c = stdscr.getch()
        if c == ord('q'):
            break
        elif c == ord('a') or c == curses.KEY_LEFT:
            turn -= 1
        
        elif c == ord('d') or c == curses.KEY_RIGHT:
            turn += 1
        if turn < 0:
            turn = 0
        if turn >= len(move_history):
            turn = len(move_history)

wrapper(start)