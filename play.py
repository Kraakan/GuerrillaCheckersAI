import guerilla_checkers
import random
import copy
import curses
from curses import wrapper

# Input:
# c = stdscr.getch()

def start(stdscr):
    
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_YELLOW, -1)
    curses.init_pair(2, curses.COLOR_MAGENTA, -1)

    while True:
        stdscr.clear()
        stdscr.addstr("There is no AI yet, just random choice.\n")
        stdscr.addstr("Do you want to play with 0, 1 or 2 players? (q to quit) (0/1/2/q)")
        player_choice = stdscr.getch()
        if player_choice == ord('0'): # 0
            game_record, move_history = randomized_game(stdscr)
            review_game(stdscr, game_record, move_history)
            break
        
        if str(player_choice) == "1":
            while True:
                player_side = input("Will you play as guerilla or COIN? (g/c)")
                if player_side == "g":
                    one_player_game(1, stdscr)
                    break
                if player_side == "c":
                    one_player_game(0, stdscr)
                    break
                stdscr.addstr("You have to type 'g' or 'c'!")
            break
        
        if str(player_choice) == "2":
            two_player_game(stdscr)
            break
        
        if player_choice == ord('q'):
            stdscr.addstr("Bye!")
            break
        stdscr.addstr("Incorrect input")

def draw_board_with_curses(board, stdscr, y, x, move = None):
    stones, squares, grid = guerilla_checkers.decompress_board(board)
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
    blackstone =u"\u25D9"
    stdscr.addstr("  A B C D E F G H\n")
    stdscr.addstr(u" \u2588\u2588\u259B\u2580\u259C\u2588\u259B\u2580\u259C\u2588\u259B\u2580\u259C\u2588\u259B\u2580\u259C")
    if move != None:
        stdscr.addstr(str(move))
    
    for i, row in enumerate(squares):
        #stdscr.addstr(i, row)
        # Move cursor to next row?
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
                            stdscr.addstr(blackstone + black_top)                        
                    else:
                        if cross == 0:
                            stdscr.addstr(black_corners_a + black_bottom)
                        else:
                            stdscr.addstr(blackstone + black_bottom)
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
                            stdscr.addstr(blackstone + black_bottom)                    
                    else:
                        if cross == 0:
                            stdscr.addstr(black_corners_b + black_top)
                        else:
                            stdscr.addstr(blackstone + black_top)    
                stdscr.addstr(u"\u259C")
    # Move cursor to next row?
    stdscr.addstr("\n ")
    stdscr.addstr(u"\u2599\u2584\u259F\u2588\u2599\u2584\u259F\u2588\u2599\u2584\u259F\u2588\u2599\u2584\u259F\u2588\u2588")
    stdscr.chgat(y, x, 1, curses.A_STANDOUT)
    # TODO?: return available coordinates

def two_player_game(stdscr):
    twoplayergame = guerilla_checkers.game()
    player = 1
    while not twoplayergame.is_game_over():
        valid_actions = twoplayergame.get_valid_actions(player)
        turn_over = False
        while not turn_over:
            draw_board_with_curses(twoplayergame.board, stdscr, y, x)
            if player == 1:
                stdscr.addstr('Turn' + str(len(twoplayergame.game_record)) + ': Guerilla has' + str(twoplayergame.board[0]) + 'stones. Guerillas move')
            if player == 0:
                stdscr.addstr('Turn' + str(len(twoplayergame.game_record)) + ': Guerilla has' + str(twoplayergame.board[0]) + 'stones. COINs move')
            try:
                c = stdscr.getch()
                #move = int(input("You have {} possible moves, please chose one. ".format(str(len(valid_actions)))))
            except ValueError:
                stdscr.addstr("Please enter a number.")
            if move in range(len(valid_actions)):
                draw_board_with_curses(twoplayergame.board, stdscr, y, x, move = valid_actions[move])
                stdscr.addstr('')
                confirm = str(input("Do you chose this move? (y/n)"))
                if confirm == "y":
                    turn_over = True
                    twoplayergame.take_action(player, valid_actions[move])
                    player = int(twoplayergame.guerillas_turn)
            else:
                stdscr.addstr("You need to enter a number between 0 and", len(valid_actions)-1)
    winner = twoplayergame.get_game_result()
    if winner == None:
        stdscr.addstr("No winner")
    if winner == -1:
        stdscr.addstr("Guerilla wins")
    if winner == 1:
        stdscr.addstr("COIN wins")
    return twoplayergame.game_record

def one_player_game(human, stdscr):
    oneplayergame = guerilla_checkers.game()
    player = 1
    while not oneplayergame.is_game_over():
        turn_over = False
        if player == human:
            valid_actions = list(oneplayergame.get_valid_actions(player).keys())
            while not turn_over:
                draw_board_with_curses(oneplayergame.board, stdscr, y, x)
                if player == 1:
                    stdscr.addstr('Turn', str(len(oneplayergame.game_record)),': Guerilla has', oneplayergame.board[0], 'stones. Guerillas move.')
                if player == 0:
                    stdscr.addstr('Turn', str(len(oneplayergame.game_record)),': Guerilla has', oneplayergame.board[0], 'stones. COINs move.')
                try:
                    move = int(input("You have {} possible moves, please chose one. ".format(str(len(valid_actions)))))
                except ValueError:
                    stdscr.addstr("Please enter a number.")
                if move in range(len(valid_actions)):
                    draw_board_with_curses(oneplayergame.board, stdscr, y, x, move = valid_actions[move])
                    stdscr.addstr('')
                    confirm = str(input("Do you chose this move? (y/n)"))
                    if confirm == "y":
                        turn_over = True
                        oneplayergame.take_action(player, valid_actions[move])
                        player = int(oneplayergame.guerillas_turn)
                else:
                    stdscr.addstr("You need to enter a number between 0 and", len(valid_actions)-1)
        else:
            valid_actions = oneplayergame.get_valid_actions(player)
            oneplayergame.take_action(player, random.choice(valid_actions))
            player = int(oneplayergame.guerillas_turn)
        
    winner = oneplayergame.get_game_result()
    if winner == None:
        stdscr.addstr("No winner")
    if winner == -1:
        stdscr.addstr("Guerilla wins")
    if winner == 1:
        stdscr.addstr("COIN wins")
    return oneplayergame.game_record

def randomized_game(stdscr, draw=False):
    random_game = guerilla_checkers.game()
    player = 1
    move_history = []
    if draw:
        draw_board_with_curses(random_game.board, stdscr, 0, 0)
    while not random_game.is_game_over():
        player = int(random_game.guerillas_turn)
        valid_actions = random_game.get_valid_actions(player)
        valid_actions_list = [k for k, v in valid_actions.items() if v == True]
        if len(valid_actions_list) > 0:
            selected_move = random.choice(valid_actions_list)
            move_history.append(selected_move)
            random_game.take_action(player, selected_move)
        player = int(random_game.guerillas_turn)
        if draw:
            if random_game.guerillas_turn:
                stdscr.addstr('Guerilla', end='')
            else:
                stdscr.addstr('COIN', end='')
            stdscr.addstr(' turn ' + str(len(random_game.game_record)))
            stdscr.addstr('Guerilla has', random_game.board[0], 'stones.')
            draw_board_with_curses(random_game.board, stdscr, 0, 0)
    winner = random_game.get_game_result()
    if winner == None:
        stdscr.addstr("No winner")
    if winner == -1:
        stdscr.addstr("Guerilla wins")
    if winner == 1:
        stdscr.addstr("COIN wins")
    return random_game.game_record, move_history

def review_game(stdscr, game_record, move_history):
    stdscr.clear()
    stdscr.addstr("RESULTS")
    turn = 0
    while True:
        if turn > 0:
            previous_move = move_history[turn - 1]
        else:
            previous_move = None
        draw_board_with_curses(game_record[turn], stdscr, 0, 0, move = previous_move)
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