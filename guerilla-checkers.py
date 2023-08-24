import numpy as np
import copy
import random
import pickle

def compress_board(stones, squares, grid):
    
    # Number of stones first
    board = [stones]
    
    # squares (unused squares never need to be represented)
    squares_list = squares.flatten().tolist()
    board = board + squares_list
    
    # Grid last
    grid_list = grid.flatten().tolist()
    board = board + grid_list
    
    return board

def decompress_board(board):
    stones = board[0]
    squares_list = board[1:33]
    squares = np.array(squares_list).reshape((8, 4))
    # add empty squares
    grid_list = board[33:]
    grid = np.array(grid_list).reshape((7, 7))
    return stones, squares, grid

def create_starting_board():
    squares = np.zeros((8,4), dtype='b')
    squares[3:5,1:3] = 1
    squares[2,1] = 1
    squares[5,2] = 1
    grid = np.zeros((7,7), dtype='b')
    starting_board = compress_board(66,squares,grid)
    return starting_board

# IMPORTANT! Make sure the positions lists are always up to date
def list_checker_positions(board):
    checker_list = []
    for index, square in enumerate(board[1:33]):
        if square == 1:
            checker_list.append(index + 1)
    return checker_list

try:
    rules = pickle.load( open( "rules.pickle", "rb" ))
except FileNotFoundError:
    # Generating "rules" - mostly about the board layout
    
    def diagonal_steps(position):
        steps = []
        # Beacuse the checker board is compressed, diagonals work differently on add and even rows
        if (position // 4) % 2 == 0:
            if not ((position + 5) % 8 == 0):
                steps.append([position - 3])
            steps.append([position - 4])
            if not ((position + 5) % 8 == 0):
                steps.append([position + 5])
            steps.append([position + 4])
        else:
            if not ((position + 4) % 8 == 0):
                steps.append([position - 5])
            steps.append([position - 4])
            if not ((position + 4) % 8 == 0):
                steps.append([position + 3])
            steps.append([position + 4])
        i = 0
        while i < len(steps):
            if steps[i][0] < 0 or steps[i][0] > 31:
                steps.pop(i)
            else: i = i + 1
        steps.sort()
        return steps    
    
    def list_diagonal_crosses(diagonals):
        i = 33
        up = -1
        down = 3
        while i < 33 + 49:
            # each cross is next to 4 squares, what does each of them do?
            if i % 2 != 0:
                # odd = even
                # even crosses go up-right and down-left
                up += 1
                if (i-33) % 7 == 0:
                    down += 1
            else:
                # odd crosses go up-left and down-right
                down +=1
                if (i-33) % 7 == 0:
                    up += 1
            for move in diagonals[up]:
                if move[0] == down:
                    move.append(i)
            for move in diagonals[down]:
                if move[0] == up:
                    move.append(i)
            i += 1
    
    def get_neighbors(point, offset = 0):
        # offset is added to neigbors because the go board will be appended to the checkers board
        neighbors = []
        if not (point % 7 == 0):
            neighbors.append(point - 1 + offset)
        if not ((point + 1) % 7 == 0):
            neighbors.append(point + 1 + offset)
        neighbors.append(point - 7 + offset)
        neighbors.append(point + 7 + offset)
        i = 0
        while i < len(neighbors):
            if neighbors[i] - offset < 0 or neighbors[i] - offset > 48:
                neighbors.pop(i)
            else: i = i + 1
        return neighbors
    
    diagonals = []
    for n in range(32):
        diagonals.append(diagonal_steps(n))
    neighbors = []
    for n in range(49):
        neighbors.append(get_neighbors(n, 0))
    rules = {
        "diagonals" : diagonals,
        "neighbors" : neighbors,
        "starting board" : create_starting_board(),
        "checker positions" : [10, 14, 15, 18, 19, 23]
    }
    list_diagonal_crosses(rules["diagonals"])
    
    pickle.dump(rules, open( "rules.pickle", "wb" ))

class game():
    # Game object, will probably be instantiated for each game
    def __init__(self):
        self.board = rules["starting board"]
        self.checker_positions = rules["checker positions"]
        self.guerillas_turn = True
        self.game_record = [self.board]
        
    def get_current_state(self):
        # This function returns the current state of the game.
        return self.board, self.guerillas_turn
        
    def get_valid_actions(self, player):
        # This function takes the current player as input and returns a list of valid actions for that player.
        moves = None
        if self.board[0] < 66:
            self.checker_positions = list_checker_positions(self.board)
        # player: 0 = guerilla 1 = COIN
        # Not sure if I need to check whose turn it is, 
        # or if returning an empty list is the right response for players "acting out of turn"
        if player == -1:
            if self.guerillas_turn:
                moves = self.get_guerilla_moves()
            else: moves = []
        else:
            if not self.guerillas_turn:
                moves = self.get_COIN_moves()
            else: moves = []
        return moves
    
    def take_action(self, player, action):
        # This function takes the current player and an action as input, updates the game state based on the action, checks if the game has ended, and returns the outcome.
        #result = None
        if (player == -1 and self.guerillas_turn) or (player == 1 and not self.guerillas_turn):
            self.board = action
            self.game_record.append(self.board)
            self.guerillas_turn = not self.guerillas_turn
        #if is_game_over():
        #    result = get_game_result()
        return self.get_game_result()

    def is_game_over(self):
        # This function checks if the game is over and returns a boolean value.
        #breakpoint()
        if self.board[0] <= 0:
            return True
        # It is important to have elif here, or the game will end before the guerilla player's first move!
        if sum(self.board[33:]) == 0 and self.board[0] < 66:
            return True
        if len(self.checker_positions) == 0:
            return True
        return False
    
    def get_game_result(self):
        # This function returns the result of the game if it's over.
        # 0 = guerilla wins, 1 = COIN wins
        if self.board[0] <= 0:
            return 1
        if sum(self.board[33:]) == 0 and self.board[0] < 66:
            return 1
        if len(self.checker_positions) == 0:
            return -1
        return None

    def get_remaining_stones(self):
        # his function returns the number of remaining Guerrilla stones.
        # Not sure if this is needed
        return self.board[0]
    
    def get_guerilla_moves(self):
        move_list=[]
        # TODO: Find out if it's necessary to make a copy of the board for each move
        # First move, if the guerilla player is still holding all their stones
        if self.board[0] == 66:
            for i in range(48):
                if i < 42:
                    new_board = copy.copy(self.board)
                    new_board[i + 33] = 1
                    new_board[i + 33 + 7] = 1
                    new_board[0] -= 2
                    move_list.append(new_board)
                if (i+1)%7 != 0:
                    new_board = copy.copy(self.board)
                    new_board[i + 33] = 1
                    new_board[i + 33 + 1] = 1
                    new_board[0] -= 2
                    move_list.append(new_board)
        else:
            # Find occupied crosses
            for index, cross in enumerate(self.board[33:]):
                if cross == 1:
                    # TODO: Make sure the name and index of the neighbors list is correct!
                    for neighbor in rules["neighbors"][index]:
                        if self.board[neighbor + 33] == 0:
                            for neighborbor in rules["neighbors"][neighbor]:
                                if self.board[neighborbor + 33] == 0:
                                    new_board = copy.copy(self.board)
                                    new_board[neighbor + 33] = 1
                                    new_board[neighborbor + 33] = 1
                                    new_board = self.check_surround(new_board, self.checker_positions)
                                    new_board[0] -= 2
                                    move_list.append(new_board)
        return move_list

    def check_surround(self, board, positions):
        #breakpoint()
        for position in positions:
            surrounded = True
            for cross in rules["diagonals"][position - 1]:
                if board[cross[1]] == 0:
                    surrounded = False
            if surrounded:
                board[position] = 0
        return board
    
    def get_COIN_moves(self, debug=False):
        move_list=[]

        for position in self.checker_positions:
                # Check for possible moves
                for diagonal in rules["diagonals"][position - 1]:
                    if self.board[diagonal[0] + 1] == 0:
                        new_board = copy.copy(self.board)
                        new_board[position] = 0
                        new_board[diagonal[0] + 1] = 1
                        if self.board[diagonal[1]] == 1:
                            new_board[diagonal[1]] = 0
                            move_list += self.capture_and_move(new_board, diagonal[0] + 1, debug)
                        else:
                            if debug:
                                breakpoint()
                            move_list.append(new_board)
                            # Capture and recurse
        # A move is represented by the board state produed by that move
        return move_list

    def capture_and_move(self, board, position, debug=False):
        new_moves = []
        jumped = False
        for diagonal in rules["diagonals"][position - 1]:
            if board[diagonal[1]] == 1 and board[diagonal[0] + 1] == 0:
                jumped = True
                new_board = copy.copy(board)
                new_board[position] = 0
                new_board[diagonal[0] + 1] = 1
                new_board[diagonal[1]] = 0
                if debug:
                    breakpoint()
                new_moves += self.capture_and_move(new_board, diagonal[0] + 1)
        if not jumped:
            new_moves.append(board)
        return new_moves

def randomised_game(draw=False):
    random_game = game()
    player = -1
    if draw:
        draw_board(random_game.board)
    while not random_game.is_game_over():
        valid_actions = random_game.get_valid_actions(player)
        if len(valid_actions) > 0: # Looks like I need to change how I deal with whose turn it is
            random_game.take_action(player, random.choice(valid_actions))
            player = player * -1
        if draw:
            if random_game.guerillas_turn:
                print('Guerilla', end='')
            else:
                print('COIN', end='')
            print(' turn ' + str(len(random_game.game_record)))
            draw_board(random_game.board)
    winner = random_game.get_game_result()
    if winner == None:
        print("No winner")
    if winner == -1:
        print("Guerilla wins")
    if winner == 1:
        print("COIN wins")
    #breakpoint()
    return random_game.game_record

def two_player_game():
    twoplayergame = game()
    player = -1
    while not twoplayergame.is_game_over():
        valid_actions = twoplayergame.get_valid_actions(player)
        turn_over = False
        while not turn_over:
            if player == -1:
                print('Turn', str(len(twoplayergame.game_record)),'guerillas move')
            if player == 1:
                print('Turn', str(len(twoplayergame.game_record)),'COINs move')
            try:
                move = int(input("You have {} possible moves, please chose one. ".format(str(len(valid_actions)))))
            except ValueError:
                print("Please enter a number.")
            if move in range(len(valid_actions)):
                draw_board(valid_actions[move])
                confirm = str(input("Do you chose this move? (y/n)"))
                if confirm == "y":
                    turn_over = True
                    twoplayergame.take_action(player, valid_actions[move])
                    player = player * -1
            else:
                print("You need to enter a number between 0 and", len(valid_actions)-1)
    winner = twoplayergame.get_game_result()
    if winner == None:
        print("No winner")
    if winner == -1:
        print("Guerilla wins")
    if winner == 1:
        print("COIN wins")
    #breakpoint()
    return twoplayergame.game_record

def one_player_game(human):
    oneplayergame = game()
    player = -1
    while not oneplayergame.is_game_over():
        valid_actions = oneplayergame.get_valid_actions(player)
        turn_over = False
        if player == human:
            while not turn_over:
                draw_board(oneplayergame.board)
                if player == -1:
                    print('Turn', str(len(oneplayergame.game_record)),'guerillas move')
                if player == 1:
                    print('Turn', str(len(oneplayergame.game_record)),'COINs move')
                try:
                    move = int(input("You have {} possible moves, please chose one. ".format(str(len(valid_actions)))))
                except ValueError:
                    print("Please enter a number.")
                if move in range(len(valid_actions)):
                    draw_board(valid_actions[move])
                    confirm = str(input("Do you chose this move? (y/n)"))
                    if confirm == "y":
                        turn_over = True
                        oneplayergame.take_action(player, valid_actions[move])
                        player = player * -1
                else:
                    print("You need to enter a number between 0 and", len(valid_actions)-1)
        else:
            oneplayergame.take_action(player, random.choice(valid_actions))
            player = player * -1
    winner = twoplayergame.get_game_result()
    if winner == None:
        print("No winner")
    if winner == -1:
        print("Guerilla wins")
    if winner == 1:
        print("COIN wins")
    #breakpoint()
    return twoplayergame.game_record
            
def draw_board(board):
    stones, squares, grid = decompress_board(board)
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
    print(u"\u2588\u2588\u259B\u2580\u259C\u2588\u259B\u2580\u259C\u2588\u259B\u2580\u259C\u2588\u259B\u2580\u259C")
    for i, row in enumerate(squares):
        #print(i, row)
        str = ' '
        if i%2 == 0:
            print(black_middle, end='')
            for j, square in enumerate(row):
                if j > 2:
                    if square == 1:
                        str = copyright
                    else: str = " "
                    print(black_middle, str, end=black_left, sep=black_right)
                    
                else:
                    if square == 1:
                        str = copyright
                    else: str = " "
                    print(black_middle, str, end=black_left, sep=black_right)
            print('')
            if i<7:
                print(u"\u259B", end='')
                #print(grid[i, :])
                print(black_bottom, end='')
                for j, cross in enumerate(grid[i, :]):
                    if j%2 == 0:
                        if cross == 0:
                            print(black_corners_b, end=black_top)
                        else:
                            print(blackstone, end=black_top)                        
                    else:
                        if cross == 0:
                            print(black_corners_a, end=black_bottom)
                        else:
                            print(blackstone, end=black_bottom)
                print(u"\u259F")
        else:
            print(black_right, end='')
            for j, square in enumerate(row):
                if j > 2:
                    if square == 1:
                        str = copyright
                    else: str = " "
                    print(str, black_middle, end=black_middle, sep=black_left)
                else:
                    if square == 1:
                        str = copyright
                    else: str = " "
                    print(str, black_middle, end=black_right, sep=black_left)
            print('')
            if i<7:
                print(u"\u2599", end='')
                #print(grid[i, :])
                print(black_top, end='')
                for j, cross in enumerate(grid[i, :]):
                    if j%2 == 0:
                        if cross == 0:
                            print(black_corners_a, end=black_bottom)
                        else:
                            print(blackstone, end=black_bottom)                    
                    else:
                        if cross == 0:
                            print(black_corners_b, end=black_top)
                        else:
                            print(blackstone, end=black_top)    
                print(u"\u259C")
    print(u"\u2599\u2584\u259F\u2588\u2599\u2584\u259F\u2588\u2599\u2584\u259F\u2588\u2599\u2584\u259F\u2588\u2588")

print("There is no AI yet, just random choice.")
while True:
    player_choice = input("Do you want to play with 0, 1 or 2 players? (0/1/2/q)")
    if str(player_choice) == "0":
        randomised_game(draw = True)
        break
    
    if str(player_choice) == "1":
        while True:
            player_side = input("Will you play as guerilla or COIN? (g/c)")
            if player_side == "g":
                one_player_game(-1)
                break
            if player_side == "c":
                one_player_game(1)
                break
            print("You have to type 'g' or 'c'!")
        break
    
    if str(player_choice) == "2":
        two_player_game()
        break
    
    if str(player_choice) == "q":
        print("Bye!")
        break
    print("Incorrect input")