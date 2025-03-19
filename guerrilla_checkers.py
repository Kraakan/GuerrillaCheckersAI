import numpy as np
import copy
import random
import pickle
import json

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

def dictify_board(stones, squares, grid):
    board = {"stones": stones}
    squares_dict = {}
    for iy, ix in np.ndindex(squares.shape):
        squares_dict[(iy,ix)] = squares[iy,ix]
    board["squares"] = squares_dict
    grid_dict = {}
    for iy, ix in np.ndindex(grid.shape):
        grid_dict[(iy,ix)] = grid[iy,ix]
    board["grid"] = grid_dict
    return board

def decompress_board(board):
    stones = board[0]
    squares_list = board[1:33]
    squares = np.array(squares_list).reshape((8, 4))
    #TODO add empty squares?
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
    starting_board_dict = dictify_board(66,squares,grid)
    return starting_board, starting_board_dict

# IMPORTANT! Make sure the positions lists are always up to date
def list_checker_positions(board):
    checker_list = []
    for index, square in enumerate(board[1:33]):
        if square == 1:
            checker_list.append(index + 1)
    return checker_list

try:
    rules = pickle.load( open( "rules.pickle", "rb" ))
    #breakpoint()
except FileNotFoundError:
    # Generating "rules" - mostly about the board layout
    print('Generating rules...')

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

    list_diagonal_crosses(diagonals)
    
    # This method changes the mess above to a dict of tuples representing all the coordinates involved
    def dictify_diagonals(list_, width, height):
        return_dict = {}
        row = 0
        column = 0
        i = 0
        while row < height:
            while column < width:
                for target in list_[i]:
                    target_tuple = (target[0] // 4, target[0] % 4)
                    cross_index = target[1] - 33
                    cross = (cross_index // 7, cross_index % 7)
                    return_dict[(row, column) + target_tuple] = cross
                column +=1
                i +=1
            row +=1
            column = 0
        return return_dict

    diagonal_dict = dictify_diagonals(diagonals, 4, 8)

    starting_board, starting_board_dict = create_starting_board()
    COIN_moves = []
    guerrilla_moves = []
    # I'm naming the tuple-coordinates to make the next part less confusing
    y = 0
    x = 1
    for origin in starting_board_dict["squares"]:
        if origin[y] % 2 == 0:
            if origin[y] > 0:
                COIN_moves.append(origin + (origin[y] - 1,origin[x]))
                if origin[x] < 3:
                    COIN_moves.append(origin + (origin[y] - 1,origin[x] + 1))
            if origin[y] < 7:
                COIN_moves.append(origin + (origin[y] + 1,origin[1]))
                if origin[x] < 3:
                    COIN_moves.append(origin + (origin[y] + 1,origin[x] + 1))
        else:
            if origin[y] > 0:
                if origin[x] > 0:
                    COIN_moves.append(origin + (origin[y] - 1,origin[x] - 1))
                COIN_moves.append(origin + (origin[y] - 1,origin[x]))
            if origin[y] < 7:
                if origin[x] > 0:
                    COIN_moves.append(origin + (origin[y] + 1,origin[x] - 1))
                COIN_moves.append(origin + (origin[y] + 1, origin[x]))

    for origin in starting_board_dict["grid"]:
        if origin[y] > 0:
            guerrilla_moves.append(origin + (origin[y] - 1,origin[x]))
        if origin[y] < 6:
            guerrilla_moves.append(origin + (origin[y] + 1,origin[x]))
        if origin[x] > 0:
            guerrilla_moves.append(origin + (origin[y],origin[x] - 1))
        if origin[x] < 6:
            guerrilla_moves.append(origin + (origin[y],origin[x] + 1))

    all_guerrilla_moves  = dict.fromkeys(guerrilla_moves, False)
    all_COIN_moves  = dict.fromkeys(COIN_moves, False)
    rules = {
        "diagonals" : diagonals,
        "neighbors" : neighbors,
        "starting board" : starting_board,
        "checker positions" : frozenset({10, 14, 15, 18, 19, 23}),
        "all COIN moves" : all_COIN_moves,
        "all guerrilla moves" : all_guerrilla_moves,
        "diagonal dict": diagonal_dict
    }
    
    pickle.dump(rules, open( "rules.pickle", "wb" ))
try:
    starting_board_file = open("starting_board.json", "r")
    starting_board = json.load(starting_board_file)
except:
    with open('starting_board.json', 'w') as f:
        json.dump(rules["starting board"], f, indent=4)

class game():
    # Game object, will probably be instantiated for each game
    def __init__(self, num_checkers=6, small_reward_factor = 1.0, big_reward_factor = 1.0):
        self.board = rules["starting board"]
        self.starting_checkers_num = num_checkers
        self.small_reward_factor = small_reward_factor
        self.big_reward_factor = big_reward_factor
        self.initialize_checkers()
        self.guerrillas_turn = True
        # NOTE: I may want to remove or disable game_record for training!
        self.game_record = [self.board]
        self.COINjump = None
        
    def initialize_checkers(self):
        if self.starting_checkers_num < 1 or self.starting_checkers_num > 5:
            self.checker_positions = list(rules["checker positions"])
        else:
            self.checker_positions = random.sample(list(rules["checker positions"]), k=self.starting_checkers_num)
            for index in list(rules["checker positions"]):
                self.board[index] = 0
            for index in self.checker_positions:
                self.board[index] = 1

    def reset(self):
        self.board = rules["starting board"]
        self.initialize_checkers()
        self.guerrillas_turn = True
        self.game_record = [self.board]
        self.COINjump = None
        normalized_board = copy.copy(self.board)
        normalized_board[0] /= 66 
        return normalized_board
        
    def get_current_state(self):
        # This function returns the current state of the game.
        normalized_board = copy.copy(self.board)
        normalized_board[0] /= 66
        return normalized_board, self.guerrillas_turn
    
    def get_valid_action_indexes(self, player):
        indexes = []
        if player == 1: # guerrilla
            if self.guerrillas_turn:
                #indexes = self.get_guerrilla_moves()
                    move_list = list(rules["all guerrilla moves"].keys())
                    # TODO: Find out if it's necessary to make a copy of the board for each move
                    # First move, if the guerrilla player is still holding all their stones
                    if self.board[0] == 66:
                        # Permit all moves
                        indexes = list(range(len(move_list)))
                    else:
                        # Find occupied crosses
                        for index, cross in enumerate(self.board[33:]):
                            if cross == 1:
                                cross_index = index + 33
                                # TODO: Start ADJACENT, not on occupied crosses!
                                free_crosses = []
                                # UP
                                if cross_index - 7 > 33 and self.board[cross_index - 7] == 0:
                                    free_crosses.append(cross_index - 7 - 33)
                                # DOWN
                                if cross_index + 7 < 82 and self.board[cross_index + 7] == 0:
                                    free_crosses.append(cross_index + 7 - 33)
                                # LEFT
                                if (cross_index - 33) % 7 > 0 and self.board[cross_index - 1] == 0:
                                    free_crosses.append(cross_index - 1 - 33)
                                # RIGHT
                                if (cross_index -33) % 7 < 6 and self.board[cross_index + 1] == 0:
                                    free_crosses.append(cross_index + 1 - 33)
                                for free_cross in free_crosses:
                                    cross_y = free_cross // 7
                                    cross_x = free_cross % 7
                                    if (cross_y, cross_x, cross_y - 1, cross_x) in move_list:
                                        # Check if up is occupied
                                        if self.board[free_cross - 7 + 33] == 0:
                                            indexes.append(move_list.index((cross_y, cross_x, cross_y - 1, cross_x)))
                                    if (cross_y, cross_x, cross_y + 1, cross_x) in move_list:
                                        # Check if down is occupied
                                        if self.board[free_cross + 7 + 33] == 0:
                                            indexes.append(move_list.index((cross_y, cross_x, cross_y + 1, cross_x)))
                                    if (cross_y, cross_x, cross_y, cross_x - 1) in move_list:
                                        # Check if left is occupied
                                        if cross_x > 0 and self.board[free_cross - 1 + 33] == 0:
                                            indexes.append(move_list.index((cross_y, cross_x, cross_y, cross_x - 1)))
                                    if (cross_y, cross_x, cross_y, cross_x + 1) in move_list:
                                        # Check if right is occupied
                                        if cross_x < 6 and self.board[free_cross + 1 + 33] == 0:
                                            indexes.append(move_list.index((cross_y, cross_x, cross_y, cross_x + 1)))
            else:
                print("Wrong player! It's not guerrillas_turn")
                indexes = []
        else: # COIN
            if not self.guerrillas_turn:
                move_list = list(rules["all COIN moves"].keys())
                #move_dict = list(rules["all COIN moves"])
                if self.COINjump == None:
                    for position in self.checker_positions:
                        abs_pos = position - 1
                        for diagonal in rules["diagonals"][abs_pos]:
                            if self.board[diagonal[0] + 1] == 0:
                                indexes.append(move_list.index((abs_pos // 4, abs_pos % 4, diagonal[0] // 4, diagonal[0] % 4)))
                else:
                    for diagonal in rules["diagonals"][self.COINjump[0] - 1]:
                        if self.board[diagonal[1]] == 1 and self.board[diagonal[0] + 1] == 0:
                            abs_jump = self.COINjump[0] - 1
                            abs_diag = diagonal[0]
                            indexes.append(move_list.index((abs_jump // 4, abs_jump % 4, abs_diag // 4, abs_diag % 4)))
            else:
                print("Wrong player! It's guerrillas_turn")
                indexes = []
        return indexes
    
    def get_valid_actions(self, player):
        # This function takes the current player as input and returns a list of valid actions for that player.
        
        # An acton for either player can be defined by two spots on the board:
        # For the guerrilla player the first spot is where they place one stone, 
        # and the second place is where they place a second stone next to the first.
        # For the COIN player the first spot is the location one of their checkers that is able to move,
        # and the second spot is where it ends up.
        
        moves = None
        #if self.board[0] < 66:
            #self.checker_positions = list_checker_positions(self.board)
        # player: 1 = guerrilla 0 = COIN
        # Not sure if I need to check whose turn it is, 
        # or if returning an empty list is the right response for players "acting out of turn"
        if player == 1:
            if self.guerrillas_turn:
                moves = self.get_guerrilla_moves()
            else:
                print("Wrong player! It's not guerrillas_turn")
                moves = {}
        else:
            if not self.guerrillas_turn:
                moves = self.get_COIN_moves()
            else:
                print("Wrong player! It's guerrillas_turn")
                moves = {}
        return moves

    def is_game_over(self):
        # This function checks if the game is over and returns a boolean value.
        if self.board[0] <= 0:
            return True
        if sum(self.board[33:]) == 0 and self.board[0] < 66:
            return True
        self.checker_positions = list_checker_positions(self.board)
        if len(self.checker_positions) == 0:
            return True
        if self.guerrillas_turn:
            valid_actions_dict = self.get_guerrilla_moves()
            valid_actions_list = [k for k, v in valid_actions_dict.items() if v]
            if len (valid_actions_list) < 1:
                return True
        return False
    
    def get_game_result(self):
        # This function returns the result of the game if it's over.
        # -1 = guerrilla wins, 1 = COIN wins, 0 = game isn't over
        if self.board[0] <= 0:
            return 1
        if sum(self.board[33:]) == 0 and self.board[0] < 66:
            return 1
        self.checker_positions = list_checker_positions(self.board) # Unsure if this is redundant, let's hope it doesn't break things
        if len(self.checker_positions) == 0:
            return -1
        if self.guerrillas_turn:
            valid_actions_dict = self.get_guerrilla_moves()
            valid_actions_list = [k for k, v in valid_actions_dict.items() if v]
            if len (valid_actions_list) < 1:
                return 1
        return 0

    def get_small_reward(self, player):
        reward = len(self.checker_positions)/6
        reward = reward * self.small_reward_factor
        # Time penalty, might be useful if:
        # 1. It's small enough
        # 2. It applies correctly to both players
        # reward -= self.board[0]/66
        if player == 1:
            reward *= -1
        return reward
    
    def set_small_reward_factor(self, new_factor):
        self.small_reward_factor = new_factor
    
    def set_big_reward_factor(self, new_factor):
        self.big_reward_factor = new_factor
    
    def get_remaining_stones(self):
        # his function returns the number of remaining Guerrilla stones.
        # Not sure if this is needed
        return self.board[0]

    def check_surround(self, board, positions):
        for position in positions:
            surrounded = True
            for cross in rules["diagonals"][position - 1]:
                if board[cross[1]] == 0:
                    surrounded = False
            if surrounded:
                board[position] = 0
        return board

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
    
    def get_guerrilla_moves(self):
        move_dict = copy.copy(rules["all guerrilla moves"])
        # TODO: Find out if it's necessary to make a copy of the board for each move
        # First move, if the guerrilla player is still holding all their stones
        if self.board[0] == 66:
            # Permit all moves
            move_dict = dict.fromkeys(move_dict, True)
        else:
            # Find occupied crosses
            for index, cross in enumerate(self.board[33:]):
                if cross == 1:
                    cross_index = index + 33
                    all_moves = list(move_dict.keys())
                    # TODO: Start ADJACENT, not on occupied crosses!
                    free_crosses = []
                    # UP
                    if cross_index - 7 > 33 and self.board[cross_index - 7] == 0:
                        free_crosses.append(cross_index - 7 - 33)
                    # DOWN
                    if cross_index + 7 < 82 and self.board[cross_index + 7] == 0:
                        free_crosses.append(cross_index + 7 - 33)
                    # LEFT
                    if (cross_index - 33) % 7 > 0 and self.board[cross_index - 1] == 0:
                        free_crosses.append(cross_index - 1 - 33)
                    # RIGHT
                    if (cross_index -33) % 7 < 6 and self.board[cross_index + 1] == 0:
                        free_crosses.append(cross_index + 1 - 33)
                    for free_cross in free_crosses:
                        cross_y = free_cross // 7
                        cross_x = free_cross % 7
                        if (cross_y, cross_x, cross_y - 1, cross_x) in all_moves:
                            # Check if up is occupied
                            if self.board[free_cross - 7 + 33] == 0:
                                move_dict[(cross_y, cross_x, cross_y - 1, cross_x)] = True
                        if (cross_y, cross_x, cross_y + 1, cross_x) in all_moves:
                            # Check if down is occupied
                            if self.board[free_cross + 7 + 33] == 0:
                                move_dict[(cross_y, cross_x, cross_y + 1, cross_x)] = True
                        if (cross_y, cross_x, cross_y, cross_x - 1) in all_moves:
                            # Check if left is occupied
                            if cross_x > 0 and self.board[free_cross - 1 + 33] == 0:
                                move_dict[(cross_y, cross_x, cross_y, cross_x - 1)] = True
                        if (cross_y, cross_x, cross_y, cross_x + 1) in all_moves:
                            # Check if right is occupied
                            if cross_x < 6 and self.board[free_cross + 1 + 33] == 0:
                                move_dict[(cross_y, cross_x, cross_y, cross_x + 1)] = True
        return move_dict
                                          
    def old_get_guerrilla_moves(self):
        move_list=[]
        # TODO: Find out if it's necessary to make a copy of the board for each move
        # First move, if the guerrilla player is still holding all their stones
        if self.board[0] == 66:
            for i in range(48):
                if i < 42:
                    # Vertical orientation
                    new_move = (i + 33, i + 33 + 7)
                    move_list.append(new_move)
                if (i+1)%7 != 0:
                    # Horizontal orientation
                    new_move = (i + 33, i + 33 + 1)
                    move_list.append(new_move)
        else:
            # Find occupied crosses
            for index, cross in enumerate(self.board[33:]):
                if cross == 1:
                    # TODO: Make sure the name and index of the neighbors list is correct!
                    for neighbor in rules["neighbors"][index]:
                        if self.board[neighbor + 33] == 0:
                            for neighborbor in rules["neighbors"][neighbor]:
                                if self.board[neighborbor + 33] == 0:
                                    new_move = (neighbor + 33, neighborbor + 33)
                                    move_list.append(new_move)
        return move_list
        
    def take_action(self, player, action):
        # This function takes the current player and an action as input, updates the game state based on the action, checks if the game has ended, and returns the outcome.
        #result = None
        new_board = copy.copy(self.board)
        if player == 1 and self.guerrillas_turn:
            # guerrilla
            new_board[0] -= 2
            first = action[0] * 7 + action[1] + 33
            second = action[2] * 7 + action[3] + 33
            new_board[first] = 1
            new_board[second] = 1
            new_board = self.check_surround(new_board, self.checker_positions)
            self.guerrillas_turn = False

        if (player == 0 and not self.guerrillas_turn):
            # COIN
            first = action[0] * 4 + action[1] + 1
            second = action[2] * 4 + action[3] + 1
            new_board[first] = 0
            new_board[second] = 1
            # If the COIN player has captured a guerrilla stone, and there is one or more stones that can be captured with the same piece, they have to do so
            #diagonals = rules["diagonals"][action[0] - 1]
            #index = [square for square, cross in diagonals].index(action[1] - 1)
            #diagonal = diagonals[index]
            cross_tuple = rules["diagonal dict"][action]
            cross_index = cross_tuple[0] * 7 + cross_tuple[1] + 33
            self.guerrillas_turn = True
            self.COINjump = None
            if self.board[cross_index] == 1:
                new_board[cross_index] = 0
                for new_diagonal in rules["diagonals"][second - 1]:
                    if new_board[new_diagonal[0] + 1] == 0 and new_board[new_diagonal[1]] == 1:
                        self.COINjump = (second, cross_index)
                        self.guerrillas_turn = False
        self.board = new_board
        normalized_board = copy.copy(self.board)
        normalized_board[0] /= 66 
        self.game_record.append(self.board)
        reward = self.get_reward(player)
        terminated = self.is_game_over()
        return (normalized_board, reward, terminated)
    
    def get_reward(self, player):
        COIN_reward = self.get_game_result() * self.big_reward_factor
        if player == 1: # guerrilla
            return -1 * COIN_reward
        else:
            return COIN_reward
        
    def get_COIN_moves(self, debug=False):
        move_dict = copy.copy(rules["all COIN moves"])
        if self.COINjump == None:
            for position in self.checker_positions:
                abs_pos = position - 1
                for diagonal in rules["diagonals"][abs_pos]:
                    if self.board[diagonal[0] + 1] == 0:
                        #breakpoint()
                        move_dict[(abs_pos // 4, abs_pos % 4, diagonal[0] // 4, diagonal[0] % 4)] = True
        else:
            for diagonal in rules["diagonals"][self.COINjump[0] - 1]:
                if self.board[diagonal[1]] == 1 and self.board[diagonal[0] + 1] == 0:
                    abs_jump = self.COINjump[0] - 1
                    abs_diag = diagonal[0]
                    move_dict[(abs_jump // 4, abs_jump % 4, abs_diag // 4, abs_diag % 4)] = True
        return move_dict
                          
def draw_board(board, move = None):
    if move != None:
        board = copy.copy(board)
        # TODO: CONVERT!
        if move[0] > 32:
            board[move[0]] = 1
        else: board[move[0]] = 0
        board[move[1]] = 1
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
