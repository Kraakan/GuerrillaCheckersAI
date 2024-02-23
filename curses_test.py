# https://docs.python.org/3/howto/curses.html

import curses
stdscr = curses.initscr()

# Usually curses applications turn off automatic echoing of keys to the screen,
# in order to be able to read keys and only display them under certain circumstances.
# This requires calling the noecho() function.

#curses.noecho()

# Applications will also commonly need to react to keys instantly, without requiring
# the Enter key to be pressed; this is called cbreak mode, as opposed to the usual
# buffered input mode.

#curses.cbreak()

# Terminals usually return special keys, such as the cursor keys or navigation keys
# such as Page Up and Home, as a multibyte escape sequence. While you could write your
# application to expect such sequences and process them accordingly, curses can do it
# for you, returning a special value such as curses.KEY_LEFT. To get curses to do the
# job, you’ll have to enable keypad mode.

stdscr.keypad(True)

# A common problem when debugging a curses application is to get your terminal messed up
# when the application dies without restoring the terminal to its previous state.
# Keys are no longer echoed to the screen when you type them, for example, which makes
# using the shell difficult. In Python you can avoid these complications and make debugging
# much easier by importing the curses.wrapper() function and using it like this:

from curses import wrapper
import copy
from guerilla_checkers import create_starting_board, decompress_board


def main(stdscr):
    # Clear screen
    stdscr.clear()

    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_YELLOW, -1)
    curses.init_pair(2, curses.COLOR_MAGENTA, -1)

    board, nothing = create_starting_board()
    #y, x =  stdscr.getyx()
    y = 0
    x = 0
    min_y, min_x = stdscr.getbegyx()
    max_y, max_x = stdscr.getmaxyx()
    while True:
        yy, xx =  curses.getsyx()
        #stdscr.addstr(str(min_y) + " " + str(min_x) + "\n")
        #stdscr.addstr(str(max_y) + " " + str(max_x))
        #stdscr.addstr("\n")
        #stdscr.addstr(str(y) + "\n")
        #stdscr.addstr(str(x))
        #stdscr.addstr("\n")
        #stdscr.move(min_y, min_x)
        stdscr.move(0, 0)
        test_colors(stdscr)
        c = stdscr.getch()
        stdscr.clear()
        if c == ord('q'):
            break
        elif c == curses.KEY_MOUSE:
            mouse_event = curses.getmouse()
            stdscr.addstr(str(mouse_event))
        elif c == ord('a') or c == curses.KEY_LEFT:
            x -= 1
        
        elif c == ord('d') or c == curses.KEY_RIGHT:
            x += 1
            
        elif c == ord('w') or c == curses.KEY_UP:
            y -= 1
        elif c == ord('s') or c == curses.KEY_DOWN:
            y += 1
            
        if x > max_x:
            x = max_x
        if y > max_y:
            y = max_y
        if x < min_x:
            x = min_x
        if y < min_y:
            y = min_y
        stdscr.addstr(str(y),curses.A_BLINK)
        stdscr.addstr(str(x),curses.A_STANDOUT)
        
        stdscr.refresh()
        #stdscr.addstr(str(c))
   

def draw_board_with_curses(board, x, y, move = None):
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
    stdscr.addstr("  A B C D E F G H\n")
    stdscr.addstr(u" \u2588\u2588\u259B\u2580\u259C\u2588\u259B\u2580\u259C\u2588\u259B\u2580\u259C\u2588\u259B\u2580\u259C")
    
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
    # TODO: return available coordinates

def test_colors(stdscr):
    curses.start_color()
    curses.use_default_colors()
    stdscr.addstr("colors changeable? " + str(curses.can_change_color()))
    for i in range(0, curses.COLORS):
        curses.init_pair(i + 1, i-1, i)
    for i in range(0, curses.COLORS):
        # Medium shade: u"\u2592"
        stdscr.addstr(str(i) + u": \u2591 \u2592 \u2593 ", curses.color_pair(i))
        colors = curses.color_content(i)
        try:
            # color_content()
            rgb = ''
            
            for nr, color in enumerate(colors):
                rgb += str(" ")
                rgb += str(color)
        except:
            rgb = "error"
        stdscr.addstr(rgb)
    stdscr.getch()


wrapper(main)

# Terminating a curses application is much easier than starting one. Call:

curses.nocbreak()
stdscr.keypad(False)
curses.echo()

#to reverse the curses-friendly terminal settings. Then call the endwin() function to restore the terminal to its original operating mode.

curses.endwin()