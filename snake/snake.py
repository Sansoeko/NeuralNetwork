"""
A simple game of snake.
"""
import random as rd
import time
import curses
import numpy as np

WIDTH = 42
HEIGHT = 42
SNAKE_CHAR = "X"
POWERUP_CHAR = "Q"
EMPTY_CHAR = "."
WALL_CHAR = "#"
SPEED = 0.24  # Time between two frames
START_LENGTH = 6
POWERUP = False #is powerup spawned?
X = 0.2 #change powerup is spawned every frame
LIVE = True
SCORE = 0 #ik heb 26 gehaald d;


def clean_grid():
    """
    Creates a grid.
    """
    grid = np.chararray((HEIGHT, WIDTH))
    grid[:] = EMPTY_CHAR
    grid[0:HEIGHT,0] = WALL_CHAR
    grid[0:HEIGHT,WIDTH-1] = WALL_CHAR
    grid[0,1:WIDTH] = WALL_CHAR
    grid[HEIGHT-1,1:WIDTH] = WALL_CHAR
    return grid


def print_grid(stdscr, grid):
    """
    Print the grid to terminal.
    """
    for row in grid:
        stdscr.addstr(" ".join(row) + "\n")

    stdscr.addstr("\nPress 'q' to quit.")
    stdscr.addstr("\nYOUR SCORE: " + str(SCORE) + "!!!")
    stdscr.refresh()
    stdscr.move(0, 0)


def new_game():
    """
    Init the grid, a random direction and a start postion for the snake.
    """
    grid = clean_grid()
    direction = key_to_direction(rd.randint(258, 261))
    snake = [(rd.randint(5, HEIGHT - 5), rd.randint(5, WIDTH - 5))]  # row, col
    for _ in range(START_LENGTH - 1):
        snake.append(next_pos(snake[-1], direction))
    return grid, snake, direction


def next_pos(pos, direction):
    """
    Get next position based on current position and direction.
    """
    change = {
        -1: (1, 0),
        1: (-1, 0),
        -2: (0, -1),
        2: (0, 1),
    }[direction]
    return (pos[0] + change[0], pos[1] + change[1])


def key_to_direction(key):
    """
    Convert keypress code to direction.
    """
    return {
        258: -1, 	#Down
        259: 1,  	#Up
        260: -2, 	#Left
        261: 2,		#Right
    }[key]


def update_snake(grid, snake, direction):
    """
    Update the snake by removing from tail, adding the next position, and updating the grid.
    """
    global POWERUP
    global SPEED
    global SCORE

    if grid[next_pos(snake[-1], direction)] == POWERUP_CHAR:
    	snake.append(next_pos(snake[-1], direction))
    	grid[snake[-1][0] % HEIGHT, snake[-1][1] % WIDTH] = SNAKE_CHAR
    	POWERUP = False
    	if not SPEED < 0.04:
    		SPEED = SPEED * 0.88
    	SCORE = SCORE + 1
    	return True
    elif grid[next_pos(snake[-1], direction)] == SNAKE_CHAR:
    	return False
    elif grid[next_pos(snake[-1], direction)] == WALL_CHAR:
    	return False
    else:
    	grid[snake[0][0] % HEIGHT, snake[0][1] % WIDTH] = EMPTY_CHAR
    	del snake[0]
    	snake.append(next_pos(snake[-1], direction))
    	grid[snake[-1][0] % HEIGHT, snake[-1][1] % WIDTH] = SNAKE_CHAR
    	return True


def spawn_powerup(grid):
    
    """
    If there is not yet a powerup, with possibility of x, add powerup to grid.
    """
    global POWERUP
    global X
    if POWERUP == False:
    	if rd.random() < X:
    		grid[(rd.randint(2, HEIGHT - 2), rd.randint(2, WIDTH - 2))] = POWERUP_CHAR
    		POWERUP = True



def main(stdscr):
    """
    Main game loop.
    """
    stdscr.nodelay(1)
    grid, snake, direction = new_game()
    global LIVE
    global SCORE
    while LIVE:
        # Keyinput to change direction of snake.
        char = stdscr.getch()
        if char in range(258, 262):
            if not direction == -1*key_to_direction(char):
            	direction = key_to_direction(char)
        if char == 113:
            break

        # Update game state
        LIVE = update_snake(grid, snake, direction)
        spawn_powerup(grid)

        print_grid(stdscr, grid)
        time.sleep(SPEED)
    

if __name__ == "__main__":
    curses.wrapper(main)
