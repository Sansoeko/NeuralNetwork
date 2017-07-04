"""
A simple game of snake.
"""
import random as rd
import time
import curses
import numpy as np

WIDTH = 30
HEIGHT = 30
SNAKE_CHAR = "X"
POWERUP_CHAR = "o"
EMPTY_CHAR = "."
SPEED = 0.12  # Time between two frames
START_LENGTH = 6
POWERUP = False #is powerup spawned?
X = 0.2 #change powerup is spawned every frame


def clean_grid():
    """
    Creates a grid.
    """
    grid = np.chararray((HEIGHT, WIDTH))
    grid[:] = EMPTY_CHAR
    return grid


def print_grid(stdscr, grid):
    """
    Print the grid to terminal.
    """
    for row in grid:
        stdscr.addstr(" ".join(row) + "\n")

    stdscr.addstr("\nPress 'q' to quit.")
    stdscr.refresh()
    stdscr.move(0, 0)


def new_game():
    """
    Init the grid, a random direction and a start postion for the snake.
    """
    grid = clean_grid()
    direction = key_to_direction(rd.randint(258, 261))
    snake = [(rd.randint(1, HEIGHT - 1), rd.randint(1, WIDTH - 1))]  # row, col
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
    # TODO reset game when colliding with tail
    global POWERUP
    if grid[next_pos(snake[-1], direction)] == POWERUP_CHAR:
    	snake.append(next_pos(snake[-1], direction))
    	grid[snake[-1][0] % HEIGHT, snake[-1][1] % WIDTH] = SNAKE_CHAR
    	POWERUP = False
    elif grid[next_pos(snake[-1], direction)] == SNAKE_CHAR:
    	grid, snake, direction = new_game()
    else:
    	grid[snake[0][0] % HEIGHT, snake[0][1] % WIDTH] = EMPTY_CHAR
    	del snake[0]
    	snake.append(next_pos(snake[-1], direction))
    	grid[snake[-1][0] % HEIGHT, snake[-1][1] % WIDTH] = SNAKE_CHAR


def spawn_powerup(grid):
    
    """
    If there is not yet a powerup, with possibility of x, add powerup to grid.
    """
    # TODO implement here
    global POWERUP
    global X
    if POWERUP == False:
    	if rd.random() < X:
    		grid[(rd.randint(1, HEIGHT - 1), rd.randint(1, WIDTH - 1))] = POWERUP_CHAR
    		POWERUP = True



def main(stdscr):
    """
    Main game loop.
    """
    stdscr.nodelay(1)
    grid, snake, direction = new_game()
    while True:
        # Keyinput to change direction of snake.
        char = stdscr.getch()
        if char in range(258, 262):
            if not direction == -1*key_to_direction(char):
            	direction = key_to_direction(char)
        if char == 113:
            break

        # Update game state
        update_snake(grid, snake, direction)
        spawn_powerup(grid)

        print_grid(stdscr, grid)
        time.sleep(SPEED)


if __name__ == "__main__":
    curses.wrapper(main)
