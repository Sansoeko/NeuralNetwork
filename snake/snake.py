"""
A simple game of snake.
"""
import random as rd
import time
import curses
import numpy as np

WIDTH = 30
HEIGHT = 30
SNAKE_CHAR = "#"
POWERUP_CHAR = "*"
EMPTY_CHAR = "."
SPEED = 0.05  # Time between two frames
START_LENGTH = 6


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
        'down': (1, 0),
        'up': (-1, 0),
        'left': (0, -1),
        'right': (0, 1),
    }[direction]
    return (pos[0] + change[0], pos[1] + change[1])


def key_to_direction(key):
    """
    Convert keypress code to direction.
    """
    return {
        258: 'down',
        259: 'up',
        260: 'left',
        261: 'right',
    }[key]


def update_snake(grid, snake, direction):
    """
    Update the snake by removing from tail, adding the next position, and updating the grid.
    """
    # TODO do not remove from tail if powerup is collected
    # TODO reset game when colliding with tail
    grid[snake[0][0] % HEIGHT, snake[0][1] % WIDTH] = EMPTY_CHAR
    del snake[0]
    snake.append(next_pos(snake[-1], direction))
    grid[snake[-1][0] % HEIGHT, snake[-1][1] % WIDTH] = SNAKE_CHAR


def spawn_powerup(grid):
    """
    If there is not yet a powerup, with possibility of x, add powerup to grid.
    """
    # TODO implement here
    pass


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
            # TODO do not accept change of direction to the opposite of current direction
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
