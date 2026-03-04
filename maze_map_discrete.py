import numpy as np
import matplotlib.pyplot as plt
import random
import sys

sys.setrecursionlimit(100000)

def generate_maze(width, height):
    # Make dimensions odd to make sure the first and last columns and rows are walls 
    # OR maze is bounded by walls in all 4 directions
    width = width if width % 2 == 1 else width + 1
    height = height if height % 2 == 1 else height + 1

    maze = np.ones((height, width))  # start full of walls

    def carve(x, y):
        directions = [(2,0), (-2,0), (0,2), (0,-2)]
        random.shuffle(directions)  # choose random directions each time to prevent the same pattern

        # Loop through all possible direction until finding a valid one
        for dx, dy in directions:
            nx, ny = x + dx, y + dy     # compute coordinates of new cell

            if 0 < nx < (width - 1) and 0 < ny < (height - 1):
                if maze[ny][nx] == 1:   # check if we've already been at the cell/if cell is wall
                    maze[ny][nx] = 0    # set new cell to open
                    maze[y + dy//2][x + dx//2] = 0      # open the middle cell between old cell and new cell
                    carve(nx, ny)

    # Start carving from (1,1)
    maze[1][1] = 0      # set (1, 1) to zero to show we've already been at the cell
    carve(1,1)

    return maze

maze = generate_maze(60, 60)

plt.figure(figsize=(6,6))
plt.imshow(maze, cmap='gray_r', interpolation='nearest')    # reverse grayscale color map so that 1 is black and 0 is white
plt.axis('on')
plt.show()