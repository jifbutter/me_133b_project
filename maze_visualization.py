# '''rrt_maze.py

#    RRT in a continuous maze (thin line-segment walls from gym-continuous-maze style).
#    Uses continuous_maze.walls and obstacles; connectsTo like rrttriangles (LineString + one check).
# '''

# import matplotlib.pyplot as plt
# import numpy as np
# import random

# from math import sqrt
# from shapely.geometry import MultiLineString
# from shapely.geometry import LineString
# from shapely.geometry   import Point, LineString, Polygon, MultiPolygon
# from shapely.prepared import prep
# from continuous_maze import walls, obstacles




# ######################################################################
# #
# #   Parameters
# #
# #   Define the step size and fraction of time to target the goal.
# #   Also set the maximum number of nodes.
# #
# #                  (a)           (b)               (c)
# #     DSTEP    =  0.25    0.25  0.25  0.25     1.00  5.00    
# #     GOALFRAC =     0    0.05  0.50  0.99     0.05  0.05
# #
# DSTEP    = 0.25
# GOALFRAC = 0.10
# CLEARANCE = 0.3

# # Maximum number of steps (attempts) or nodes (successful steps).
# SMAX = 50000
# NMAX = 5000
# R=0.95
# RADIUS=1.0
# ALPHA=1.0
# BETA=0



# # walls_prep = obstacles          # already prepared
# # walls_geom = obstacles.context  # this is the MultiLineString

# # walls_inflated_prep = prep(walls_geom.buffer(CLEARANCE, cap_style=2, join_style=2))

# ######################################################################
# #
# #   World Definitions
# #
# #   List of obstacles/objects as well as the start/goal.
# #
# (xmin, xmax) = (-12, 12)
# (ymin, ymax) = (-12, 12)

# GRID = 1.0
# NX = int(np.ceil((xmax - xmin) / GRID))
# NY = int(np.ceil((ymax - ymin) / GRID))

# counts_start = np.zeros((NX, NY), dtype=np.int32)  # tree growing from start
# counts_goal  = np.zeros((NX, NY), dtype=np.int32)  # tree growing from goal

# def cell(x, y):
#     i = int((x - xmin) / GRID)
#     j = int((y - ymin) / GRID)
#     # clamp to valid range
#     i = max(0, min(NX - 1, i))
#     j = max(0, min(NY - 1, j))
#     return i, j
# # Maze is continuous_maze.walls (thin line segments). No Shapely obstacles.

# # Start at center, goal in upper-right area (inside the gym maze bounds).
# (xstart, ystart) = (-9, 11)
# # (xstart, ystart) = (0, 0)
# # (xgoal,  ygoal)  = (9, 8)
# (xgoal,  ygoal)  = (-1, 3)
# # (xgoal,  ygoal)  = (5, -3)


# ######################################################################
# #
# #   Visualization Class
# #
# #   This renders the world.  In particular it provides the methods:
# #     show(text = '')                   Show the current figure
# #     drawNode(node,         **kwargs)  Draw a single node
# #     drawEdge(node1, node2, **kwargs)  Draw an edge between nodes
# #     drawPath(path,         **kwargs)  Draw a path (list of nodes)
# #
# class Visualization:
#     def __init__(self):
#         # Clear the current, or create a new figure.
#         plt.clf()

#         # Create a new axes, enable the grid, and set axis limits.
#         plt.axes()
#         plt.grid(True)
#         plt.gca().axis('on')
#         plt.gca().set_xlim(xmin, xmax)
#         plt.gca().set_ylim(ymin, ymax)
#         plt.gca().set_aspect('equal')

#         # Show the maze walls (thin line segments).
#         from binary_maze import walls1, walls2  #as maze_walls
#         for wall in maze_walls:
#             plt.plot([wall[0][0], wall[1][0]], [wall[0][1], wall[1][1]], 'k-', linewidth=2)

#         # Show immediately.
#         self.show()

#     def show(self, text = ''):
#         # Show the plot.
#         plt.pause(0.001)
#         # If text is specified, print and wait for confirmation.
#         if len(text)>0:
#             input(text + ' (hit return to continue)')

#     def drawNode(self, node, **kwargs):
#         plt.plot(node.x, node.y, **kwargs)

#     def drawEdge(self, head, tail, **kwargs):
#         plt.plot([head.x, tail.x], [head.y, tail.y], **kwargs)

#     def drawPath(self, path, **kwargs):
#         for i in range(len(path)-1):
#             self.drawEdge(path[i], path[i+1], **kwargs)


# def main():
#     # Report the parameters.
#     print('Running with step size ', DSTEP, ' and up to ', NMAX, ' nodes.')

#     # Create the figure.  Some computers seem to need an additional show()?
#     visual = Visualization()
#     visual.show()

#     # Create the start/goal nodes.
#     # startnode = Node(xstart, ystart)
#     # goalnode  = Node(xgoal,  ygoal)

#     # # Show the start/goal nodes.
#     # visual.drawNode(startnode, color='orange', marker='o')
#     # visual.drawNode(goalnode,  color='purple', marker='o')
#     # visual.show("Showing basic world")


#     # # Run the RRT planner.
#     # print("Running RRT...")
#     # # path = rrt_connect(startnode, goalnode, visual)
#     # path=rrt_connect_star(startnode, goalnode, visual)

#     # # If unable to connect, just note before closing.
#     # if not path:
#     #     visual.show("UNABLE TO FIND A PATH")
#     #     return

#     # # Show the path.
#     # cost = pathCost(path)
#     # visual.drawPath(path, color='r', linewidth=2)
#     # visual.show("Showing the raw path (cost/length %.1f)" % cost)


#     # # Post process the path.
#     # finalpath = postProcess(path)

#     # # Show the post-processed path.
#     # cost = pathCost(finalpath)
#     # visual.drawPath(finalpath, color='b', linewidth=2)
#     # visual.show("Showing the post-processed path (cost/length %.1f)" % cost)


# if __name__== "__main__":
#     main()

'''rrt_maze_time_varying.py

   Visualization of a time-varying continuous maze.
   For 0 <= t < 50, show walls1 / obstacles1.
   For 50 <= t <= 100, show walls2 / obstacles2.
'''

import matplotlib.pyplot as plt
import numpy as np
import time

from math import sqrt
from shapely.geometry import MultiLineString, LineString, Point, Polygon, MultiPolygon
from shapely.prepared import prep

# Import the two wall sets and the two prepared obstacle sets
from binary_maze import walls, walls1, walls2, obstacles, obstacles1, obstacles2


######################################################################
# Parameters
######################################################################

DSTEP    = 0.25
GOALFRAC = 0.10
CLEARANCE = 0.3

SMAX = 50000
NMAX = 5000
R = 0.95
RADIUS = 1.0
ALPHA = 1.0
BETA = 0
MAXSPEED=0.5

TMAX = 100.0        # total visualization time
SWITCH_TIME = 5  # switch from maze 1 to maze 2
FRAME_DT = 0.05     # seconds between redraws

(xmin, xmax) = (-12, 12)
(ymin, ymax) = (-12, 12)

(xstart, ystart) = (-9, 11)
(xgoal,  ygoal)  = (2.9, -3)


######################################################################
# Time-dependent maze helpers
######################################################################
# Version 1
# def get_walls(t):
#     # Single 100-second run:
#     return walls1 if t < SWITCH_TIME else walls2

#     # If instead you want it periodic every 100 sec, use:
#     # tau = t % TMAX
#     # return walls1 if tau < SWITCH_TIME else walls2


# def get_obstacles(t):
#     # Single 100-second run:
#     return obstacles1 if t < SWITCH_TIME else obstacles2

#     # Periodic version:
#     # tau = t % TMAX
#     # return obstacles1 if tau < SWITCH_TIME else obstacles2

# def current_phase_name(t):
#     return "maze 1" if t < SWITCH_TIME else "maze 2"

# Version 2
def get_walls(t):
    if t < 0 or t > TMAX:
        raise ValueError(f"t must satisfy 0 <= t <= {TMAX}")
    tau = t % (2.0 * SWITCH_TIME)
    return walls if tau < SWITCH_TIME else walls2

def get_obstacles(t):
    if t < 0 or t > TMAX:
        raise ValueError(f"t must satisfy 0 <= t <= {TMAX}")
    tau = t % (2.0 * SWITCH_TIME)
    return obstacles if tau < SWITCH_TIME else obstacles2

def current_phase_name(t):
    if t < 0 or t > TMAX:
        raise ValueError(f"t must satisfy 0 <= t <= {TMAX}")
    tau = t % (2.0 * SWITCH_TIME)
    return "maze" if tau < SWITCH_TIME else "maze 2"

######################################################################
# Visualization Class
######################################################################

class Visualization:
    def __init__(self):
        plt.ion()  # interactive mode
        self.fig, self.ax = plt.subplots()
        self._setup_axes()
        self.draw_world(0.0)
        self.show()

    def _setup_axes(self):
        self.ax.clear()
        self.ax.grid(True)
        self.ax.axis('on')
        self.ax.set_xlim(xmin, xmax)
        self.ax.set_ylim(ymin, ymax)
        self.ax.set_aspect('equal')

    def draw_world(self, t):
        self._setup_axes()

        maze_walls = get_walls(t)
        for wall in maze_walls:
            self.ax.plot(
                [wall[0][0], wall[1][0]],
                [wall[0][1], wall[1][1]],
                'k-',
                linewidth=2
            )

        self.ax.set_title(f"t = {t:.1f} sec   ({current_phase_name(t)})")

    def show(self, text=''):
        self.fig.canvas.draw_idle()
        plt.pause(0.001)
        if len(text) > 0:
            input(text + ' (hit return to continue)')

    def drawNode(self, node, **kwargs):
        self.ax.plot(node.x, node.y, **kwargs)

    def drawPoint(self, x, y, **kwargs):
        self.ax.plot(x, y, **kwargs)

    def drawEdge(self, head, tail, **kwargs):
        self.ax.plot([head.x, tail.x], [head.y, tail.y], **kwargs)

    def drawPath(self, path, **kwargs):
        for i in range(len(path)-1):
            self.drawEdge(path[i], path[i+1], **kwargs)


######################################################################
# Main
######################################################################

def main():
    print('Showing time-varying maze for', TMAX, 'seconds.')

    visual = Visualization()

    # simple visualization loop for 100 seconds
    start_time = time.perf_counter()

    while True:
        t = time.perf_counter() - start_time
        if t > TMAX:
            break

        visual.draw_world(t)

        # draw start and goal each frame
        visual.drawPoint(xstart, ystart, color='orange', marker='o', markersize=10)
        visual.drawPoint(xgoal,  ygoal,  color='purple', marker='o', markersize=10)

        visual.show()
        time.sleep(FRAME_DT)

    visual.show("Finished 100-second visualization")


if __name__ == "__main__":
    main()