'''rrttriangles_solution.py

   This is the RRT solution code for the 2D triangular problem.

   Use RRT to find a path around polygonal obstacles.

'''

import matplotlib.pyplot as plt
import numpy as np
import random
import time

from math               import inf, pi, sin, cos, atan2, sqrt, ceil, dist

from shapely.geometry   import Point, LineString, Polygon, MultiPolygon
from shapely.prepared   import prep


######################################################################
#
#   Parameters
#
#   Define the step size and fraction of time to target the goal.
#   Also set the maximum number of nodes.
#
#                  (a)           (b)               (c)
#     DSTEP    =  0.25    0.25  0.25  0.25     1.00  5.00    
#     GOALFRAC =     0    0.05  0.50  0.99     0.05  0.05
#
DSTEP    = 0.25
GOALFRAC = 0.10

# Maximum number of steps (attempts) or nodes (successful steps).
SMAX = 50000
NMAX = 1500


######################################################################
#
#   World Definitions
#
#   List of obstacles/objects as well as the start/goal.
#
(xmin, xmax) = (0, 10)
(ymin, ymax) = (0, 12)

# Collect all the triangles and prepare (for faster checking).
obstacles = prep(MultiPolygon([
    Polygon([[7,  3], [3,  3], [3,  4], [7,  3]]),
    Polygon([[5,  5], [7,  7], [4,  6], [5,  5]]),
    Polygon([[9,  2], [8,  7], [6,  5], [9,  2]]),
    Polygon([[1, 10], [7, 10], [4,  8], [1, 10]])]))

# Define the start/goal states (x, y, theta)
(xstart, ystart) = (6, 1)
(xgoal,  ygoal)  = (5, 11)


######################################################################
#
#   Visualization Class
#
#   This renders the world.  In particular it provides the methods:
#     show(text = '')                   Show the current figure
#     drawNode(node,         **kwargs)  Draw a single node
#     drawEdge(node1, node2, **kwargs)  Draw an edge between nodes
#     drawPath(path,         **kwargs)  Draw a path (list of nodes)
#
class Visualization:
    def __init__(self):
        # Clear the current, or create a new figure.
        plt.clf()

        # Create a new axes, enable the grid, and set axis limits.
        plt.axes()
        plt.grid(True)
        plt.gca().axis('on')
        plt.gca().set_xlim(xmin, xmax)
        plt.gca().set_ylim(ymin, ymax)
        plt.gca().set_aspect('equal')

        # Show the triangles.
        for poly in obstacles.context.geoms:
            plt.plot(*poly.exterior.xy, 'k-', linewidth=2)

        # Show immediately.
        self.show()

    def show(self, text = ''):
        # Show the plot.
        plt.pause(0.001)
        # If text is specified, print and wait for confirmation.
        if len(text)>0:
            input(text + ' (hit return to continue)')

    def drawNode(self, node, **kwargs):
        plt.plot(node.x, node.y, **kwargs)

    def drawEdge(self, head, tail, **kwargs):
        plt.plot([head.x, tail.x], [head.y, tail.y], **kwargs)

    def drawPath(self, path, **kwargs):
        for i in range(len(path)-1):
            self.drawEdge(path[i], path[i+1], **kwargs)


######################################################################
#
#   Node Definition
#
class Node:
    #################
    # Initialization:
    def __init__(self, x, y):
        # Define/remember the state/coordinates (x,y).
        self.x = x
        self.y = y

        # Define a parent (cleared for now).
        self.parent = None

    ################
    # Planner functions:
    # Compute the relative distance to another node.
    def distance(self, other):
        return sqrt((other.x - self.x)**2 + (other.y - self.y)**2)

    # Check whether in free space.
    def inFreespace(self):
        if (self.x <= xmin or self.x >= xmax or
            self.y <= ymin or self.y >= ymax):
            return False
        point = Point(self.x, self.y)
        return obstacles.disjoint(point)

    # Check the local planner - whether this connects to another node.
    def connectsTo(self, other):
        line = LineString([(self.x, self.y), (other.x, other.y)])
        return obstacles.disjoint(line)

    ############
    # Utilities:
    # In case we want to print the node.
    def __repr__(self):
        return ("<Point %5.2f,%5.2f>" % (self.x, self.y))

    # Compute/create an intermediate node.  This can be useful if you
    # need to check the local planner by testing intermediate nodes.
    def intermediate(self, other, alpha):
        return Node(self.x + alpha * (other.x - self.x),
                    self.y + alpha * (other.y - self.y))


######################################################################
#
#   RRT Functions
#
def rrt(startnode, goalnode, visual=None):
    # Start the tree with the startnode (set no parent just in case).
    startnode.parent = None
    tree = [startnode]

    # Function to attach a new node to an existing node: attach the
    # parent, add to the tree, and show in the figure.
    def addtotree(oldnode, newnode):
        newnode.parent = oldnode
        tree.append(newnode)
        if visual:
            visual.drawEdge(oldnode, newnode, color='g', linewidth=1)
            visual.show()

    # Loop - keep growing the tree.
    steps = 0
    while True:
        # Determine the target state.
        if (random.uniform(0.0, 1.0) < GOALFRAC):
            targetnode = goalnode
        else:
            targetnode = Node(random.uniform(xmin, xmax),
                              random.uniform(ymin, ymax))

        # Directly determine the distances to the target node.
        distances = np.array([node.distance(targetnode) for node in tree])
        index     = np.argmin(distances)
        grownode  = tree[index]
        d         = distances[index]

        # Determine the next node.
        if (d <= DSTEP):
            newnode = targetnode
        else:
            newnode = grownode.intermediate(targetnode, DSTEP/d)

        # Check whether to attach.
        if newnode.inFreespace() and grownode.connectsTo(newnode):
            addtotree(grownode, newnode)

            # Also try to connect the goal.  Stop growing.
            if (newnode.distance(goalnode) <= DSTEP and
                newnode.connectsTo(goalnode)):
                addtotree(newnode, goalnode)
                break

        # Check whether we should abort - too many steps or nodes.
        steps += 1
        if (steps >= SMAX) or (len(tree) >= NMAX):
            print("Aborted after %d steps and the tree having %d nodes" %
                  (steps, len(tree)))
            return None

    # Build the path.
    path = [goalnode]
    while path[0].parent is not None:
        path.insert(0, path[0].parent)

    # Report and return.
    print("Finished after %d steps and the tree having %d nodes" %
          (steps, len(tree)))
    return path


# Compute the path cost
def pathCost(path):
    cost = 0
    for i in range(1, len(path)):
        cost += path[i-1].distance(path[i])
    return cost

# Post process the path
def postProcess(path):
    shortpath = [path[0]]
    for i in range(2, len(path)):
        if not shortpath[-1].connectsTo(path[i]):
            shortpath.append(path[i-1])
    shortpath.append(path[-1])
    return shortpath


######################################################################
#
#  Main Code
#
def main():
    # Report the parameters.
    print('Running with step size ', DSTEP, ' and up to ', NMAX, ' nodes.')

    # Create the figure.  Some computers seem to need an additional show()?
    visual = Visualization()
    visual.show()

    # Create the start/goal nodes.
    startnode = Node(xstart, ystart)
    goalnode  = Node(xgoal,  ygoal)

    # Show the start/goal nodes.
    visual.drawNode(startnode, color='orange', marker='o')
    visual.drawNode(goalnode,  color='purple', marker='o')
    visual.show("Showing basic world")


    # Run the RRT planner.
    print("Running RRT...")
    path = rrt(startnode, goalnode, visual)

    # If unable to connect, just note before closing.
    if not path:
        visual.show("UNABLE TO FIND A PATH")
        return

    # Show the path.
    cost = pathCost(path)
    visual.drawPath(path, color='r', linewidth=2)
    visual.show("Showing the raw path (cost/length %.1f)" % cost)


    # Post process the path.
    finalpath = postProcess(path)

    # Show the post-processed path.
    cost = pathCost(finalpath)
    visual.drawPath(finalpath, color='b', linewidth=2)
    visual.show("Showing the post-processed path (cost/length %.1f)" % cost)


if __name__== "__main__":
    main()
