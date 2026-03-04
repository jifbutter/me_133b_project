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
CLEARANCE = 0.15

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
    Polygon([[1, 10], [7, 10], [4,  8], [1, 10]]),
    Polygon([[1,2],[1,8],[2,8],[2,2]]),
    Polygon([[3,5],[3,8],[3.5,6]]),
    Polygon([[1,1],[3,2],[5,2],[5,1]]),
    Polygon([[6,8],[9,8],[9,10]])]))



# Define the start/goal states (x, y, theta)
(xstart, ystart) = (6, 1)
(xgoal,  ygoal)  = (4.5, 11)


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

    # def drawEdge(self, head, tail, **kwargs):
    #     plt.plot([head.x, tail.x], [head.y, tail.y], **kwargs)

    def drawEdge(self, head, tail, **kwargs):
        (line,) = plt.plot([head.x, tail.x], [head.y, tail.y], **kwargs)
        return line

    def updateEdge(self, line, head, tail):
        line.set_data([head.x, tail.x], [head.y, tail.y])

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
    def __init__(self, x, y, d=0):
        # Define/remember the state/coordinates (x,y).
        self.x = x
        self.y = y
        self.children=set()
        self.d=d
        self.cost=d
        # Define a parent (cleared for now).
        self.parent = None
        self.edge_handle = None

    ################
    # Planner functions:
    # Compute the relative distance to another node.
    def distance(self, other):
        return sqrt((other.x - self.x)**2 + (other.y - self.y)**2)
    
    def distance_obstacle(self):
        p=Point(self.x,self.y)
        return obstacles.context.boundary.distance(p)
    # Check whether in free space.
    def inFreespace(self):
        if (self.x <= xmin or self.x >= xmax or
            self.y <= ymin or self.y >= ymax):
            return False
        point = Point(self.x, self.y)
        return obstacles.disjoint(point)

    # Check the local planner - whether this connects to another node.
    def connectsTo(self, other, clearance=CLEARANCE):
        line = LineString([(self.x, self.y), (other.x, other.y)])

        # 1) Must be collision-free (no intersection with obstacle interiors)
        if not obstacles.disjoint(line):
            return False

        # 2) Must keep clearance from obstacle boundaries
        if clearance > 0.0:
            # distance from the segment to the boundary of the MultiPolygon
            if obstacles.context.distance(line) < clearance:
                return False
        return True
    
    def update_subtree_cost(self):
        for c in self.children:
            c.cost=self.cost+c.d
            c.update_subtree_cost()
    
    def rewire(self, newparent, d):
        if self.parent is not None:
            self.parent.children.remove(self)
        self.parent=newparent
        self.parent.children.add(self)
        self.d=d
        self.cost=self.parent.cost+d
        self.update_subtree_cost()




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

def addtotree(tree,oldnode,newnode, is_start=True, visual=None):
    newnode.parent = oldnode
    oldnode.children.add(newnode)
    newnode.d=oldnode.distance(newnode)
    newnode.cost=oldnode.cost+newnode.d
    tree.append(newnode)
    if visual:
        # if is_start:
        #     visual.drawEdge(oldnode, newnode, color='g', linewidth=1)
        # else:
        #     visual.drawEdge(oldnode, newnode, color='r', linewidth=1)
        # visual.show()
        if visual:
            color = 'g' if is_start else 'r'
            newnode.edge_handle = visual.drawEdge(oldnode, newnode, color=color, linewidth=1)
            visual.show()


def extend_towards(tree, target, is_start=True, ratio=1, visual=None):
    """
    Extend the tree toward target by repeatedly stepping DSTEP.

    Returns:
        last:    last node added (or the nearest existing node if no progress)
        success: True iff we reached the target (within DSTEP and connected)
        n_added: number of new nodes appended to the tree in this call
    """
    # 1) nearest node in tree
    distances = np.array([node.distance(target) for node in tree])
    index     = int(np.argmin(distances))
    last      = tree[index]
    last0=Node(last.x,last.y)
    n_added = 0
    while True:
        d = last.distance(target)

        # Already there (rare, but possible)
        if d == 0.0:
            return last, True, n_added

        # 2) propose next node
        if d <= DSTEP:
            # clone target so we don't steal a node object from another tree
            newnode = Node(target.x, target.y)
            reached_target = True
        else:
            newnode = last.intermediate(target, DSTEP / d)
            reached_target = False

        # 3) collision checks for this step
        if (not newnode.inFreespace()) or (not last.connectsTo(newnode)):
            # cannot extend further
            return last, False, n_added

        # 4) safe: add it
        addtotree(tree, last, newnode, is_start, visual)
        last = newnode
        n_added += 1

        if last0.distance(newnode)>ratio*d:
            return last, True

        # 5) did we reach the target?
        if reached_target:
            return last, True, n_added
    
def trace(node):
    path=[]
    while node is not None:
        path.append(node)
        node=node.parent
    return path

def rrt_star(startnode,goalnode,visual=None,radius=1.5):
    startnode.parent = None
    startnode.cost=0
    startnode.d=0
    startnode.children=set()
    tree = [startnode]
    goal_n=None
    # Loop - keep growing the tree.
    steps = 0
    while True:
        # Determine the target state.
        # Abort?
        if (steps >= SMAX) or (len(tree) >= NMAX):
            print("Aborted after %d steps and the tree having %d nodes" % (steps, len(tree)))
            break

        if (random.uniform(0.0, 1.0) < GOALFRAC):
            targetnode = Node(goalnode.x, goalnode.y)
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
        # Collision/local planner
        if (not newnode.inFreespace()) or (not grownode.connectsTo(newnode)):
            steps += 1
            continue
            

        neighbors=[n for n in tree if n.distance(newnode) <= radius]
        best_parent = grownode
        best_cost = grownode.cost + grownode.distance(newnode)

        for n in neighbors:
            if not n.connectsTo(newnode):
                continue
            if n.cost+n.distance(newnode)<best_cost:
                best_parent=n
                best_cost=n.cost+n.distance(newnode)
        addtotree(tree, best_parent, newnode, True, visual)

        for n in neighbors:
            if n is newnode.parent:
                continue
            new_cost_to_n=newnode.cost+newnode.distance(n)
            if new_cost_to_n<n.cost and newnode.connectsTo(n):
                n.rewire(newnode, newnode.distance(n))
                if visual:
                    # visual.drawEdge(newnode, n, color='g', linewidth=1)
                    # update the existing line instead of adding a new one
                    if n.edge_handle is None:
                        n.edge_handle = visual.drawEdge(newnode, n, color='g', linewidth=1)
                    else:
                        visual.updateEdge(n.edge_handle, newnode, n)
                    visual.show()
        if newnode.distance(goalnode)<=DSTEP and newnode.connectsTo(goalnode):
            if goal_n is None:
                goal_n = Node(goalnode.x, goalnode.y, newnode.distance(goalnode))
                addtotree(tree, newnode, goal_n, True, visual)
            else:
                cand = newnode.cost + newnode.distance(goal_n)
                if cand < goal_n.cost and newnode.connectsTo(goal_n):
                    goal_n.rewire(newnode, newnode.distance(goal_n))
                    if visual:
                        if goal_n.edge_handle is None:
                            goal_n.edge_handle = visual.drawEdge(newnode, goal_n, color='g', linewidth=1)
                        else:
                            visual.updateEdge(goal_n.edge_handle, newnode, goal_n)
                        visual.show()
        # Check whether we should abort - too many steps or nodes.
        steps += 1

    # Build the path.
    path = [goal_n]
    while path[0].parent is not None:
        path.insert(0, path[0].parent)

    # Report and return.
    print("Finished after %d steps and the tree having %d nodes" %
          (steps, len(tree)))
    return (path, goal_n.cost)


def rrt_connect(startnode, goalnode, visual=None):
    # Start the tree with the startnode (set no parent just in case).
    startnode.parent = None
    goalnode.parent=None
    tree_ini, tree_goal = [startnode],[goalnode]

    # Function to attach a new node to an existing node: attach the
    # parent, add to the tree, and show in the figure.

    # Loop - keep growing the tree.
    steps = 0
    while True:
        if (steps >= SMAX) or (len(tree_ini)+len(tree_goal) >= NMAX):
            print("Aborted after %d steps and the tree having %d nodes" %
                    (steps, len(tree_ini)+len(tree_goal) ))
            return None
        # Determine the target state.
        swap = len(tree_ini) > len(tree_goal)
        if (random.uniform(0.0, 1.0) < GOALFRAC):
            targetnode = goalnode if not swap else startnode
        else:
            targetnode = Node(random.uniform(xmin, xmax),
                              random.uniform(ymin, ymax))
        if not swap:
            is_start_1=True
            is_start_2=False
            tree1,tree2=tree_ini,tree_goal
        else:
            is_start_1=False
            is_start_2=True
            tree1, tree2 = tree_goal, tree_ini
        last1, success1, n1 = extend_towards(tree1, targetnode, is_start_1, ratio=2/3, visual=visual)
        last2, success2, n2= extend_towards(tree2, last1, is_start_2,visual)
        steps += n1+n2
        if success2:
            if not swap:
                path_ini=list(reversed(trace(last1)))
                path_goal=trace(last2)
                path=path_ini+path_goal[1:]
            else:
                path_ini=list(reversed(trace(last2)))
                path_goal=trace(last1)
                path=path_ini+path_goal[1:]

            print("Finished after %d steps and total nodes %d" %
          (steps, len(tree_ini) + len(tree_goal)))
            return path



    

def rrt(startnode, goalnode, visual=None):
    # Start the tree with the startnode (set no parent just in case).
    startnode.parent = None
    tree = [startnode]
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
            addtotree(tree, grownode, newnode,True, visual)

            # Also try to connect the goal.  Stop growing.
            if (newnode.distance(goalnode) <= DSTEP and
                newnode.connectsTo(goalnode)):
                addtotree(tree,newnode, goalnode,True, visual)
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
    # path= rrt_connect(startnode, goalnode, visual)
    path, _ = rrt_star(startnode, goalnode, visual)

    # If unable to connect, just note before closing.
    if not path:
        visual.show("UNABLE TO FIND A PATH")
        return

    # Show the path.
    path_cost = pathCost(path)


    visual.drawPath(path, color='r', linewidth=2)
    visual.show("Showing the raw path (length %.1f)" % path_cost)


    # Post process the path.
    finalpath = postProcess(path)

    # Show the post-processed path.
    cost = pathCost(finalpath)
    visual.drawPath(finalpath, color='b', linewidth=2)
    visual.show("Showing the post-processed path (cost/length %.1f)" % cost)


if __name__== "__main__":
    main()
