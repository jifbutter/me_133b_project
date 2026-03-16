'''rrt_maze.py

   RRT in a continuous maze (thin line-segment walls from gym-continuous-maze style).
   Uses continuous_maze.walls and obstacles; connectsTo like rrttriangles (LineString + one check).
'''

import matplotlib.pyplot as plt
import numpy as np
import random
import time

from math import sqrt
from shapely.geometry import MultiLineString
from shapely.geometry import LineString
from shapely.geometry   import Point, LineString, Polygon, MultiPolygon
from shapely.prepared import prep
# from continuous_maze import walls, obstacles
from binary_maze import walls, walls1, walls2, obstacles, obstacles1, obstacles2




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
DSTEP    = 0.4
GOALFRAC = 0.10
CLEARANCE = 0.4
MOVE=0.9
# Maximum number of steps (attempts) or nodes (successful steps).

# R=0.95
RADIUS=1.0
ALPHA=1.0
BETA=0


MAXSPEED=3
MARGIN=0.01

# Group1_time: 
# TMAX = 20
# Group2_time:
TMAX=20
SWITCH_TIME = 2  # switch from maze 1 to maze 2
FRAME_DT = 0.05     # seconds between redraws
DT=0.05
NUMNODE = 5000

# Hard caps
NMAX = 5000
SMAX= 15000
TWEIGHT=10

######################################################################
#
#   World Definitions
#
#   List of obstacles/objects as well as the start/goal.
#
(xmin, xmax) = (-12, 12)
(ymin, ymax) = (-12, 12)
(tmin,tmax)=(0,TMAX)
GRID = 1.0
NX = int(np.ceil((xmax - xmin) / GRID))
NY = int(np.ceil((ymax - ymin) / GRID))
NT=int(np.ceil((tmax - tmin) / GRID))

counts_start = np.zeros((NX, NY, NT), dtype=np.int32)  # tree growing from start
counts_goal  = np.zeros((NX, NY, NT), dtype=np.int32)  # tree growing from goal

def cell(x, y):
    i = int((x - xmin) / GRID)
    j = int((y - ymin) / GRID)
    # clamp to valid range
    i = max(0, min(NX - 1, i))
    j = max(0, min(NY - 1, j))
    return i, j
def cell_t(x,y,t):
    i = int((x - xmin) / GRID)
    j = int((y - ymin) / GRID)
    k =int((t-tmin)/GRID)
    # clamp to valid range
    i = max(0, min(NX - 1, i))
    j = max(0, min(NY - 1, j))
    k = max(0, min(NT-1,k))

    return i, j,k
# Maze is continuous_maze.walls (thin line segments). No Shapely obstacles.

# Start at center, goal in upper-right area (inside the gym maze bounds).
# First group:
# (xstart, ystart, tstart) = (-9, 11, 0)
# (xgoal, ygoal,tgoal)  = (3, 5,TMAX)
# Second group:
# (xstart, ystart,tstart) = (-9, 11,0)
# (xgoal, ygoal,tgoal)  = (-1, 3, TMAX)
# Third group:
# (xstart, ystart,tstart) = (-9, 11,0)
# (xgoal, ygoal,tgoal)  = (11, 3,TMAX)

# Fourth group:
(xstart, ystart,tstart) = (-9, 11,0)
(xgoal, ygoal,tgoal)  = (6,-3,TMAX)

# (xgoal, ygoal,tgoal)  = (6, -3,TMAX)

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
#
#   Visualization Class
#
#   This renders the world.  In particular it provides the methods:
#     show(text = '')                   Show the current figure
#     drawNode(node,         **kwargs)  Draw a single node
#     drawEdge(node1, node2, **kwargs)  Draw an edge between nodes
#     drawPath(path,         **kwargs)  Draw a path (list of nodes)
#
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
#         from continuous_maze import walls as maze_walls
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
class Visualization:
    def __init__(self):
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self._setup_axes()
        self.draw_world_time(0.0)
        self.show()

    def _setup_axes(self):
        self.ax.clear()
        self.ax.grid(True)
        self.ax.axis('on')
        self.ax.set_xlim(xmin, xmax)
        self.ax.set_ylim(ymin, ymax)
        self.ax.set_aspect('equal')

    def draw_walls(self, wall_array, title=''):
        self._setup_axes()
        for wall in wall_array:
            self.ax.plot(
                [wall[0][0], wall[1][0]],
                [wall[0][1], wall[1][1]],
                'k-',
                linewidth=2
            )
        if title:
            self.ax.set_title(title)

    def draw_world_time(self, t):
        self.draw_walls(get_walls(t), title=f"t = {t:.1f} sec ({current_phase_name(t)})")

    def draw_world_full(self):
        self.draw_walls(walls, title="Planning view")

    def show(self, text=''):
        self.fig.canvas.draw_idle()
        plt.pause(0.001)
        if len(text) > 0:
            input(text + ' (hit return to continue)')

    def drawNode(self, node, **kwargs):
        self.ax.plot(node.x, node.y, **kwargs)

    def drawPoint(self, x, y, **kwargs):
        self.ax.plot(x, y, **kwargs)

    # def drawEdge(self, head, tail, **kwargs):
    #     self.ax.plot([head.x, tail.x], [head.y, tail.y], **kwargs)
    def drawEdge(self, head, tail, **kwargs):
        line, = self.ax.plot([head.x, tail.x], [head.y, tail.y], **kwargs)
        return line

    def drawPath(self, path, **kwargs):
        for i in range(len(path) - 1):
            self.drawEdge(path[i], path[i+1], **kwargs)

    def updateEdge(self, line, head, tail):
        line.set_data([head.x, tail.x], [head.y, tail.y])



######################################################################
#
#   Node Definition
#
class Node:
    #################
    # Initialization:
    def __init__(self, x, y, t, d=0):
        # Define/remember the state/coordinates (x,y).
        self.x = x
        self.y = y
        self.d=d
        self.t=t
        self.cost=d
        # Define a parent (cleared for now).
        self.parent = None
        self.edge_handle = None
        self.children=set()

    # def intermediate(self, other, alpha):
    #     return Node(self.x + alpha * (other.x - self.x),
    #                 self.y + alpha * (other.y - self.y))
    def intermediate_t(self, other, alpha):
        return Node(
            self.x + alpha * (other.x - self.x),
            self.y + alpha * (other.y - self.y),
            self.t + alpha * (other.t - self.t),
        )
    ################
    # Planner functions:
    # Compute the relative distance to another node.
    def distance_t(self, other):
        return sqrt((other.x - self.x)**2 + (other.y - self.y)**2+ (other.t-self.t)**2)
    
    def distance(self,other):
        return sqrt((other.x - self.x)**2 + (other.y - self.y)**2)
    # def inFreespace(self):
    #     if self.x <= xmin or self.x >= xmax or self.y <= ymin or self.y >= ymax:
    #         return False
    #     return True
    def speed(self,other):
        dd=self.distance(other)
        tt=other.t-self.t
        return dd/tt

    def inFreespace(self, clearance=CLEARANCE):
        tt=self.t
        curr_obstacles=get_obstacles(tt)
        if (self.x <= xmin or self.x >= xmax or self.y <= ymin or self.y >= ymax):
            return False
        p = Point(self.x, self.y)
        if not curr_obstacles.disjoint(p):
            return False
        if curr_obstacles.context.distance(p)<clearance:
            return False
        return True
        
    

    def connectsTo(self, other, clearance=CLEARANCE, dt=DT):
        t_diff = other.t - self.t
        if t_diff <= 0:
            return False

        dist_xy = self.distance(other)
        speed = dist_xy / t_diff
        if not (speed <= MAXSPEED+MARGIN):
            return False

        if not self.inFreespace(clearance):
            return False
        if not other.inFreespace(clearance):
            return False

        n_steps = int(np.ceil(t_diff / dt))
        for k in range(1, n_steps):
            tt = self.t + k * dt
            alpha = (tt - self.t) / t_diff
            node = self.intermediate_t(other, alpha)
            if not node.inFreespace(clearance):
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
    

######################################################################
#
#   RRT Functions
#



def addtotree(tree,oldnode,newnode, is_start=True, visual=None):
    newnode.parent = oldnode
    oldnode.children.add(newnode)
    ## Encourage as less movement as possible: use spatial distance instead of space_time
    newnode.d=oldnode.distance(newnode)
    newnode.cost=oldnode.cost+newnode.d
    tree.append(newnode)
    i,j,k=cell_t(newnode.x,newnode.y, newnode.t)
    if is_start:
        counts_start[i, j,k] += 1
    else:
        counts_goal[i, j,k] += 1
    if visual:
        # if is_start:
        #     visual.drawEdge(oldnode, newnode, color='g', linewidth=1)
        # else:
        #     visual.drawEdge(oldnode, newnode, color='r', linewidth=1)
        # visual.show()
        color = 'g' if is_start else 'r'
        newnode.edge_handle = visual.drawEdge(oldnode, newnode, color=color, linewidth=1)
        visual.show()

def sample_fn_t(is_start_tree=True,alpha=ALPHA, beta=BETA,max_tries=10):
    own   = counts_start if is_start_tree else counts_goal
    other = counts_goal  if is_start_tree else counts_start
    w = (1.0 / np.power(own + 1.0, alpha)) * np.power(other + 1.0, beta)
    p = (w / w.sum()).ravel()
    for _ in range(max_tries):
        idx = np.random.choice(NX * NY*NT, p=p)
        i, j, k = np.unravel_index(idx, (NX, NY, NT))

        x0 = xmin + i * GRID
        x1 = min(x0 + GRID, xmax)
        y0 = ymin + j * GRID
        y1 = min(y0 + GRID, ymax)
        t0 = tmin + k * GRID
        t1 = min(t0 + GRID, tmax)

        x = random.uniform(x0, x1)
        y = random.uniform(y0, y1)
        t= random.uniform(t0, t1)
        n = Node(x, y,t)
        if n.inFreespace():
            return n
    return None

def sample_fn(is_start_tree=True, alpha=ALPHA, beta=BETA, max_tries=10):
    own3   = counts_start if is_start_tree else counts_goal
    other3 = counts_goal  if is_start_tree else counts_start

    own2   = own3.sum(axis=2)
    other2 = other3.sum(axis=2)

    w = (1.0 / np.power(own2 + 1.0, alpha)) * np.power(other2 + 1.0, beta)
    p = (w / w.sum()).ravel()

    for _ in range(max_tries):
        idx = np.random.choice(NX * NY, p=p)
        i, j = divmod(idx, NY)

        x0 = xmin + i * GRID
        x1 = min(x0 + GRID, xmax)
        y0 = ymin + j * GRID
        y1 = min(y0 + GRID, ymax)

        x = random.uniform(x0, x1)
        y = random.uniform(y0, y1)
        return x, y

    return None

def extend_towards(tree, target, is_start=True, visual=None):
    """
    Extend the tree toward target in (x,y,t).

    If is_start=True:
        grow forward in time toward later target nodes.

    If is_start=False:
        grow backward in time toward earlier target nodes,
        but feasibility is still checked in forward time, i.e.
        newnode.connectsTo(last).

    Returns:
        last:    last node added (or nearest valid existing node if no progress)
        success: True iff we reached the target
        n_added: number of new nodes appended to the tree in this call
    """

    # Pick only nodes that can move toward the target in the correct time direction.
    if is_start:
        valid_indices = [i for i, node in enumerate(tree) if (node.t < target.t and node.speed(target)<=MAXSPEED)]
    else:
        valid_indices = [i for i, node in enumerate(tree) if (node.t > target.t and target.speed(node)<=MAXSPEED)]

    if not valid_indices:
        return tree[0], False, 0
    
    index = min(valid_indices, key=lambda i: tree[i].distance_t(target))
    last = tree[index]
    n_added = 0

    while True:
        d = last.distance_t(target)

        # Already there
        if d == 0.0:
            return last, True, n_added

        # Propose next node by spacetime interpolation
        if d <= DSTEP:
            newnode = Node(target.x, target.y, target.t)
            reached_target = True
        else:
            alpha = DSTEP / d
            newnode = last.intermediate_t(target, alpha)
            reached_target = False

        # Enforce monotone time in the correct tree-growth direction
        if is_start:
            if newnode.t <= last.t:
                return last, False, n_added
        else:
            if newnode.t >= last.t:
                return last, False, n_added

        # Check new node itself
        if not newnode.inFreespace():
            return last, False, n_added

        # Check dynamic feasibility in forward time
        if is_start:
            ok = last.connectsTo(newnode)
        else:
            ok = newnode.connectsTo(last)

        if not ok:
            return last, False, n_added

        # Safe: add it
        addtotree(tree, last, newnode, is_start, visual)
        last = newnode
        n_added += 1

        # Reached target
        if reached_target:
            return last, True, n_added

        
def extend_towards_node(tree, last, target, is_start=True,visual=None):
    ## This function is the part of extend_towards without choosing the grownode. 
    ## the grownode is chosen already and put int as last
    n_added=0
    while True:
        d=last.distance_t(target)
        if d == 0.0:
            return last, True, n_added

        if d <= DSTEP:
            newnode = Node(target.x, target.y, target.t)
            reached_target = True
        else:
            alpha = DSTEP / d
            newnode = last.intermediate_t(target, alpha)
            reached_target = False
        if is_start:
            if newnode.t <= last.t:
                return last, False, n_added
        else:
            if newnode.t >= last.t:
                return last, False, n_added

        if not newnode.inFreespace():
            return last, False, n_added

        if is_start:
            ok = last.connectsTo(newnode)
        else:
            ok = newnode.connectsTo(last)

        if not ok:
            return last, False, n_added
        
        addtotree(tree, last, newnode, is_start, visual)
        last = newnode
        n_added += 1

        if reached_target:
            return last, True, n_added

def growth_node(tree,x,y,is_start=True,movefrac=MOVE):
    move_branch = (random.uniform(0.0, 1.0) < movefrac)
    candidates = []
    for i, node in enumerate(tree):
        dist = sqrt((x - node.x)**2 + (y - node.y)**2)

        if move_branch:
            if is_start:
                t_target = node.t + dist / MAXSPEED
            else:
                t_target = node.t - dist / MAXSPEED
        else:
            if is_start:
                t_target = node.t + DSTEP
            else:
                t_target = node.t - DSTEP

        if not (tmin <= t_target <= tmax):
            continue

        key = dist
        candidates.append((key, i, t_target))
    if not candidates:
        return None, None

    _, index, t_target = min(candidates, key=lambda z: z[0])
    grownode = tree[index]

    if move_branch:
        targetnode = Node(x, y, t_target)
    else:
        targetnode = Node(grownode.x, grownode.y, t_target)

    return grownode, targetnode


def trace(node):
    path=[]
    while node is not None:
        path.append(node)
        node=node.parent
    return path


def local_rewire_last(newnode, tree, radius, is_start=True, visual=None):
    # Neighborhood in spacetime
    neighbors = [n for n in tree if n.distance_t(newnode) <= radius]

    # Best-parent step for newnode
    best_parent = newnode.parent
    best_cost = best_parent.cost + best_parent.distance(newnode)

    for n in neighbors:
        if n is newnode:
            continue

        if is_start:
            ok = n.connectsTo(newnode)
        else:
            ok = newnode.connectsTo(n)

        if not ok:
            continue

        cand = n.cost + n.distance(newnode)
        if cand < best_cost:
            best_parent = n
            best_cost = cand

    if best_parent is not newnode.parent:
        newnode.rewire(best_parent, best_parent.distance(newnode))
        if visual and newnode.edge_handle is not None:
            visual.updateEdge(newnode.edge_handle, best_parent, newnode)

    # Optional: rewire neighbors through newnode
    for n in neighbors:
        if n is newnode or n is newnode.parent:
            continue

        cand = newnode.cost + newnode.distance(n)

        if is_start:
            ok = newnode.connectsTo(n)
        else:
            ok = n.connectsTo(newnode)

        if cand < n.cost and ok:
            n.rewire(newnode, newnode.distance(n))
            if visual and n.edge_handle is not None:
                visual.updateEdge(n.edge_handle, newnode, n)

def rrt_star(startnode,goalnode,visual=None,radius=RADIUS):
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

                prev_goal_cost = goal_n.cost
                print(f"[step {steps}] Found first path to goal, cost = {prev_goal_cost:.4f}")

            else:
                cand = newnode.cost + newnode.distance(goal_n)
                if cand < goal_n.cost and newnode.connectsTo(goal_n):
                    goal_n.rewire(newnode, newnode.distance(goal_n))
                    # after rewire, goal_n.cost is updated
                    if (prev_goal_cost is None) or (goal_n.cost < prev_goal_cost - 1e-9):
                        prev_goal_cost = goal_n.cost
                        print(f"[step {steps}] Improved goal cost -> {prev_goal_cost:.4f}")
        
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
    return path

def rrt_connect(startnode, goalnode, visual=None):
    # Start the tree with the startnode (set no parent just in case).
    startnode.parent = None
    goalnode.parent=None
    tree_ini, tree_goal = [startnode],[goalnode]

    # Function to attach a new node to an existing node: attach the
    # parent, add to the tree, and show in the figure.

    # Loop - keep growing the tree.
    steps = 0
    next_report = 1000
    while True:
        total_nodes = len(tree_ini) + len(tree_goal)
        while total_nodes >= next_report:
            print(f'have {total_nodes} nodes')
            next_report += 1000

        if (steps >= SMAX) or (len(tree_ini)+len(tree_goal) >= NMAX):
            print("Aborted after %d steps and the tree having %d nodes" %
                    (steps, len(tree_ini)+len(tree_goal) ))
            return None
        # Determine the target state.
        swap = len(tree_ini) > len(tree_goal)
        if not swap:
            is_start_1=True
            is_start_2=False
            tree1,tree2=tree_ini,tree_goal
        else:
            is_start_1=False
            is_start_2=True
            tree1, tree2 = tree_goal, tree_ini
        
        if random.uniform(0.0, 1.0) < GOALFRAC:
            # exact temporal target
            targetnode = goalnode if not swap else startnode
            last1, success1, n1 = extend_towards(tree1, targetnode, is_start_1, visual=visual)
        else:
            xy=sample_fn(is_start_tree=is_start_1)
            if xy is None:
                continue
            x,y=xy

            grownode,targetnode=growth_node(tree1,x,y,is_start=is_start_1)
            if targetnode is None:
                continue
            last1, success1, n1 = extend_towards_node(tree1, grownode,targetnode, is_start_1, visual=visual)
        #if targetnode is goalnode then can already stop
        last2, success2, n2= extend_towards(tree2, last1, is_start_2,visual=visual)
        steps += n1+n2
        if success2: # one path is found
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


    
def rrt_connect_star(startnode, goalnode, visual=None,
                     radius=RADIUS):
    # init roots
    startnode.parent=None; startnode.cost=0.0; startnode.d=0.0; startnode.children=set()
    goalnode.parent=None;  goalnode.cost =0.0; goalnode.d =0.0; goalnode.children=set()
    tree_ini, tree_goal = [startnode], [goalnode]

    best_path = None
    best_cost = float("inf")
    first_solution_step = None

    steps = 0
    next_report = 1000
    while True:
        total_nodes = len(tree_ini) + len(tree_goal)
        while total_nodes >= next_report:
            print(f'have {total_nodes} nodes')
            next_report += 1000
        # if no path is found and total_nodes>=50000
        if (steps >= SMAX) or (total_nodes>= NMAX):
            break
        # if at least one path is found: stops when the total_nodes>=5000
        if best_path is not None:
        #and total_nodes >= NUMNODE:
            print(f"Stopped after refinement budget: nodes={total_nodes}")
            break

        swap = len(tree_ini) > len(tree_goal)
        if not swap:
            tree1, tree2 = tree_ini, tree_goal
            is_start_1, is_start_2 = True, False
        else:
            tree1, tree2 = tree_goal, tree_ini
            is_start_1, is_start_2 = False, True

        if random.uniform(0.0, 1.0) < GOALFRAC:
            targetnode = goalnode if not swap else startnode
            last1, success1, n1 = extend_towards(tree1, targetnode, is_start_1, visual=visual)
        else:
            xy=sample_fn(is_start_tree=is_start_1)
            if xy is None:
                continue
            x,y=xy
            grownode,targetnode=growth_node(tree1,x,y,is_start=is_start_1)
            if targetnode is None:
                continue
            last1, success1, n1 = extend_towards_node(tree1, grownode,targetnode, is_start_1, visual=visual)

        
        steps += n1
        if n1 == 0:
            continue

        # (optional) light rewiring only around the newest node in tree1
        local_rewire_last(last1, tree1, radius, is_start=is_start_1, visual=visual)

        # try to connect tree2 to last1
        last2, success2, n2 = extend_towards(tree2, last1, is_start_2, visual=visual)
        steps += n2

        if success2: # this is when it finds a path
            # Build the path from the tree
            if not swap:
                path_ini = list(reversed(trace(last1)))
                path_goal = trace(last2)
                path = path_ini + path_goal[1:]
            else:
                path_ini = list(reversed(trace(last2)))
                path_goal = trace(last1)
                path = path_ini + path_goal[1:]
            # Compare the cost of this new founded path with the existing
            cost = pathCost(path)
            if cost < best_cost:
                best_cost = cost
                best_path = path
                if first_solution_step is None:
                    first_solution_step = steps
                    print(f"[step {steps}] first solution, cost = {best_cost:.3f}")
                else:
                    print(f"[step {steps}] improved cost -> {best_cost:.3f}")

    return best_path

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

def point_on_path_at_time(path, t):
    # path is assumed to have increasing time
    if t <= path[0].t:
        return path[0].x, path[0].y

    if t >= path[-1].t:
        return path[-1].x, path[-1].y

    for i in range(len(path) - 1):
        a = path[i]
        b = path[i + 1]
        if a.t <= t <= b.t:
            alpha = (t - a.t) / (b.t - a.t)
            x = a.x + alpha * (b.x - a.x)
            y = a.y + alpha * (b.y - a.y)
            return x, y

    return path[-1].x, path[-1].y
######################################################################
#
#  Main Code
#
# def main():
#     # Report the parameters.
#     print('Running with step size ', DSTEP, ' and up to ', NMAX, ' nodes.')

#     # Create the figure.  Some computers seem to need an additional show()?
#     visual = Visualization()
#     visual.show()

#     # Create the start/goal nodes.
#     startnode = Node(xstart, ystart)
#     goalnode  = Node(xgoal,  ygoal)

#     # Show the start/goal nodes.
#     visual.drawNode(startnode, color='orange', marker='o')
#     visual.drawNode(goalnode,  color='purple', marker='o')
#     visual.show("Showing basic world")


#     # Run the RRT planner.
#     print("Running RRT...")
#     # path = rrt_connect(startnode, goalnode, visual)
#     path=rrt_connect_star(startnode, goalnode, visual)

#     # If unable to connect, just note before closing.
#     if not path:
#         visual.show("UNABLE TO FIND A PATH")
#         return

#     # Show the path.
#     cost = pathCost(path)
#     visual.drawPath(path, color='r', linewidth=2)
#     visual.show("Showing the raw path (cost/length %.1f)" % cost)


#     # Post process the path.
#     finalpath = postProcess(path)

#     # Show the post-processed path.
#     cost = pathCost(finalpath)
#     visual.drawPath(finalpath, color='b', linewidth=2)
#     visual.show("Showing the post-processed path (cost/length %.1f)" % cost)
def animate_path(visual, path):
    t = 0.0
    while t <= TMAX:
        visual.draw_world_time(t)
        visual.drawPath(path, color='b', linewidth=2)

        x, y = point_on_path_at_time(path, t)
        visual.drawPoint(x, y, color='red', marker='o', markersize=10)

        visual.drawPoint(xstart, ystart, color='orange', marker='o', markersize=8)
        visual.drawPoint(xgoal,  ygoal,  color='purple', marker='o', markersize=8)

        visual.show()
        time.sleep(FRAME_DT)
        t += FRAME_DT
def main():
    print('Running with step size ', DSTEP, ' and up to ', NMAX, ' nodes.')

    visual = Visualization()

    # --------------------------------------------------
    # Part 1: show how the maze changes over time
    # --------------------------------------------------
    # t = 0.0
    # while t <= TMAX/2:
    #     visual.draw_world_time(t)
    #     visual.drawPoint(xstart, ystart, color='orange', marker='o', markersize=8)
    #     visual.drawPoint(xgoal,  ygoal,  color='purple', marker='o', markersize=8)
    #     visual.show()
    #     time.sleep(FRAME_DT)
    #     t += FRAME_DT

    # visual.show("Finished showing time-varying maze")

    # --------------------------------------------------
    # Part 2: planning view on the full maze
    # --------------------------------------------------
    startnode = Node(xstart, ystart, 0.0)
    goalnode  = Node(xgoal,  ygoal,  TMAX)

    visual.draw_world_full()
    visual.drawNode(startnode, color='orange', marker='o', markersize=8)
    visual.drawNode(goalnode,  color='purple', marker='o', markersize=8)
    visual.show("Showing planning view")

    print("Running RRT...")
    path = rrt_connect_star(startnode, goalnode, visual)
    # path = rrt_connect(startnode, goalnode, visual)
    

    if not path:
        visual.show("UNABLE TO FIND A PATH")
        return
    finalpath=postProcess(path)
    finalcost = pathCost(finalpath)
    print(f"Final path cost: {finalcost:.3f}")
    

    # --------------------------------------------------
    # Part 3: show the final path projected to x-y
    # --------------------------------------------------
    visual.draw_world_full()
    visual.drawNode(startnode, color='orange', marker='o', markersize=8)
    visual.drawNode(goalnode,  color='purple', marker='o', markersize=8)
    visual.drawPath(path, color='r', linewidth=2)
    visual.show("Showing projected final path")

    # --------------------------------------------------
    # Part 4: animate a dot moving along the path
    # --------------------------------------------------
    animate_path(visual, finalpath)

    while True:
        ans = input("Replay animation? (y/n): ").strip().lower()
        if ans == 'y':
            animate_path(visual, finalpath)
        elif ans == 'n':
            break

    visual.show("Finished path animation")


if __name__== "__main__":
    main()
