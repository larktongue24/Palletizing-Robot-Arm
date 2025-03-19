import numpy as np
import random
from lib.detectCollision import detectCollision
from lib.loadmap import loadmap
from copy import deepcopy
from lib.calculateFK import FK

def isRobotCollided(q, map):
    """
    Check if a single configuration causes a collision.
    :param q: Configuration (joint angles)
    :param map: Map struct containing obstacles
    :return: True if configuration causes collision, False otherwise
    """
    fk = FK()
    joint_positions, _ = fk.forward(q)

    linePt1 = joint_positions[:-1]  # Start of each link
    linePt2 = joint_positions[1:]   # End of each link

    for obstacle in map.obstacles:
        if np.any(detectCollision(linePt1, linePt2, obstacle)):
            return True
    return False

def isPathCollided(q1, q2, map, step_size=0.01):
    """
    Check if the path between two configurations is collision-free.
    """
    distance = np.linalg.norm(q2 - q1)
    steps = int(distance / step_size)
    
    for i in range(steps + 1):
        q_intermediate = q1 + (q2 - q1) * (i / steps)
        if isRobotCollided(q_intermediate, map):
            return True
    return False

def nearest_neighbors(q_rand, tree, radius):
    """
    Find all nodes in tree within a given radius of q_rand.
    """
    return [node for node in tree if np.linalg.norm(node[0] - q_rand) < radius]

def nearest_neighbor(q_rand, tree):
    """Find the nearest neighbor to q_rand in the given tree."""
    return min(tree, key=lambda node: np.linalg.norm(node[0] - q_rand))

def construct_path(tree, end):
    """Construct a path from the tree."""
    path = []
    current = end
    while current is not None:
        path.append(current)
        current = next((node[1] for node in tree if np.all(node[0] == current)), None)
    path.reverse()
    return path

def path_cost(tree, node):
    """Calculate the total cost to reach a node."""
    cost = 0
    while node is not None:
        parent = next((p for p in tree if np.all(p[0] == node)), None)
        if parent is None:
            break
        cost += np.linalg.norm(parent[0] - node)
        node = parent[1]
    return cost

def rrt_star(map, start, goal, max_iter=1000, radius=0.5):
    """
    Implement RRT* algorithm.
    """
    path = []
    np.random.seed(2)

    lowerLim = np.array([-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973])
    upperLim = np.array([2.8973,1.7628,2.8973,-0.0698,2.8973,3.7525,2.8973])

    if isRobotCollided(start, map) or isRobotCollided(goal, map):
        return np.array([])

    tree = [(start, None, 0)]  # Each node is (q, parent, cost)

    for _ in range(max_iter):
        q_rand = np.random.uniform(lowerLim, upperLim)
        q_nearest, parent, _ = nearest_neighbor(q_rand, tree)

        if isPathCollided(q_nearest, q_rand, map):
            continue

        # Get neighbors within a given radius
        neighbors = nearest_neighbors(q_rand, tree, radius)

        # Choose the best parent (lowest cost)
        best_parent = q_nearest
        min_cost = path_cost(tree, q_nearest) + np.linalg.norm(q_rand - q_nearest)
        for neighbor, _, _ in neighbors:
            cost = path_cost(tree, neighbor) + np.linalg.norm(q_rand - neighbor)
            if cost < min_cost and not isPathCollided(neighbor, q_rand, map):
                best_parent = neighbor
                min_cost = cost

        tree.append((q_rand, best_parent, min_cost))

        # Rewire
        for neighbor, old_parent, old_cost in neighbors:
            new_cost = min_cost + np.linalg.norm(q_rand - neighbor)
            if new_cost < old_cost and not isPathCollided(q_rand, neighbor, map):
                tree.remove((neighbor, old_parent, old_cost))
                tree.append((neighbor, q_rand, new_cost))

        if np.linalg.norm(q_rand - goal) < radius and not isPathCollided(q_rand, goal, map):
            tree.append((goal, q_rand, min_cost + np.linalg.norm(q_rand - goal)))
            return np.array(construct_path(tree, goal))

    return np.array([])

if __name__ == '__main__':
    map_struct = loadmap("../maps/map1.txt")
    start = np.array([0,-1,0,-2,0,1.57,0])
    goal =  np.array([-1.2, 1.57 , 1.57, -2.07, -1.57, 1.57, 0.7])
    path = rrt_star(deepcopy(map_struct), deepcopy(start), deepcopy(goal))