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
    # Get the position of each joint using FK or some helper functions.
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
    :param q1: Start configuration
    :param q2: End configuration
    :param map: Map struct containing obstacles
    :param step_size: Step size for interpolation
    :return: True if any part of the path causes a collision, False otherwise
    """

    distance = np.linalg.norm(q2 - q1)
    steps = int(distance / step_size)
    
    for i in range(steps + 1):
        q_intermediate = q1 + (q2 - q1) * (i / steps)
        if isRobotCollided(q_intermediate, map):
            return True
    return False


def nearest_neighbor(q_rand, tree):
    """Find the nearest neighbor to q_rand in the given tree."""
    return min(tree, key=lambda node: np.linalg.norm(node[0] - q_rand))[0]

def construct_path(tree, end):
    """Construct a path from the tree."""
    path = []
    current = end
    while current is not None:
        path.append(current)
        current = next((node[1] for node in tree if np.all(node[0] == current)), None)
    path.reverse()
    return path

def rrt(map, start, goal):
    """
    Implement RRT algorithm in this file.
    :param map:         the map struct
    :param start:       start pose of the robot (0x7).
    :param goal:        goal pose of the robot (0x7).
    :return:            returns an mx7 matrix, where each row consists of the configuration of the Panda at a point on
                        the path. The first row is start and the last row is goal. If no path is found, PATH is empty
    """

    # initialize path
    path = []
    np.random.seed(2)

    # get joint limits
    lowerLim = np.array([-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973])
    upperLim = np.array([2.8973,1.7628,2.8973,-0.0698,2.8973,3.7525,2.8973])

    if isRobotCollided(start, map) or isRobotCollided(goal, map):
        return np.array([])
    
    T_start = [(start, None)]
    T_goal = [(goal, None)]

    n_iter = 1000

    for _ in range(n_iter):
        q_rand = np.random.uniform(lowerLim, upperLim)

        # Extend from T_start
        q_nearest_start = nearest_neighbor(q_rand, T_start)
        collision_start = isPathCollided(q_nearest_start, q_rand, map)
        if not collision_start:
            T_start.append((q_rand, q_nearest_start))

        # Extend from T_goal
        q_nearest_goal = nearest_neighbor(q_rand, T_goal)
        collision_goal = isPathCollided(q_nearest_goal, q_rand, map)
        if not collision_goal:
            T_goal.append((q_rand, q_nearest_goal))
        
        # Check if the two trees intersect
        if not collision_start and not collision_goal:
            path_start = construct_path(T_start, q_rand)[:-1]
            path_goal = construct_path(T_goal, q_rand)[::-1]
            return np.vstack((path_start, path_goal))

    return np.array([])

if __name__ == '__main__':
    map_struct = loadmap("../maps/map1.txt")
    start = np.array([0,-1,0,-2,0,1.57,0])
    goal =  np.array([-1.2, 1.57 , 1.57, -2.07, -1.57, 1.57, 0.7])
    path = rrt(deepcopy(map_struct), deepcopy(start), deepcopy(goal))
