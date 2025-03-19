import sys
import numpy as np
from copy import deepcopy
from math import pi
import matplotlib.pyplot as plt

import rospy
# Common interfaces for interacting with both the simulation and real environments!
from core.interfaces import ArmController
from core.interfaces import ObjectDetector

# for timing that is consistent with simulation or real time as appropriate
from core.utils import time_in_seconds

# import lib functions
from lib.calculateFK import FK
from lib.IK_position_null import IK

def get_gripper_positions_and_vertices(block_pose, gripper_width = 0.2, gripper_size = 0.01):
    # gripper_width = arm._gripper.MAX_WIDTH * (1 - 1e-2)
    half_width = gripper_width / 2

    block_x = block_pose[0, 3]
    block_y = block_pose[1, 3]

    rotation_matrix = block_pose[:2, :2]

    gripper_positions = [
        np.dot(rotation_matrix, [-half_width, 0]) + [block_x, block_y],
        np.dot(rotation_matrix, [half_width, 0]) + [block_x, block_y],
        np.dot(rotation_matrix, [0, -half_width]) + [block_x, block_y],
        np.dot(rotation_matrix, [0, half_width]) + [block_x, block_y]
    ]

    # gripper upper corner
    half_gripper_size = gripper_size / 2
    gripper_corners = np.array([
        [-half_gripper_size, -half_gripper_size],
        [half_gripper_size, -half_gripper_size],
        [half_gripper_size, half_gripper_size],
        [-half_gripper_size, half_gripper_size]
    ])

    gripper_vertices = np.array([
        np.dot(rotation_matrix, gripper_corners.T).T + pos
        for pos in gripper_positions
    ])

    return gripper_positions,  gripper_vertices

def get_block_corners(block_pose):
    # if isinstance(block_pose, tuple):
    #     block_pose = np.array(block_pose)
    print(block_pose)

    block_x = block_pose[0, 3]
    block_y = block_pose[1, 3]

    block_size = 0.05
    half_block_size = block_size / 2

    block_corners = np.array([
        [-half_block_size, -half_block_size],
        [half_block_size, -half_block_size],
        [half_block_size, half_block_size],
        [-half_block_size, half_block_size]
    ])

    rotation_matrix = block_pose[:2, :2]
    rotated_corners = np.dot(rotation_matrix, block_corners.T).T + [block_x, block_y]

    return rotated_corners

def point_in_polygon(point, polygon):
    x, y = point
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

def collision_detection(block_poses):
    n = len(block_poses)
    collision_results = np.zeros((4, n), dtype=int)
    all_block_corners = [get_block_corners(block_pose[1]) for block_pose in block_poses]
    all_gripper_vertices = []
    for block_pose in block_poses:
        _, gripper_vertices = get_gripper_positions_and_vertices(block_pose[1])
        all_gripper_vertices.append(gripper_vertices)

    for i, gripper_vertices in enumerate(all_gripper_vertices):
        for j, vertices in enumerate(gripper_vertices):
            for vertex in vertices:
                for block_corners in all_block_corners:
                    if point_in_polygon(vertex, block_corners):
                        collision_results[j, i] = 1
                        break
    
    show_collision_results(all_block_corners, all_gripper_vertices, collision_results)
    print (collision_results)
    return collision_results

def show_collision_results(all_block_corners, all_gripper_vertices, collision_results):
    plt.figure()

    for block_corners in all_block_corners:
        plt.fill(block_corners[:, 0], block_corners[:, 1], 'k', alpha=0.5, label='Block')

    for i, gripper_vertices in enumerate(all_gripper_vertices):
        for j, vertices in enumerate(gripper_vertices):
            color = 'r' if collision_results[j, i] == 1 else 'b'
            # color = 'r' if j==0 or j==1 else 'b'
            plt.fill(vertices[:, 0], vertices[:, 1], color, alpha=0.5)

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Collision Detection Results')
    plt.axis('equal')
    plt.grid(True)
    plt.show()

def divide_blocks(block_poses):
    collision_results = collision_detection(block_poses)
    collided_blocks = []
    non_collided_blocks = []
    for i, block_pose in enumerate(block_poses):
        name, pose = block_pose
        if collision_results[2][i] == 0 and collision_results[3][i] == 0:
            non_collided_blocks.append(block_pose)
        elif collision_results[0][i] == 0 and collision_results[1][i] == 0:
            rotated_pose = rotate_pose_90_degrees(pose)
            non_collided_blocks.append((name, rotated_pose))
        else:
            collided_blocks.append(block_pose)
    return collided_blocks, non_collided_blocks

def rotate_pose_90_degrees(pose):
    rotation_matrix = np.array([
        [0, -1, 0, 0],
        [1,  0, 0, 0],
        [0,  0, 1, 0],
        [0,  0, 0, 1]
    ])
    return np.dot(pose, rotation_matrix)

