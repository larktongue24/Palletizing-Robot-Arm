import sys
import numpy as np
from copy import deepcopy
from math import pi, sqrt

import rospy
# Common interfaces for interacting with both the simulation and real environments!
from core.interfaces import ArmController
from core.interfaces import ObjectDetector

# for timing that is consistent with simulation or real time as appropriate
from core.utils import time_in_seconds

# import lib functions
from lib.calculateFK import FK
from lib.IK_position_null import IK
import lib.collision_detection

def safeMove(q):
    arm.set_arm_speed(0.28) 
    arm.safe_move_to_position(q)


def adjust_transform_matrix(matrix):
    rotation_matrix = matrix[:3, :3]
    translation_vector = matrix[:3, 3]

    if np.argmax(abs(matrix[2, :3])) == 1:
        print(abs(matrix[2, :3]))
        rotation_matrix[:, 2] = [0, 0, 1]
        rotation_matrix[:, 0] = np.cross(rotation_matrix[:, 0], rotation_matrix[:, 2])
        rotation_matrix[:, 0] /= np.linalg.norm(rotation_matrix[:, 0])
        rotation_matrix[:, 1] = np.cross(rotation_matrix[:, 2], rotation_matrix[:, 0])
        rotation_matrix[:, 1] /= np.linalg.norm(rotation_matrix[:, 1])
    else:
        print(abs(matrix[2, :3]))
        rotation_matrix[:, 2] = [0, 0, 1]
        rotation_matrix[:, 0] = np.cross(rotation_matrix[:, 1], rotation_matrix[:, 2])
        rotation_matrix[:, 0] /= np.linalg.norm(rotation_matrix[:, 0])
        rotation_matrix[:, 1] = np.cross(rotation_matrix[:, 2], rotation_matrix[:, 0])
        rotation_matrix[:, 1] /= np.linalg.norm(rotation_matrix[:, 1])

    adjusted_matrix = np.eye(4)
    adjusted_matrix[:3, :3] = rotation_matrix
    adjusted_matrix[:3, 3] = translation_vector

    return adjusted_matrix

def detect_blocks(q, H_ee_camera, detector):
    fk = FK()
    _, T0e = fk.forward(q)

    adjusted_poses = []
    for (name, pose) in detector.get_detections():
        current_pose = deepcopy(pose)
        current_pose = T0e @ H_ee_camera @ current_pose
        current_pose = adjust_transform_matrix(current_pose)
        adjusted_poses.append((name, current_pose))
        print(name)
        print(np.array2string(current_pose, formatter={'float_kind':lambda x: "%.7f" % x}))

    return adjusted_poses

def move_gripper_above_block(arm, block_pose, current_q, height=0.2):
    # Extract the position of the block
    block_position = block_pose[:3, 3]
    print(block_position)

    # Calculate the target position 20cm above the block
    target_position = block_position + np.array([0, 0, height])

    # Create a target pose matrix
    target_pose = deepcopy(block_pose)
    target_pose[:3, 3] = target_position

    # Adjust the target pose so that the gripper's z-axis points in the negative direction of the block's z-axis
    target_pose[:3, 2] = -block_pose[:3, 2]
    target_pose[:3, 0] = np.cross(target_pose[:3, 1], target_pose[:3, 2])
    target_pose[:3, 0] /= np.linalg.norm(target_pose[:3, 0])
    target_pose[:3, 1] = np.cross(target_pose[:3, 2], target_pose[:3, 0])
    target_pose[:3, 1] /= np.linalg.norm(target_pose[:3, 1])

    # Use IK to find the joint angles for the target pose
    from lib.IK_position_null import IK
    ik = IK()
    seed = arm.neutral_position() # use neutral configuration as seed
    q, rollout, _, _ = ik.inverse(target_pose, seed, method='J_pseudo', alpha=.5)

    if q[6] > 2.8973:
        q[6] -= pi
        print("q6 > 2.8973")
    elif q[6] < -2.8973:
        q[6] += pi
        print("q6 < -2.8973")

    print("move_gripper_above_block")
    print(len(rollout))

    # Move the arm to the target position
    arm.safe_move_to_position(q)
    return q

def move_gripper_to_block(arm, block_pose, current_q):
    # Extract the position of the block
    block_position = block_pose[:3, 3]
    print(block_position)

    # Create a target pose matrix
    target_pose = deepcopy(block_pose)

    # Adjust the target pose so that the gripper's z-axis points in the negative direction of the block's z-axis
    target_pose[:3, 2] = -block_pose[:3, 2]
    target_pose[:3, 0] = np.cross(target_pose[:3, 1], target_pose[:3, 2])
    target_pose[:3, 0] /= np.linalg.norm(target_pose[:3, 0])
    target_pose[:3, 1] = np.cross(target_pose[:3, 2], target_pose[:3, 0])
    target_pose[:3, 1] /= np.linalg.norm(target_pose[:3, 1])

    # Use IK to find the joint angles for the target pose
    from lib.IK_position_null import IK
    ik = IK()
    seed = arm.neutral_position() # use neutral configuration as seed
    q, rollout, _, _ = ik.inverse(target_pose, seed, method='J_pseudo', alpha=.5)

    if q[6] > 2.8973:
        q[6] -= pi
        print("q6 > 2.8973")
    elif q[6] < -2.8973:
        q[6] += pi
        print("q6 < -2.8973")
    
    print("move_gripper_to_block")
    print(len(rollout))

    # Move the arm to the target position
    arm.safe_move_to_position(q)
    return q

def move_gripper_to_platform(arm, current_q, num, height=0.03):
    # Adjust the target pose so that the gripper's z-axis points in the negative direction of the target's z-axis
    target_pose = np.array([[0, -1, 0, 0.562],
                            [1, 0, 0, -0.134],
                            [0, 0, -1, 0.225+height],
                            [0, 0, 0, 1]])
    
    target_pose[2,3] += 0.05 * num
    
    # Use IK to find the joint angles for the target pose
    from lib.IK_position_null import IK
    ik = IK()
    seed = arm.neutral_position()
    q, rollout, _, _ = ik.inverse(target_pose, seed, method='J_pseudo', alpha=.5)
    print(len(rollout))

    # Move the arm to the target position
    arm.safe_move_to_position(q)
    return q

def move_gripper_to_platform_dynamic(arm, current_q, num, height=0.03):
    seed = arm.neutral_position()
    from lib.IK_position_null import IK
    ik = IK()

    target_pose = np.array([[ sqrt(2)/2, 0, sqrt(2)/2,  0.562],
                [0, -1, 0, -0.134],
                [sqrt(2)/2, 0, -sqrt(2)/2, 0.225+height],
                [0.,  0., 0., 1.]])
    target_pose[2,3] += 0.05 * num
    
    q, _, _, _ = ik.inverse(target_pose, seed, method='J_pseudo', alpha=.5)

    # Move the arm to the target position
    arm.safe_move_to_position(q)
    #arm.exec_gripper_cmd(arm._gripper.MAX_WIDTH * (1 - 1e-2),force=50)

    return q

def calculate_weighted_target_z(adjusted_poses):
    weighted_sum = 0.0  
    total_weight = 0.0  

    for _, pose in adjusted_poses:
        x, y, z = pose[:3, 3]

        distance = np.sqrt(x**2 + y**2 + z**2)

        weight = 1 / distance

        weighted_sum += weight * z
        total_weight += weight

    target_z = weighted_sum / total_weight
    return target_z

def static_block(arm, block_poses):
    grabbed_blocks = []
    remaining_blocks = block_poses
    i = 0

    while i < 4 and remaining_blocks:
        collided_blocks, non_collided_blocks = collision_detection.divide_blocks(remaining_blocks)
        print ("Collided blocks: ", collided_blocks)
        print ("Non-collided blocks: ", non_collided_blocks)
        if not non_collided_blocks:
            print("No Solution")
            break
        for block in non_collided_blocks:
            name, pose = block
            
            ########## pick and place the block ##########
            current_q = move_gripper_above_block(arm, pose, start_position, height=0.2)
            above_block_q = current_q

            current_q = move_gripper_to_block(arm, pose, current_q)
            arm.exec_gripper_cmd(0.04, 50)

            safeMove(above_block_q)

            current_q = move_gripper_to_platform(team, arm, current_q, i, height=0.025)
            arm.exec_gripper_cmd(arm._gripper.MAX_WIDTH * (1 - 1e-2),50)

            current_q = move_gripper_to_platform(team, arm, current_q, i, height=0.1)
            ##############################################

            grabbed_blocks.append(block)
            i += 1

        remaining_blocks = collided_blocks

    return grabbed_blocks

if __name__ == "__main__":
    try:
        team = rospy.get_param("team") # 'red' or 'blue'
    except KeyError:
        print('Team must be red or blue - make sure you are running final.launch!')
        exit()

    rospy.init_node("team_script")


    arm = ArmController()
    detector = ObjectDetector()

    start_position = np.array([-0.01779206, -0.76012354,  0.01978261, -2.34205014, 0.02984053, 1.54119353+pi/8, 0.75344866])
    arm.safe_move_to_position(start_position) # on your mark!
    arm.exec_gripper_cmd(arm._gripper.MAX_WIDTH * (1 - 1e-2),50)

    print("\n****************")
    if team == 'blue':
        print("** BLUE TEAM  **")
    else:
        print("**  RED TEAM  **")
    print("****************")
    input("\nWaiting for start... Press ENTER to begin!\n") # get set!
    print("Go!\n") # go!


#####################################STATIC##################################################
    # STUDENT CODE HERE
    fk = FK()
    _, T0e = fk.forward(start_position)
    print(np.array2string(T0e, formatter={'float_kind':lambda x: "%.7f" % x}))

    ik = IK()
    seed = arm.neutral_position()
    Target = np.array([[sqrt(2)/2, 0, sqrt(2)/2, 0.562-0.45*sqrt(2)/2],
                        [0, -1, 0, 0.1],
                        [sqrt(2)/2, 0, -sqrt(2)/2, 0.2+0.45*sqrt(2)/2],
                        [0, 0, 0, 1]])
    
    q, rollout, _, _ = ik.inverse(Target, seed, method='J_pseudo', alpha=.5)
    print(len(rollout))
    arm.safe_move_to_position(q)
    print(q)
    

    # get the transform from camera to panda_end_effector
    start_position = q
    arm.safe_move_to_position(start_position)
    H_ee_camera = detector.get_H_ee_camera()

    # Call the function
    adjusted_poses = detect_blocks(start_position, H_ee_camera, detector)

    # Without collision avoidance
    # for i in range(4):

    #     current_q = move_gripper_above_block(arm, adjusted_poses[i][1], start_position, height=0.2)
    #     above_block_q = current_q

    #     current_q = move_gripper_above_block(arm, adjusted_poses[i][1], current_q, height=0.05)

    #     current_q = move_gripper_to_block(arm, adjusted_poses[i][1], current_q)
    #     arm.exec_gripper_cmd(0.04,50)
        
    #     arm.safe_move_to_position(above_block_q)

    #     current_q = move_gripper_to_platform(arm, current_q, i, height=0.02)
    #     arm.exec_gripper_cmd(arm._gripper.MAX_WIDTH * (1 - 1e-2),50)

    #     current_q = move_gripper_to_platform(arm, current_q, i, height=0.1)

    static_block(arm, adjusted_poses)

################################################DYNAMIC####################################################

    start_position_D = np.array([-0.01779206-pi/2-pi/16, -0.76012354,  0.01978261, -2.34205014, 0.02984053, 1.54119353+pi/4, 0.75344866])
    arm.safe_move_to_position(start_position_D)
    adjusted_poses_D = detect_blocks(start_position_D, H_ee_camera, detector)

    #target_z = calculate_weighted_target_z(adjusted_poses_D)
    target_z = calculate_weighted_target_z(adjusted_poses_D) - 0.01  ################# 0.01 IS TO BE MEASURED 

    seed = np.array([ 0.17542, -1.06209, -1.74489, -2.01193, -0.31041,  2.16587, -0.69733])

    pre_target = np.array([[ sqrt(2)/2, 0, sqrt(2)/2,  0],
                [0, -1, 0, -0.58],
                [sqrt(2)/2, 0, -sqrt(2)/2,  target_z],
                [0.,  0., 0., 1.]])   
    pre_grab_pos, _, _, _ = ik.inverse(pre_target, seed, method='J_pseudo', alpha=.2)
    arm.safe_move_to_position(pre_grab_pos)

    arm.exec_gripper_cmd(arm._gripper.MAX_WIDTH * (1 - 1e-2),force=50)

    mid_target = np.array([[ sqrt(2)/2, 0, sqrt(2)/2 , 0],
                [0, -1, 0, -0.68],
                [sqrt(2)/2, 0, -sqrt(2)/2,  target_z],
                [0.,  0., 0., 1.]])
    mid_grab_pos, _, _, _ = ik.inverse(mid_target, seed, method='J_pseudo', alpha=.2)

    target = np.array([[ sqrt(2)/2, 0, sqrt(2)/2,  0],
                    [0, -1, 0, -0.75],
                    [sqrt(2)/2, 0, -sqrt(2)/2,  target_z],
                    [0.,  0., 0., 1.]])
    grab_pos, _, _, _ = ik.inverse(target, seed, method='J_pseudo', alpha=.2)

    def is_block_grabbed(arm):
        gripper_state = arm.get_gripper_state()
        positions = gripper_state['position']
        width = abs(positions[1] + positions[0]) 
        # forces = gripper_state['force']
        # force = abs(forces[1] + forces[0]) 
        # print(forces)
        return 0.05 <= width <= 0.07 # Simulation
        #return 0.03 <= width <= 0.055 
    
    for i in range(4, 8):
        arm.safe_move_to_position(mid_grab_pos)
        arm.safe_move_to_position(grab_pos)
        while not rospy.is_shutdown():
            rospy.sleep(8)
            arm.exec_gripper_cmd(0.01,force=50)
            if is_block_grabbed(arm):
                current_q = np.array([])
                current_q = move_gripper_to_platform_dynamic(arm, current_q, i, 0.15)
                current_q = move_gripper_to_platform_dynamic(arm, current_q, i)
                arm.exec_gripper_cmd(arm._gripper.MAX_WIDTH * (1 - 1e-2),force=50)
                current_q = move_gripper_to_platform_dynamic(arm, current_q, i, 0.15)
                break
            arm.exec_gripper_cmd(arm._gripper.MAX_WIDTH * (1 - 1e-2),force=50)
        arm.safe_move_to_position(pre_grab_pos)