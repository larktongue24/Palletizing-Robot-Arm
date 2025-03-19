import numpy as np
from math import pi
from copy import deepcopy

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from lib.calculateFKJac import FK_Jac  # Assuming FK_Jac is in the 'lib' directory
from lib.detectCollision import detectCollision  # Import detectCollision
from lib.loadmap import loadmap  # Import loadmap


class PotentialFieldPlanner:
    # JOINT LIMITS
    lower = np.array([-2.8973, -1.7628, -2.8973, -3.0718,
                      -2.8973, -0.0175, -2.8973])
    upper = np.array([2.8973, 1.7628, 2.8973, -0.0698,
                      2.8973, 3.7525, 2.8973])

    center = lower + (upper - lower) / 2  # Compute middle of range of motion of each joint

    def __init__(self, tol=2e-2, max_steps=1000, min_step_size=4e-5):
        """
        Constructs a potential field planner with solver parameters.

        PARAMETERS:
        tol - the maximum distance between two joint setsn
        max_steps - number of iterations before the algorithm must terminate
        min_step_size - the minimum step size before concluding that the
                        optimizer has converged
        """
        # Solver parameters
        self.tol = tol
        self.max_steps = max_steps
        self.min_step_size = min_step_size

        # Planner parameters
        self.zeta = [30 if i <= 5 else 15 for i in range(9)]  # Attractive potential scaling factors
        self.eta = 0.0001 # Repulsive potential scaling factor
        self.d = 0.12 # Distance threshold for attractive potential
        self.rho_0 = 0.12 # Distance threshold for repulsive potential
        self.alpha = 0.02  # Initial step size (learning rate)
        self.alpha_max = 0.4  # Maximum step size
        self.max_dq = 0.1  # Maximum allowed change in joint angles per iteration
        self.perturbation_magnitude = 0.5  # Magnitude for random perturbations

        self.fk = FK_Jac()

    ######################
    ## Helper Functions ##
    ######################

    @staticmethod
    def attractive_force(target, current, zeta_i, d):
        """
        Computes the attractive force vector between the target joint position and the current joint position.

        INPUTS:
        target - 3x1 numpy array representing the desired joint position in the world frame
        current - 3x1 numpy array representing the current joint position in the world frame
        zeta_i - scalar, the attractive potential scaling factor for this joint
        d - scalar, the distance threshold for attractive potential

        OUTPUTS:
        att_f - 3x1 numpy array representing the force vector that pulls the joint
                from the current position to the target position
        """
        # Euclidean distance between target and current positions
        distance = np.linalg.norm(current - target)

        # Attractive force vector
        if distance **2 > d:
            # Conical potential (linear scaling)
            att_f = - (current - target) / distance  # Normalize the direction
        else:
            # Parabolic potential (quadratic scaling)
            att_f = - zeta_i * (current - target)  # Quadratic scaling without normalization

        return att_f

    @staticmethod
    def repulsive_force(obstacle, current, eta, rho_0):
        """
        Computes the repulsive force vector between the obstacle and the current joint position.

        INPUTS:
        obstacle - 1x6 numpy array representing an obstacle box in the world frame
        current - 3x1 numpy array representing the current joint position in the world frame
        eta - scalar, the repulsive potential scaling factor
        rho_0 - scalar, the distance threshold for repulsive potential

        OUTPUTS:
        rep_f - 3x1 numpy array representing the force vector that pushes the joint
                away from the obstacle
        """
        # Compute distance and unit vector using `dist_point2box`
        distance, unitvec = PotentialFieldPlanner.dist_point2box(current.T, obstacle)
        distance = distance[0]  # Extract scalar distance
        unitvec = unitvec[0].reshape(3, 1)  # Extract 3x1 unit vector

        # If the distance is greater than the threshold or zero, no repulsive force

        if distance == 0:
            distance = 0.01

        if distance > rho_0:
            return np.zeros((3, 1))

        # Compute the repulsive force components
        rho_1 = (1 / distance) - (1 / rho_0)
        rho_2 = (1 / (distance ** 2))
        rep_f = -eta * rho_1 * rho_2 * unitvec

        return rep_f

    @staticmethod
    def dist_point2box(p, box):
        """
        Computes the closest point on the box to a given point.

        INPUTS:
        p - nx3 numpy array of points [x, y, z]
        box - 1x6 numpy array of minimum and maximum points of box

        OUTPUTS:
        dist - nx1 numpy array of distance between the points and the box
               dist > 0 point outside
               dist = 0 point is on or inside box
        unit - nx3 numpy array where each row is the corresponding unit vector
               from the point to the closest spot on the box
               norm(unit) = 1 point is outside the box
               norm(unit) = 0 point is on/inside the box
        """
        # Get box info
        boxMin = np.array([box[0], box[1], box[2]])
        boxMax = np.array([box[3], box[4], box[5]])
        boxCenter = (boxMin + boxMax) * 0.5
        p = np.array(p)

        # Get distance info from point to box boundary
        dx = np.maximum.reduce([boxMin[0] - p[:, 0], p[:, 0] - boxMax[0], np.zeros(p[:, 0].shape)])
        dy = np.maximum.reduce([boxMin[1] - p[:, 1], p[:, 1] - boxMax[1], np.zeros(p[:, 1].shape)])
        dz = np.maximum.reduce([boxMin[2] - p[:, 2], p[:, 2] - boxMax[2], np.zeros(p[:, 2].shape)])

        # Convert to distance
        distances = np.vstack([dx, dy, dz]).T
        dist = np.linalg.norm(distances, axis=1)

        # Figure out the signs
        signs = np.sign(boxCenter - p)

        # Calculate unit vector and replace with
        unit = distances / dist[:, np.newaxis] * signs
        unit[np.isnan(unit)] = 0
        unit[np.isinf(unit)] = 0
        return dist, unit

    def compute_forces(self, target, obstacle_list, current):
        """
        Computes the sum of forces (attractive, repulsive) on each joint.

        INPUTS:
        target - 3x9 numpy array representing the desired joint/end effector positions
                 in the world frame
        obstacle_list - nx6 numpy array representing the obstacle box min and max positions
                        in the world frame
        current - 3x9 numpy array representing the current joint/end effector positions
                  in the world frame

        OUTPUTS:
        joint_forces - 3x9 numpy array representing the force vectors on each
                       joint/end effector
        """
        # Initialize joint forces
        joint_forces = np.zeros((3, 9))

        # Loop through each joint
        for i in range(target.shape[1]):  # Loop over all 9 joints
            total_rep_f = np.zeros((3, 1))

            # Compute attractive force for the current joint
            total_att_force = PotentialFieldPlanner.attractive_force(
                target[:, i].reshape(3, 1),
                current[:, i].reshape(3, 1),
                self.zeta[i],
                self.d
            )

            # Loop over all obstacles
            for obs in obstacle_list:
                # Compute repulsive force
                rep_force = PotentialFieldPlanner.repulsive_force(
                    obs,
                    current[:, i].reshape(3, 1),
                    self.eta,
                    self.rho_0
                )
                total_rep_f += rep_force

            # Combine attractive and repulsive forces and assign to joint_forces
            joint_forces[:, i] = (total_att_force + total_rep_f).flatten()

        return joint_forces

    def compute_torques(self, joint_forces, q):
        """
        Converts joint forces to joint torques. Computes the torques on each joint.

        INPUTS:
        joint_forces - 3x9 numpy array representing the force vectors on each
                       joint/end effector
        q - 1x7 numpy array representing the current joint angles

        OUTPUTS:
        joint_torques - 1x9 numpy array representing the torques on each joint, with
                        zeros for virtual joints
        """
        # Get joint positions and transformation matrices
        joint_positions, T0e = self.fk.forward_expanded(q)  # T0e: 10x4x4 homogeneous transformations

        # Initialize joint torques as (9,)
        joint_torques = np.zeros(9)

        # Number of physical joints
        num_physical_joints = 7  # Joints 0 to 6

        # For each force applied at joint i (i from 0 to 8) but in joint positions, it is 1 to 9
        for i in range(joint_forces.shape[1]):  # joint_forces has shape (3, 9)
            F_i = joint_forces[:, i].reshape(3, 1)  # Force at joint i

            # If the force is zero, skip computation
            if np.linalg.norm(F_i) == 0:
                continue

            # For each joint j that affects joint i
            # Joints 0 to min(i, num_physical_joints - 1)
            max_joint = min(i, num_physical_joints - 1)
            for j in range(max_joint + 1):  # Include max_joint
                z_j = T0e[j][:3, 2].reshape(3, 1)  # Z-axis of joint j
                p_j = joint_positions[j].reshape(3, 1)  # Position of joint j
                p_i = joint_positions[i+1].reshape(3, 1)  # Position where force is applied

                # Compute torque contribution at joint j due to force at joint i
                # tau_j = (Jvi_j)^T * F_i
                Jvi_j = np.cross(z_j.flatten(), (p_i - p_j).flatten()).reshape(3, 1)
                tau_j = Jvi_j.T @ F_i  # Scalar torque

                joint_torques[j] += tau_j.item()

        return joint_torques  # Returns a 1x9 array with zeros for virtual joints

    @staticmethod
    def q_distance(target, current):
        """
        Computes the distance between any two joint configurations.

        INPUTS:
        target - 1x7 numpy array representing some joint angles
        current - 1x7 numpy array representing some joint angles

        OUTPUTS:
        distance - the distance between the target and the current joint sets
        """
        distance = np.linalg.norm(target - current)
        return distance

    def compute_gradient(self, q, joint_forces):
        """
        Computes the joint gradient step based on the joint torques computed from joint forces.

        INPUTS:
        q - 1x7 numpy array representing the current joint angles
        joint_forces - 3x9 numpy array representing the forces on each joint

        OUTPUTS:
        dq - 1x7 numpy array representing the desired change in joint angles
        """
        # Compute joint torques (1x9 array)
        joint_torques = self.compute_torques(joint_forces, q)  # Returns 1x9 array

        # print("Joint torques:", joint_torques)
        # Extract torques for the physical joints (first 7 joints)
        tau_physical = joint_torques[:7]
        
        dq = tau_physical / np.linalg.norm(tau_physical)
        
        # Normalize torques
        # Limit the maximum step size
        # dq_norm = np.linalg.norm(dq)
        # if dq_norm > self.max_dq:
        #     dq = dq * (self.max_dq / dq_norm)

        return dq
    
    def isRobotCollided(self, q, map):
        """
        Check if a single configuration causes a collision.
        :param q: Configuration (joint angles)
        :param map: Map struct containing obstacles
        :return: True if configuration causes collision, False otherwise
        """
        # Get the position of each joint using FK or some helper functions.
        # Check if any joint angle is out of bounds
        if np.any(q < self.lower) or np.any(q > self.upper):
            return True
        joint_positions, _ = self.fk.forward_expanded(q)

        linePt1 = joint_positions[:-1]  # Start of each link
        linePt2 = joint_positions[1:]   # End of each link
        linePt1 = np.vstack((linePt1, joint_positions[7]))  # Connect joint 7
        linePt2 = np.vstack((linePt2, joint_positions[9]))  # Connect joint 9

        for obstacle in map.obstacles:
            if np.any(detectCollision(linePt1, linePt2, obstacle)):
                return True
        return False
    
    def isPathCollided(self, q1, q2, map, step_size=0.01):
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
            if self.isRobotCollided(q_intermediate, map):
                return True
        return False
    
    ###############################
    ### Potential Field Solver  ###
    ###############################

    def plan(self, map_struct, start, goal):
        """
        Uses potential field to move the robot arm from the starting configuration to
        the goal configuration.

        INPUTS:
        map_struct - a map struct containing min and max positions of obstacle boxes
        start - 1x7 numpy array representing the starting joint angles for a configuration
        goal - 1x7 numpy array representing the desired joint angles for a configuration

        OUTPUTS:
        q_path - nx7 numpy array of joint angles [q0, q1, q2, q3, q4, q5, q6]. This should contain
                 all the joint angles throughout the path of the planner.
        """
        # Ensure start and goal are numpy arrays of shape (7,)
        np.random.seed(2)
        start = np.array(start).flatten()
        goal = np.array(goal).flatten()

        # Initial collision check
        if self.isRobotCollided(start, map_struct) or self.isRobotCollided(goal, map_struct):
            print("Error: Start and goal configuration is in collision.")
            return np.array([])  # Return an empty path indicating failure
        else:
            print("Start and goal configuration are collision-free.")
                  
        q_path = [start.copy()]
        q = start.copy()
        max_iterations = self.max_steps
        step_size = self.alpha  # Initialize step size

        # Compute target joint positions (3x9)
        target_positions, _ = self.fk.forward_expanded(goal)
        target_positions = target_positions[1:10, :].T  # Shape: 3x9

        for iteration in range(max_iterations):
            # Compute current joint positions (3x9)
            current_positions, _ = self.fk.forward_expanded(q)
            current_positions = current_positions[1:10, :].T  # Shape: 3x9

            # Compute forces
            joint_forces = self.compute_forces(target_positions, map_struct.obstacles, current_positions)

            # Compute gradient (dq)
            dq = self.compute_gradient(q, joint_forces) * step_size # dq is 1x7 array

            if len(q_path) > 1 and self.q_distance(q + dq, q_path[-2]) < self.min_step_size:
                print("Converged to local minima, performing random perturbation.")
                dq = np.random.uniform(-self.perturbation_magnitude, self.perturbation_magnitude, size=7)

            if self.q_distance(q, goal) < self.tol:
                break

            if self.isPathCollided(q, q + dq, map_struct):
                print("Collision detected, performing random perturbation.")

                while self.isPathCollided(q, q + dq, map_struct):
                    dq = np.random.uniform(-self.perturbation_magnitude, self.perturbation_magnitude, size=7)
                    
            # Update configuration
            new_q = q + dq  # dq already includes step size

            # Append new_q to path
            q_path.append(new_q.copy())
            q = new_q

            # Print error for debugging
            current_error = np.linalg.norm(q - goal)
            print(f"Iteration {iteration}: Error = {current_error}")

        else:
            # After the loop completes without breaking
            print("Reached maximum iterations without convergence.")
            
            
        q_path = np.array(q_path)
        return q_path


################################
## Simple Testing Environment ##
################################

if __name__ == "__main__":

    np.set_printoptions(suppress=True,precision=5)

    planner = PotentialFieldPlanner()
    
    # inputs 
    map_struct = loadmap("../maps/map4.txt")
    start = np.array([0,-1,0,-2,0,1.57,0])
    # start = np.array([ 1.96075, -1.56921, -1.55441, -2.13421,  1.24318,  3.63385, -1.97802])
    goal =  np.array([-1.2, 1.57 , 1.57, -2.07, -1.57, 1.57, 0.7])
    
    # potential field planning
    q_path = planner.plan(deepcopy(map_struct), deepcopy(start), deepcopy(goal))
    
    # show results
    for i in range(q_path.shape[0]):
        error = PotentialFieldPlanner.q_distance(q_path[i, :], goal)
        # print('iteration:',i,' q =', q_path[i, :], ' error={error}'.format(error=error))

    print("q path: ", q_path)
    print(q_path[-1])
