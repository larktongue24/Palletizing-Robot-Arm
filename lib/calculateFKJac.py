import numpy as np
from math import cos, sin, pi
from lib.calculateFK import FK


class FK_Jac():

    def __init__(self):

        self.joint_offsets = [
            [0, 0, 0.141, 1],  # Joint 1 at the origin of frame 0
            [0, 0, 0, 1],  # Joint 2 at the origin of frame 1, no offset
            [0, 0, 0.195, 1],  # Joint 3 at the origin of frame 4
            [0, 0, 0, 1],  # Joint 4 at origin of frame 5
            [0, 0, 0.125, 1],  # Joint 5 at origin of frame 6
            [0, 0, -0.015, 1],  # Joint 6 at origin of frame 5
            [0, 0, 0.051, 1],  # Joint 7 at origin of frame 6
            [0, 0, 0, 1],  # End effector frame 7
            [0, 0.1, -0.105, 1],  # Virtual Joint 1 of frame 7
            [0, -0.1, -0.105, 1],  # Virtual Joint 2 of frame 7
        ]

    def forward_expanded(self, q):
        """
        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]

        OUTPUTS:
        jointPositions -10 x 3 matrix, where each row corresponds to a physical or virtual joint of the robot or end effector
                  Each row contains the [x,y,z] coordinates in the world frame of the respective joint's center in meters.
        T0e       - a 10 x 4 x 4 homogeneous transformation matrix,
                  representing each joint/end effector frame expressed in the
                  world frame
        """

        # Initialize an 10x3 matrix to store joint positions (including base and virtual joints)
        jointPositions = np.zeros((10, 3))  # 10 joints including virtual joints

        # Initialize transformation matrix
        T0e = np.zeros((10, 4, 4))  # 10 transformations

        # Set the base position in the world frame
        jointPositions[0] = np.array([0, 0, 0.141])  # Base position
        T0e[0] = np.eye(4)

        # Get all individual transformation matrices
        Ai_matrices = self.compute_Ai(q)

        # Initialize cumulative transformation
        T = np.eye(4)

        # Iterate through the transformation matrices to accumulate transformations
        for i, A_i in enumerate(Ai_matrices):
            T = T @ A_i  # Accumulate transformations
            T0e[i + 1] = T  # Store transformation for this joint

            # Apply joint offset after transformation
            joint_offset = np.array(self.joint_offsets[i + 1])  # Adjust index
            jointPositions[i + 1] = (T @ joint_offset)[:3]

        # Handle virtual joints
        for i in range(8, 10):  # Virtual joints 8 and 9
            virtual_offset = np.array(self.joint_offsets[i])  # Get virtual joint offset

            # Create a new transformation matrix for the virtual joint
            virtual_T = np.eye(4)  # Identity matrix for base
            virtual_T[:3, 3] = virtual_offset[:3]  # Set translation part
            T0e[i] = T0e[7] @ virtual_T  # Transform relative to the end effector

            # Compute the virtual joint position
            jointPositions[i] = T0e[i][:3, 3]  # Extract the position from the transformation

        return jointPositions, T0e

    def get_dh_params(self, q):
        """
        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]

        OUTPUTS:
        Returns the DH parameters as a list of tuples,
        where each tuple is (a, alpha, d, theta).
        """
        
        # Define the DH parameters for each joint
        return [
            (0, -pi / 2, 0.333, q[0]),
            (0, pi / 2, 0, q[1]),
            (0.0825, pi / 2, 0.316, q[2]),
            (0.0825, pi / 2, 0, q[3] + pi),
            (0, -pi / 2, 0.384, q[4]),
            (0.088, pi / 2, 0, q[5] - pi),
            (0, 0, 0.21, q[6] - (pi / 4))
        ]

    def dh_transform(self, a, alpha, d, theta):
        """
        Compute individual transformation matrix using DH parameters.
        """
        ca, sa = cos(alpha), sin(alpha)
        ct, st = cos(theta), sin(theta)

        return np.array([
            [ct, -st * ca, st * sa, a * ct],
            [st, ct * ca, -ct * sa, a * st],
            [0, sa, ca, d],
            [0, 0, 0, 1]
        ])
    
    def compute_Ai(self, q):
        """
        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]

        OUTPUTS:
        Ai: - 4x4 list of np array of homogenous transformations describing the FK of the robot. Transformations are not
              necessarily located at the joint locations
        """
        Ai = []
        dh_params = self.get_dh_params(q)

        for a, alpha, d, theta in dh_params:
            Ai.append(self.dh_transform(a, alpha, d, theta))

        return Ai

    def calcJacobian(self, q_in):
        """
        Calculate the full Jacobian of the end effector in a given configuration.

        :param q_in: 1 x 7 configuration vector (of joint angles) [q1, q2, q3, q4, q5, q6, q7]
        :return: J - 9 x 7 matrix representing the Jacobian, where the first three rows correspond to
                 the linear velocity and the last three rows correspond to the angular velocity,
                 expressed in world frame coordinates.
        """
        # Initialize FK to use its methods for DH transformations
        fk = FK()
        fk_jac = FK_Jac()

        # Initialize Jacobian 9 X 7 (+2 virtual joints)
        J = np.zeros((9, 7))

        # Get joint positions and full transformation matrix from base to end-effector
        joint_positions, T0e = fk_jac.forward_expanded(q_in)

        # Extract the end-effector position in the world frame
        # 4th columns first three rows
        Oe = T0e[0:3, 3]

        # Use FK to get all Z vectors 3rd col
        Z_axes = fk.get_axis_of_rotation(q_in)

        # Expanded Initialize the linear (J_v) and angular (J_w)
        # velocity components of the Jacobian
        J_v = np.zeros((3, 9))
        J_w = np.zeros((3, 9))

        # Calculate the Jacobian columns for each joint = 9
        for i in range(1, 10):
            Z_prev = Z_axes[:, i - 1]  # Z_i-1
            O_prev = joint_positions[i - 1]  # O_i-1

            # Calculate the linear velocity component for joint i
            J_v[:, i - 1] = np.cross(Z_prev, (Oe - O_prev))
            # The angular velocity component is just the Z vector of that joint
            J_w[:, i - 1] = Z_prev

        # Combine linear and angular components into full Jacobian matrix
        J = np.vstack((J_v, J_w))

        return J
    
if __name__ == "__main__":

    fk = FK_Jac()

    # matches figure in the handout
    q = np.array([0,0,0,-pi/2,0,pi/2,pi/4])

    joint_positions, T0e = fk.forward_expanded(q)
    
    print("Joint Positions:\n",joint_positions)
    print("End Effector Pose:\n",T0e)
