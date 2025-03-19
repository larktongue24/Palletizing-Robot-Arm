import numpy as np
from math import pi

class FK():

    def __init__(self):

        # TODO: you may want to define geometric parameters here that will be
        # useful in computing the forward kinematics. The data you will need
        # is provided in the lab handout

        self.a = [0, 0, 0.0825, 0.0825, 0, 0.088, 0]
        self.alpha = [-pi/2, pi/2, pi/2, pi/2, -pi/2, pi/2, 0]
        self.d = [0.333, 0, 0.316, 0, 0.384, 0, 0.21]
        self.p = np.array([[0,0,0.141],[0,0,0],[0,0,0.195],[0,0,0],[0,0,0.125],[0,0,-0.015],[0,0,0.051],[0,0,0]])

        pass

    def forward(self, q):
        """
        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]

        OUTPUTS:
        jointPositions -8 x 3 matrix, where each row corresponds to a rotational joint of the robot or end effector
                  Each row contains the [x,y,z] coordinates in the world frame of the respective joint's center in meters.
                  The base of the robot is located at [0,0,0].
        T0e       - a 4 x 4 homogeneous transformation matrix,
                  representing the end effector frame expressed in the
                  world frame
        """

        # Your Lab 1 code starts here
        jointPositions = np.zeros((8,3))
        T0e = np.identity(4)

        # Define the joint angles
        theta = [q[0], q[1], q[2], q[3]+pi, q[4], q[5]-pi, q[6]-pi/4]
        
        # Compute the forward kinematics
        for i in range(7):
            jointPositions[i] = (T0e @ np.append(self.p[i],1))[:3]
            A = self.DHparam2matrix(self.a[i], self.alpha[i], self.d[i], theta[i])
            T0e = T0e @ A

        # Compute the end effector position
        jointPositions[7] = (T0e @ np.append(self.p[7],1))[:3]
        
        return jointPositions, T0e

    def DHparam2matrix(self, a, alpha, d, theta):

        T = np.array([[np.cos(theta), -np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
                      [np.sin(theta), np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
                      [0, np.sin(alpha), np.cos(alpha), d],
                      [0, 0, 0, 1]])

        return T

    # This code is for Lab 2, you can ignore it ofr Lab 1
    def get_axis_of_rotation(self, q):
        """
        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]

        OUTPUTS:
        axis_of_rotation_list: - 3x7 np array of unit vectors describing the axis of rotation for each joint in the
                                 world frame

        """

        axis_of_rotation_list = np.zeros((3,7))
        theta = [q[0], q[1], q[2], q[3]+pi, q[4], q[5]-pi, q[6]-pi/4]
        A = self.compute_Ai(q)
        
        axis_of_rotation_list[:,0] = np.array([0,0,1])
        for i in range(1,7):
            axis_of_rotation_list[:,i] = A[i-1][0:3,2]
            
        return axis_of_rotation_list

    def compute_Ai(self, q):
        """
        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]

        OUTPUTS:
        Ai: - 4x4 list of np array of homogenous transformations describing the FK of the robot. Transformations are not
              necessarily located at the joint locations
        """
        theta = [q[0], q[1], q[2], q[3]+pi, q[4], q[5]-pi, q[6]-pi/4]
        A = np.zeros((7,4,4))
        A[0] = self.DHparam2matrix(self.a[0], self.alpha[0], self.d[0], theta[0])
        for i in range(1,7):
            A[i] = A[i-1] @ self.DHparam2matrix(self.a[i], self.alpha[i], self.d[i], theta[i])
        return A

if __name__ == "__main__":

    fk = FK()

    # matches figure in the handout
    q = np.array([0,0,0,0,0,pi,pi/4])

    joint_positions, T0e = fk.forward(q)

    print(T0e)
    
    print("Joint Positions:\n",joint_positions)
    print("End Effector Pose:\n",T0e)
