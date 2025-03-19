import numpy as np
from lib.calculateFK import FK

def calcJacobian(q_in):
    """
    Calculate the full Jacobian of the end effector in a given configuration
    :param q_in: 1 x 7 configuration vector (of joint angles) [q1,q2,q3,q4,q5,q6,q7]
    :return: J - 6 x 7 matrix representing the Jacobian, where the first three
    rows correspond to the linear velocity and the last three rows correspond to
    the angular velocity, expressed in world frame coordinates
    """

    J = np.zeros((6, 7))
    fk = FK()
    jointPositions, T0e = fk.forward(q_in)
    A = fk.compute_Ai(q_in)
    axis_of_rotation = fk.get_axis_of_rotation(q_in)

    for i in range(7):
        J[:3,i] = np.cross(axis_of_rotation[:,i], jointPositions[7] - jointPositions[i])
        J[3:,i] = axis_of_rotation[:,i]
    
    return J

if __name__ == '__main__':
    q= np.array([0, 0, 0, -np.pi/2, 0, np.pi/2, np.pi/4])
    print(np.round(calcJacobian(q),3))
    