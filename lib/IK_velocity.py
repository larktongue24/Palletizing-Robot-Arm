import numpy as np 
from lib.calcJacobian import calcJacobian


def IK_velocity(q_in, v_in, omega_in):
    """
    :param q_in: 1 x 7 vector corresponding to the robot's current configuration.
    :param v_in: The desired linear velocity in the world frame. If any element is
    Nan, then that velocity can be anything
    :param omega_in: The desired angular velocity in the world frame. If any
    element is Nan, then that velocity is unconstrained i.e. it can be anything
    :return:
    dq - 1 x 7 vector corresponding to the joint velocities. If v_in and omega_in
         are infeasible, then dq should minimize the least squares error. If v_in
         and omega_in have multiple solutions, then you should select the solution
         that minimizes the l2 norm of dq
    """

    dq = np.zeros((1, 7))

    v_in = v_in.reshape((3,1))
    omega_in = omega_in.reshape((3,1))

    Xi_in = np.vstack((v_in, omega_in))
    constrained_indices = np.where(~np.isnan(Xi_in))[0]
    Xi = Xi_in[constrained_indices].reshape(-1,1)
    
    J = calcJacobian(q_in)
    J = J[constrained_indices,:]
    dq = np.linalg.lstsq(J, Xi, rcond=None)[0].squeeze()
    
    return dq