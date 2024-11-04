import numpy as np

def q2r(Q):
    """
    Convert quaternion Q = [q0, q1, q2, q3]^T into the equivalent rotation matrix R.
    
    Parameters:
        Q: 4xN array where each column is a quaternion [q0, q1, q2, q3]^T
    
    Returns:
        R: 3x3xN array of rotation matrices for each input quaternion.
    """
    num_inp = Q.shape[1]
    R = np.zeros((3, 3, num_inp))
    
    for i in range(num_inp):
        q = Q[:, i]

        # If the first quaternion element is negative, negate the quaternion
        if q[0] < 0:
            q = -q

        R[:, :, i] = np.array([
            [1 - 2 * q[2]**2 - 2 * q[3]**2,     2 * q[1] * q[2] - 2 * q[3] * q[0],     2 * q[1] * q[3] + 2 * q[2] * q[0]],
            [2 * q[1] * q[2] + 2 * q[3] * q[0], 1 - 2 * q[1]**2 - 2 * q[3]**2,         2 * q[2] * q[3] - 2 * q[1] * q[0]],
            [2 * q[1] * q[3] - 2 * q[2] * q[0], 2 * q[2] * q[3] + 2 * q[1] * q[0],     1 - 2 * q[1]**2 - 2 * q[2]**2]
        ])
    
    return R

# Example usage:
# Q = np.array([[0.707, 0.0, 0.707, 0.0], [1, 0, 0, 0]]).T  # Two example quaternions
# R = q2r(Q)
# print(R)
