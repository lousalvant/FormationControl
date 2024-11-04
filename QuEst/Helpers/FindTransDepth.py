import numpy as np
from scipy.linalg import svd

from .Q2R import q2r

def find_trans_depth(m, n, Q):
    """
    Recover translation and depths given the rotation and feature point coordinates.

    Parameters:
    m, n: Homogeneous coordinates of matched feature points in first and second frames
    Q: Rotation matrix or quaternion (if quaternion, it will be converted)

    Returns:
    T: Associated translation vector
    Z1: Depths in the first camera frame
    Z2: Depths in the second camera frame
    R: Rotation matrix representation
    Res: Residue from SVD
    """

    # Convert quaternion to rotation matrix if necessary
    if Q.shape[0] != 3:
        R = q2r(Q)
    else:
        R = Q

    # Initialize variables
    I = np.eye(3)  # Identity matrix
    numPts = m.shape[1]  # Number of feature points
    numInp = R.shape[2]  # Number of rotation matrices
    T = np.zeros((3, numInp))  # Translation vector
    Z1 = np.zeros((numPts, numInp))  # Depths in the first camera frame
    Z2 = np.zeros((numPts, numInp))  # Depths in the second camera frame
    Res = np.zeros(numInp)  # Residue from SVD

    for k in range(numInp):
        # Stack rigid motion constraints into matrix-vector form C * Y = 0
        C = np.zeros((3 * numPts, 2 * numPts + 3))
        for i in range(numPts):
            C[3 * i:3 * i + 3, 0:3] = I
            C[3 * i:3 * i + 3, 2 * i + 3:2 * i + 5] = np.hstack([R[:, :, k] @ m[:, i].reshape(-1, 1), -n[:, i].reshape(-1, 1)])

        # Use SVD to find singular vectors
        _, S, N = svd(C, full_matrices=False)

        # The right singular vector corresponding to the smallest singular value
        Y = N[-1, :]  # Y is the last row of N (right singular vector)

        t = Y[:3]  # Translation vector
        z = Y[3:]  # Depths in both camera frames

        # Adjust the sign so that the recovered depths are positive
        numPos = np.sum(z > 0)
        numNeg = np.sum(z < 0)
        if numPos < numNeg:
            t = -t
            z = -z

        z1 = z[0::2]  # Depths in camera frame 1
        z2 = z[1::2]  # Depths in camera frame 2

        # Store the results
        T[:, k] = t
        Z1[:, k] = z1
        Z2[:, k] = z2
        Res[k] = S[-1]  # The smallest singular value is the residue

    return T, Z1, Z2, R, Res
