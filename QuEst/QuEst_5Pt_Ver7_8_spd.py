import numpy as np
from numpy.linalg import svd, eig

from .Helpers.Coefs import coefs_ver3_1_1

def quest_5pt_ver7_8_spd(m, n):
    """
    QuEst (Quaternion Estimation) algorithm for 5 feature points (Speed Optimized Version).
    
    Parameters:
    m, n: Homogeneous coordinates of feature points in the first and second frames (3xN matrices).
    
    Returns:
    Q: Recovered quaternion solutions.
    """
    
    # Preallocate index matrix (adjusted for zero indexing)
    Idx = np.array([
        [1, 2, 5, 11, 21, 3, 6, 12, 22, 8, 14, 24, 17, 27, 31, 4, 7, 13, 23, 9, 15, 25, 18, 28, 32, 10, 16, 26, 19, 29, 33, 20, 30, 34, 35],
        [2, 5, 11, 21, 36, 6, 12, 22, 37, 14, 24, 39, 27, 42, 46, 7, 13, 23, 38, 15, 25, 40, 28, 43, 47, 16, 26, 41, 29, 44, 48, 30, 45, 49, 50],
        [3, 6, 12, 22, 37, 8, 14, 24, 39, 17, 27, 42, 31, 46, 51, 9, 15, 25, 40, 18, 28, 43, 32, 47, 52, 19, 29, 44, 33, 48, 53, 34, 49, 54, 55],
        [4, 7, 13, 23, 38, 9, 15, 25, 40, 18, 28, 43, 32, 47, 52, 10, 16, 26, 41, 19, 29, 44, 33, 48, 53, 20, 30, 45, 34, 49, 54, 35, 50, 55, 56]
    ]) - 1  # Adjust for zero indexing in Python
    
    # Construct the coefficient matrix
    Cf = coefs_ver3_1_1(m, n)
    numEq = Cf.shape[0]
    
    A = np.zeros((4 * numEq, 56))
    for i in range(4):
        idx = Idx[i, :]
        A[i * numEq:(i + 1) * numEq, idx] = Cf
    
    # Find the null space of A using SVD
    _, _, V = svd(A, full_matrices=False)
    N = V[:, 36:]  # Corresponds to columns 37-56 (adjusted for zero indexing)
    
    # Solve for B matrix using only B1 (faster computation)
    A0 = N[Idx[0], :]
    A1 = N[Idx[1], :]
    
    B = np.linalg.solve(A0, A1)
    B1 = B[:, :20]
    
    # Compute eigenvectors for B1
    V1, _ = eig(B1)
    
    Ve = V1  # We use only B1 in this optimized version
    
    # Use only real parts of the eigenvectors
    V0 = np.real(Ve)
    
    # Extract quaternion elements from degree 5 monomials
    X5 = N @ V0
    
    # Correct sign of each column so the first element (w^5) is always positive
    X5 = np.sign(X5[0, :]) * X5
    
    # Recover quaternion elements
    w = np.sign(X5[0, :]) * np.abs(X5[0, :]) ** (1 / 5)
    w4 = w ** 4
    x = X5[1, :] / w4
    y = X5[2, :] / w4
    z = X5[3, :] / w4
    
    Q = np.vstack([w, x, y, z])
    
    # Normalize quaternions so that each column has norm 1
    Q = Q / np.linalg.norm(Q, axis=0)
    
    return Q
