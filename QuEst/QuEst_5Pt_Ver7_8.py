import numpy as np
from numpy.linalg import svd, eig

from .Helpers.Coefs import coefs_ver3_1_1

def quest_5pt_ver7_8(m, n):
    """
    QuEst (Quaternion Estimation) algorithm for 5 feature points.
    
    Parameters:
    m, n: Homogeneous coordinates of feature points in the first and second frames (3xN matrices).
    
    Returns:
    Q: Recovered quaternion solutions.
    """
    
    # Preallocate indices and matrices (adjust for zero indexing)
    Idx = np.array([
        [1, 2, 5, 11, 21, 3, 6, 12, 22, 8, 14, 24, 17, 27, 31, 4, 7, 13, 23, 9, 15, 25, 18, 28, 32, 10, 16, 26, 19, 29, 33, 20, 30, 34, 35],
        [2, 5, 11, 21, 36, 6, 12, 22, 37, 14, 24, 39, 27, 42, 46, 7, 13, 23, 38, 15, 25, 40, 28, 43, 47, 16, 26, 41, 29, 44, 48, 30, 45, 49, 50],
        [3, 6, 12, 22, 37, 8, 14, 24, 39, 17, 27, 42, 31, 46, 51, 9, 15, 25, 40, 18, 28, 43, 32, 47, 52, 19, 29, 44, 33, 48, 53, 34, 49, 54, 55],
        [4, 7, 13, 23, 38, 9, 15, 25, 40, 18, 28, 43, 32, 47, 52, 10, 16, 26, 41, 19, 29, 44, 33, 48, 53, 20, 30, 45, 34, 49, 54, 35, 50, 55, 56]
    ]) - 1  # Adjust to zero indexing
    
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
    
    # Solve for B matrices
    A0 = N[Idx[0], :]
    A1 = N[Idx[1], :]
    A2 = N[Idx[2], :]
    A3 = N[Idx[3], :]
    
    B = np.linalg.solve(A0, np.hstack([A1, A2, A3]))
    
    B1 = B[:, :20]
    B2 = B[:, 20:40]
    B3 = B[:, 40:60]
    
    # Find eigenvectors of B matrices
    V1, _ = eig(B1)
    V2, _ = eig(B2)
    V3, _ = eig(B3)
    
    Ve = np.hstack([V1, V2, V3])
    
    # Remove duplicate complex eigenvectors
    imagIdx = np.sum(np.abs(np.imag(Ve)), axis=0) > 10 * np.finfo(float).eps
    Viall = Ve[:, imagIdx]
    srtIdx = np.argsort(np.real(Viall[0, :]))
    Vi = Viall[:, srtIdx[::2]]  # Keep only one of the duplicate complex eigenvectors
    Vr = Ve[:, ~imagIdx]
    V0 = np.real(np.hstack([Vi, Vr]))  # Use only the real parts
    
    # Extract quaternion elements from degree 5 monomials
    X5 = N @ V0
    
    # Correct sign of each column so the first element (w^5) is always positive
    X5 = np.sign(X5[0, :]) * X5
    
    # Recover quaternion elements
    w = np.sign(X5[0, :]) * np.abs(X5[0, :]) ** (1 / 5) # Equivalent to fifth root
    w4 = w ** 4
    x = X5[1, :] / w4
    y = X5[2, :] / w4
    z = X5[3, :] / w4
    
    Q = np.vstack([w, x, y, z])
    
    # Normalize quaternions so that each column has norm 1
    Q = Q / np.linalg.norm(Q, axis=0)
    
    return Q
