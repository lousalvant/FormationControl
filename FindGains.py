import numpy as np
import cvxpy as cp

def find_gains(qsMat, adj):
    # Flatten qsMat into a vector
    qs = qsMat.flatten()
    
    # Number of agents
    n = qsMat.shape[1]
    
    # Complex representation of desired formation coordinates
    p = qs[0::2]  # x-coordinates
    q = qs[1::2]  # y-coordinates
    z = p + 1j * q  # complex numbers for each UAV
    
    # Get orthogonal complement of [z ones(n,1)]
    z_augmented = np.column_stack((z, np.ones(n)))
    U, _, _ = np.linalg.svd(z_augmented)
    Q = U[:, 2:]
    
    # Subspace constraint for the given graph
    S = np.logical_not(adj).astype(float)
    np.fill_diagonal(S, 0)
    
    # Solve via CVXPY
    A = cp.Variable((n, n), hermitian=True)
    objective = cp.Maximize(cp.lambda_min(Q.T.conj() @ A @ Q))
    
    constraints = [
        A @ z_augmented == 0, # Enforce complex equality
        cp.norm(A, 'fro') <= 10,
        cp.multiply(S, A) == 0  # Enforce adjacency matrix sparsity pattern
    ]
    
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.SCS)
    
    A_complex = -A.value  # Complex gain matrix
    Ar = complex_to_real(A_complex)  # Convert to real representation
    
    return Ar

def complex_to_real(A):
    n = A.shape[0]
    AnR = np.zeros((2 * n, 2 * n))
    
    for i in range(n):
        for j in range(n):
            lij = A[i, j]
            re = np.real(lij)
            im = np.imag(lij)
            
            AnR[2 * i : 2 * i + 2, 2 * j : 2 * j + 2] = np.array([[re, -im], [im, re]])
    
    return AnR