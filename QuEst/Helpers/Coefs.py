import numpy as np

# Helper functions for coefficients

def coefs_num_ver2_0(mx1, mx2, my1, my2, nx2, ny2, r2, s1, s2):
    t2 = mx1 * my2 * r2
    t3 = mx2 * ny2 * s1
    t4 = my1 * nx2 * s2
    t5 = mx1 * nx2 * s2 * 2.0
    t6 = my1 * ny2 * s2 * 2.0
    t7 = mx1 * my2 * nx2 * 2.0
    t8 = my2 * r2 * s1 * 2.0
    t9 = mx2 * my1 * r2
    t10 = mx1 * ny2 * s2
    t11 = mx2 * my1 * ny2 * 2.0
    t12 = mx2 * r2 * s1 * 2.0
    t13 = my2 * nx2 * s1
    
    coefsN = np.column_stack([
        t2 + t3 + t4 - mx2 * my1 * r2 - mx1 * ny2 * s2 - my2 * nx2 * s1,
        t11 + t12 - mx1 * my2 * ny2 * 2.0 - mx1 * r2 * s2 * 2.0,
        t7 + t8 - mx2 * my1 * nx2 * 2.0 - my1 * r2 * s2 * 2.0,
        t5 + t6 - mx2 * nx2 * s1 * 2.0 - my2 * ny2 * s1 * 2.0,
        -t2 - t3 + t4 + t9 + t10 - my2 * nx2 * s1,
        -t5 + t6 + mx2 * nx2 * s1 * 2.0 - my2 * ny2 * s1 * 2.0,
        t7 - t8 - mx2 * my1 * nx2 * 2.0 + my1 * r2 * s2 * 2.0,
        -t2 + t3 - t4 + t9 - t10 + t13,
        -t11 + t12 + mx1 * my2 * ny2 * 2.0 - mx1 * r2 * s2 * 2.0,
        t2 - t3 - t4 - t9 + t10 + t13
    ])
    
    return coefsN

def coefs_den_ver2_0(mx2, my2, nx1, nx2, ny1, ny2, r1, r2, s2):
    t2 = mx2 * ny1 * r2
    t3 = my2 * nx2 * r1
    t4 = nx1 * ny2 * s2
    t5 = mx2 * nx2 * r1 * 2.0
    t6 = my2 * ny2 * r1 * 2.0
    t7 = mx2 * nx2 * ny1 * 2.0
    t8 = ny1 * r2 * s2 * 2.0
    t9 = my2 * nx1 * r2
    t10 = nx2 * ny1 * s2
    t11 = my2 * nx1 * ny2 * 2.0
    t12 = nx1 * r2 * s2 * 2.0
    t13 = mx2 * ny2 * r1
    
    coefsD = np.column_stack([
        t2 + t3 + t4 - mx2 * ny2 * r1 - my2 * nx1 * r2 - nx2 * ny1 * s2,
        t11 + t12 - my2 * nx2 * ny1 * 2.0 - nx2 * r1 * s2 * 2.0,
        t7 + t8 - mx2 * nx1 * ny2 * 2.0 - ny2 * r1 * s2 * 2.0,
        t5 + t6 - mx2 * nx1 * r2 * 2.0 - my2 * ny1 * r2 * 2.0,
        t2 - t3 - t4 + t9 + t10 - mx2 * ny2 * r1,
        t5 - t6 - mx2 * nx1 * r2 * 2.0 + my2 * ny1 * r2 * 2.0,
        -t7 + t8 + mx2 * nx1 * ny2 * 2.0 - ny2 * r1 * s2 * 2.0,
        -t2 + t3 - t4 - t9 + t10 + t13,
        t11 - t12 - my2 * nx2 * ny1 * 2.0 + nx2 * r1 * s2 * 2.0,
        -t2 - t3 + t4 + t9 - t10 + t13
    ])
    
    return coefsD

def coefs_num_den(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10):
    coefsND = np.column_stack([
        a1 * b1, a1 * b2 + a2 * b1, a2 * b2 + a1 * b5 + a5 * b1, a2 * b5 + a5 * b2, a5 * b5,
        a1 * b3 + a3 * b1, a2 * b3 + a3 * b2 + a1 * b6 + a6 * b1, a2 * b6 + a3 * b5 + a5 * b3 + a6 * b2, a5 * b6 + a6 * b5,
        a3 * b3 + a1 * b8 + a8 * b1, a3 * b6 + a6 * b3 + a2 * b8 + a8 * b2, a6 * b6 + a5 * b8 + a8 * b5,
        a3 * b8 + a8 * b3, a6 * b8 + a8 * b6, a8 * b8, a1 * b4 + a4 * b1,
        a2 * b4 + a4 * b2 + a1 * b7 + a7 * b1, a2 * b7 + a4 * b5 + a5 * b4 + a7 * b2,
        a5 * b7 + a7 * b5, a3 * b4 + a4 * b3 + a1 * b9 + a9 * b1, a3 * b7 + a4 * b6 + a6 * b4 + a7 * b3 + a2 * b9 + a9 * b2,
        a6 * b7 + a7 * b6 + a5 * b9 + a9 * b5, a3 * b9 + a4 * b8 + a8 * b4 + a9 * b3, a6 * b9 + a7 * b8 + a8 * b7 + a9 * b6,
        a8 * b9 + a9 * b8, a4 * b4 + a1 * b10 + a10 * b1, a4 * b7 + a7 * b4 + a2 * b10 + a10 * b2,
        a7 * b7 + a5 * b10 + a10 * b5, a3 * b10 + a4 * b9 + a9 * b4 + a10 * b3, a6 * b10 + a7 * b9 + a9 * b7 + a10 * b6,
        a8 * b10 + a9 * b9 + a10 * b8, a4 * b10 + a10 * b4, a7 * b10 + a10 * b7, a9 * b10 + a10 * b9, a10 * b10
    ])
    
    return coefsND

def coefs_ver3_1_1(m1, m2):
    """
    Coefficient matrix generator for the 5-point pose estimation problem.
    
    Parameters:
    m1, m2: Homogeneous coordinates of feature points in the first and second frames (3x5 matrices).
    
    Returns:
    C: The 11x35 coefficient matrix.
    """
    
    # Number of feature points
    num_pts = m1.shape[1]
    
    # Setup for coefficient calculation
    idx_bin1 = np.zeros((2, np.math.comb(num_pts, 2)), dtype=int)
    counter = 0
    for i in range(num_pts - 1):
        for j in range(i + 1, num_pts):
            idx_bin1[:, counter] = [i, j]
            counter += 1
    
    # Extract coordinate values from the matched points
    mx1 = m1[0, idx_bin1[0, :]]
    my1 = m1[1, idx_bin1[0, :]]
    nx1 = m2[0, idx_bin1[0, :]]
    ny1 = m2[1, idx_bin1[0, :]]
    mx2 = m1[0, idx_bin1[1, :]]
    my2 = m1[1, idx_bin1[1, :]]
    nx2 = m2[0, idx_bin1[1, :]]
    ny2 = m2[1, idx_bin1[1, :]]
    
    s1 = m1[2, idx_bin1[0, :]]
    r1 = m2[2, idx_bin1[0, :]]
    s2 = m1[2, idx_bin1[1, :]]
    r2 = m2[2, idx_bin1[1, :]]
    
    # Calculate numerator and denominator coefficients
    coefsN = coefs_num_ver2_0(mx1, mx2, my1, my2, nx2, ny2, r2, s1, s2)
    coefsD = coefs_den_ver2_0(mx2, my2, nx1, nx2, ny1, ny2, r1, r2, s2)
    
    # Setup for combining coefficients
    num_eq = np.math.comb(num_pts, 3)
    idx_bin2 = np.zeros((2, num_eq), dtype=int)
    counter = 0
    counter2 = 0
    for i in range(num_pts - 1, 1, -1):
        for j in range(counter2, i - 1 + counter2):
            for k in range(j + 1, i + counter2):
                idx_bin2[:, counter] = [j, k]
                counter += 1
        counter2 += i
    
    # Combine numerator and denominator
    a1 = np.concatenate([coefsN[idx_bin2[0, :], 0], coefsD[idx_bin2[0, :], 0]])
    a2 = np.concatenate([coefsN[idx_bin2[0, :], 1], coefsD[idx_bin2[0, :], 1]])
    a3 = np.concatenate([coefsN[idx_bin2[0, :], 2], coefsD[idx_bin2[0, :], 2]])
    a4 = np.concatenate([coefsN[idx_bin2[0, :], 3], coefsD[idx_bin2[0, :], 3]])
    a5 = np.concatenate([coefsN[idx_bin2[0, :], 4], coefsD[idx_bin2[0, :], 4]])
    a6 = np.concatenate([coefsN[idx_bin2[0, :], 5], coefsD[idx_bin2[0, :], 5]])
    a7 = np.concatenate([coefsN[idx_bin2[0, :], 6], coefsD[idx_bin2[0, :], 6]])
    a8 = np.concatenate([coefsN[idx_bin2[0, :], 7], coefsD[idx_bin2[0, :], 7]])
    a9 = np.concatenate([coefsN[idx_bin2[0, :], 8], coefsD[idx_bin2[0, :], 8]])
    a10 = np.concatenate([coefsN[idx_bin2[0, :], 9], coefsD[idx_bin2[0, :], 9]])
    
    b1 = np.concatenate([coefsD[idx_bin2[1, :], 0], coefsN[idx_bin2[1, :], 0]])
    b2 = np.concatenate([coefsD[idx_bin2[1, :], 1], coefsN[idx_bin2[1, :], 1]])
    b3 = np.concatenate([coefsD[idx_bin2[1, :], 2], coefsN[idx_bin2[1, :], 2]])
    b4 = np.concatenate([coefsD[idx_bin2[1, :], 3], coefsN[idx_bin2[1, :], 3]])
    b5 = np.concatenate([coefsD[idx_bin2[1, :], 4], coefsN[idx_bin2[1, :], 4]])
    b6 = np.concatenate([coefsD[idx_bin2[1, :], 5], coefsN[idx_bin2[1, :], 5]])
    b7 = np.concatenate([coefsD[idx_bin2[1, :], 6], coefsN[idx_bin2[1, :], 6]])
    b8 = np.concatenate([coefsD[idx_bin2[1, :], 7], coefsN[idx_bin2[1, :], 7]])
    b9 = np.concatenate([coefsD[idx_bin2[1, :], 8], coefsN[idx_bin2[1, :], 8]])
    b10 = np.concatenate([coefsD[idx_bin2[1, :], 9], coefsN[idx_bin2[1, :], 9]])
    
    coefsND = coefs_num_den(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10,
                            b1, b2, b3, b4, b5, b6, b7, b8, b9, b10)
    
    # Generate final coefficient matrix C
    C = coefsND[:num_eq, :] - coefsND[num_eq:, :]
    
    return C