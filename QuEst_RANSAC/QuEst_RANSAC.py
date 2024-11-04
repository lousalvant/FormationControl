import numpy as np

from QuEst.QuEst import quest
from QuEst.Helpers.Q2R import q2r
from .Helpers.QuatResidue import quat_residue
from .Helpers.Skew import skew

def quest_ransac(x1, x2, t, feedback=0):
    """
    Estimates the pose between two camera views using RANSAC and QuEst algorithm.

    Args:
    - x1: 3xN set of matched feature point coordinates (Euclidean coordinates) in the first view.
    - x2: 3xN set of matched feature point coordinates in the second view such that x1 matches x2.
    - t: Distance threshold between data points and model used to decide inliers.
    - feedback: Optional flag for feedback (default: 0).

    Returns:
    - M: Structure that contains the best estimated pose.
    - inliers: Indices of inliers from the best estimated pose.
    """
    s = 6  # Number of points required to uniquely fit a fundamental matrix

    # Check input validity
    if x1.shape != x2.shape:
        raise ValueError("The shape of x1 and x2 must be identical.")
    
    if x1.shape[1] < s:
        M = None
        inliers = None
        print("RANSAC was unable to find a useful solution. Not enough points.")
        return M, inliers

    # Normalize the points using L1 norm
    x1n = np.sum(np.abs(x1), axis=0)
    x2n = np.sum(np.abs(x2), axis=0)
    x1_normalized = x1 / x1n
    x2_normalized = x2 / x2n

    # Stack x1 and x2 to form a 6xN array for RANSAC
    data = np.vstack((x1_normalized, x2_normalized))

    # Define the functions needed by RANSAC
    fittingfn = pose_estimator
    distfn = fund_dist
    degenfn = is_degenerate

    # Run RANSAC
    M, inliers = ransac(data, fittingfn, distfn, degenfn, s, t, feedback)
    
    return M, inliers


def fund_dist(M, x, t):
    """
    Distance function to evaluate the fit of a fundamental matrix with respect to a set of matched points.

    Args:
    - M: The model containing the fundamental matrix.
    - x: Stacked array of matched points (6xN).
    - t: Distance threshold to determine inliers.

    Returns:
    - inliers: Indices of inliers.
    - bestM: The best model with the most inliers.
    """
    x1 = x[:3, :]  # Extract x1 and x2 from x
    x2 = x[3:, :]

    F = M['F']

    if isinstance(F, list):  # Multiple solutions to test
        nF = len(F)
        bestM = M.copy()
        bestM['F'] = F[0]
        ninliers = 0

        for k in range(nF):
            Fk = F[k]
            x2tFx1 = np.einsum('ij,ji->i', x2.T, Fk @ x1)

            Fx1 = Fk @ x1
            Ftx2 = Fk.T @ x2

            # Evaluate distances
            d = (x2tFx1 ** 2) / (
                Fx1[0, :] ** 2 + Fx1[1, :] ** 2 + Ftx2[0, :] ** 2 + Ftx2[1, :] ** 2
            )

            inliers = np.where(np.abs(d) < t)[0]

            if len(inliers) > ninliers:
                ninliers = len(inliers)
                bestM['F'] = Fk
                bestInliers = inliers
    else:  # Single solution
        x2tFx1 = np.einsum('ij,ji->i', x2.T, F @ x1)

        Fx1 = F @ x1
        Ftx2 = F.T @ x2

        # Evaluate distances
        d = (x2tFx1 ** 2) / (
            Fx1[0, :] ** 2 + Fx1[1, :] ** 2 + Ftx2[0, :] ** 2 + Ftx2[1, :] ** 2
        )

        bestInliers = np.where(np.abs(d) < t)[0]
        bestM = M

    return bestInliers, bestM


def is_degenerate(x):
    """
    Checks if a set of matched points will result in degeneracy in the calculation of the fundamental matrix.

    Parameters:
    x : numpy.ndarray
        Subset of points to test.

    Returns:
    r : int
        0 if non-degenerate, 1 if degenerate.
    """
    return 0  # Assuming degeneracy cannot happen


def pose_estimator(*args):
    """
    Estimates the relative rotation and translation between two camera views using the 5-point quaternion algorithm.

    Parameters:
    *args : variable length argument list
        Either two arrays x1 and x2, or a single array x.

    Returns:
    M : dict
        Dictionary containing the estimated pose.

    Raises:
    ValueError: If the input arguments are not in the expected format.
    """
    # Use the checkargs function to parse and validate inputs
    x1, x2, npts = checkargs(args)

    # Recover the pose using only the first 5 points (consistent with MATLAB code)
    pose = quest(x1[:, :5], x2[:, :5])  # QuEst algorithm

    # Pick the best pose solution
    res = quat_residue(x1, x2, pose['Q'])  # Scoring function
    mIdx = np.argmin(np.abs(res))
    q = pose['Q'][:, mIdx]
    t = pose['T'][:, mIdx]

    # Make the fundamental matrix from recovered rotation and translation
    R = q2r(q)
    Tx = skew(t / np.linalg.norm(t))
    F = Tx @ R

    M = {'Q': q, 't': t, 'm1': x1, 'm2': x2, 'F': F}

    return M


def checkargs(args):
    """
    Checks the arguments and returns x1, x2, and npts.

    Parameters:
    args : tuple
        Variable length argument list.

    Returns:
    x1 : numpy.ndarray
        3xN array of matched feature points in the first view.
    x2 : numpy.ndarray
        3xN array of matched feature points in the second view.
    npts : int
        Number of points.

    Raises:
    ValueError: If the input arguments are not in the expected format.
    """
    if len(args) == 2:
        x1 = args[0]
        x2 = args[1]
        if x1.shape != x2.shape:
            raise ValueError("Image datasets must have the same size.")
        elif x1.shape[0] != 3:
            raise ValueError("Image coordinates must be in a 3xN matrix.")
    elif len(args) == 1:
        x = args[0]
        if x.shape[0] != 6:
            raise ValueError("Single input argument must be a 6xN matrix.")
        else:
            x1 = x[:3, :]
            x2 = x[3:6, :]
    else:
        raise ValueError("Wrong number of arguments supplied.")

    npts = x1.shape[1]
    if npts < 6:
        raise ValueError("At least 6 points are needed to compute the fundamental matrix.")

    return x1, x2, npts


def ransac(x, fittingfn, distfn, degenfn, s, t, feedback=0, maxDataTrials=100, maxTrials=1000):
    """
    RANSAC algorithm to robustly fit a model to data.

    Parameters:
    x : numpy.ndarray
        Data set of size [d x Npts].
    fittingfn : function
        Function to fit a model to `s` data points from `x`.
    distfn : function
        Function to evaluate distances from the model to data `x`.
    degenfn : function
        Function to check for degeneracy in a set of points.
    s : int
        Minimum number of samples required by `fittingfn`.
    t : float
        Distance threshold.
    feedback : int, optional
        Flag for feedback during execution. Default is 0.
    maxDataTrials : int, optional
        Maximum attempts to select non-degenerate data. Default is 100.
    maxTrials : int, optional
        Maximum number of RANSAC iterations. Default is 1000.

    Returns:
    M : dict
        Model with the most inliers.
    inliers : numpy.ndarray
        Indices of inliers.
    """
    npts = x.shape[1]
    p = 0.99  # Desired probability of choosing at least one sample free from outliers

    bestM = None
    bestInliers = None
    bestScore = 0
    N = 1
    trialCount = 0

    while N > trialCount:
        degenerate = True
        count = 0

        while degenerate:
            # Randomly select s data points
            indices = np.random.choice(npts, s, replace=False)
            subset = x[:, indices]

            # Check for degeneracy
            degenerate = degenfn(subset)

            if not degenerate:
                # Fit model
                M = fittingfn(subset)
                if M is None or 'F' not in M or M['F'] is None:
                    degenerate = True

            count += 1
            if count > maxDataTrials:
                print("Unable to select a non-degenerate data set.")
                break

        if degenerate:
            continue

        # Compute distances and inliers
        inliers, M = distfn(M, x, t)

        ninliers = len(inliers)

        if ninliers > bestScore:
            bestScore = ninliers
            bestInliers = inliers
            bestM = M

            # Update N
            fracInliers = ninliers / npts
            pNoOutliers = 1 - fracInliers ** s
            pNoOutliers = max(np.finfo(float).eps, pNoOutliers)
            pNoOutliers = min(1 - np.finfo(float).eps, pNoOutliers)
            N = np.log(1 - p) / np.log(pNoOutliers)
            N = int(np.ceil(N))

        trialCount += 1
        if feedback:
            print(f"Trial {trialCount} out of {N}")

        if trialCount > maxTrials:
            print(f"RANSAC reached the maximum number of {maxTrials} trials.")
            break

    if bestM is not None:
        return bestM, bestInliers
    else:
        print("RANSAC was unable to find a useful solution.")
        return None, None