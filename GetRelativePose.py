import cv2
import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

from QuEst_RANSAC.QuEst_RANSAC import quest_ransac
from QuEst.Helpers.FindTransDepth import find_trans_depth

def get_relative_pose(imgArrayInp, imgWidthInp, imgHeightInp, saveIdx, adj, itr):
    """
    Recover the relative pose from images using QuEst with parallel processing.

    Args:
    - imgArrayInp: Flattened image data from onboard cameras of all UAVs.
    - imgWidthInp: Width of each image.
    - imgHeightInp: Height of each image.
    - saveIdx: Flag to save data (1 to save, 0 not to save).
    - adj: Adjacency matrix indicating neighboring UAVs.
    - itr: Iteration number for saving data.

    Returns:
    - Q: Relative quaternions.
    - T: Relative translations.
    - flag: Indicator vector showing whether the pose was successfully recovered for each UAV.
    """
    # Handle image data
    imgWidth = int(imgWidthInp)
    imgHeight = int(imgHeightInp)
    n = adj.shape[0]  # Number of agents (UAVs)

    # Flag to indicate pose estimation success for each UAV
    flag = np.ones(n, dtype=int)

    # Reshape images into a list of images for each UAV
    imgs = []
    arrayLength = imgWidth * imgHeight

    for i in range(n):
        start_idx = i * arrayLength
        end_idx = (i + 1) * arrayLength
        img_data = imgArrayInp[start_idx:end_idx]

        # Reshape and convert to uint8
        img_gray = np.frombuffer(img_data, dtype=np.uint8).reshape((imgHeight, imgWidth))
        imgs.append(img_gray)

    # Create ORB feature detector
    orb = cv2.ORB_create(nfeatures=1000)

    # Detect and extract feature points
    keypoints_list = []
    descriptors_list = []
    for img_gray in imgs:
        keypoints, descriptors = orb.detectAndCompute(img_gray, None)
        if descriptors is None:
            descriptors = np.array([], dtype=np.uint8).reshape(0, 32)  # ORB descriptors are 32-dimensional
        keypoints_list.append(keypoints)
        descriptors_list.append(descriptors)

    # Match feature points between neighbors using Hamming distance
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = [[[] for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j]:
                if len(descriptors_list[i]) > 0 and len(descriptors_list[j]) > 0:
                    raw_matches = bf.knnMatch(descriptors_list[i], descriptors_list[j], k=2)
                    # Apply Lowe's ratio test
                    good_matches = []
                    for m, n_match in raw_matches:
                        if m.distance < 0.75 * n_match.distance:
                            good_matches.append(m)
                    matches[i][j] = good_matches
                    # Reverse the matches for j to i
                    matches[j][i] = [cv2.DMatch(_m.trainIdx, _m.queryIdx, _m.imgIdx, _m.distance) for _m in good_matches]
                else:
                    matches[i][j] = []
                    matches[j][i] = []

    # Ensure there are enough matched points and find common matched points
    min_pts = 10  # Minimum number of feature points required
    num_mch = np.zeros(n, dtype=int)
    mchs = [[[] for _ in range(n)] for _ in range(n)]
    for i in range(n):
        mchi = None
        for j in range(n):
            if adj[i, j]:
                idx_pairs = np.array([[m.queryIdx, m.trainIdx] for m in matches[i][j]])
                mchs[i][j] = idx_pairs
                if idx_pairs.size > 0:
                    if mchi is None:
                        mchi = idx_pairs[:, 0]
                    else:
                        mchi = np.intersect1d(mchi, idx_pairs[:, 0])
        if mchi is not None and len(mchi) >= min_pts:
            # Update matched points to only include common points
            for j in range(n):
                if adj[i, j]:
                    idx_pairs = mchs[i][j]
                    if idx_pairs.size > 0:
                        common_indices = np.isin(idx_pairs[:, 0], mchi)
                        mchs[i][j] = idx_pairs[common_indices]
            num_mch[i] = len(mchi)
        else:
            flag[i] = 0  # Not enough matched points

    # Pose estimation parameters
    max_z_pts = 20        # Maximum number of points used for translation/depth estimation
    ran_thresh = 1e-6     # RANSAC outlier threshold (adjusted for ORB)

    # Initialize arrays to store quaternions and translations
    Qall = np.ones((4, n, n))
    Tall = np.zeros((3, n, n))

    # Camera calibration matrix
    x0, y0 = imgWidth / 2.0, imgHeight / 2.0
    fx, fy = x0, x0
    K = np.array([[fx, 0, x0],
                  [0, fy, y0],
                  [0,  0,  1]])

    # Prepare data for parallel processing
    tasks = []
    for i in range(n):
        if flag[i]:
            tasks.append((i, n, adj, K, min_pts, max_z_pts, ran_thresh, mchs, keypoints_list, flag))

    # Function to process each UAV (to be run in parallel)
    def process_uav(task):
        i, n, adj, K, min_pts, max_z_pts, ran_thresh, mchs, keypoints_list, flag = task
        Q = np.zeros((4, n))
        T = np.zeros((3, n))
        for j in range(n):
            if adj[i, j]:
                idx_pairs = mchs[i][j]
                if len(idx_pairs) < min_pts:
                    flag[i] = 0
                    return i, Q, T, flag[i]

                # Get the matched keypoints
                pi = np.array([keypoints_list[i][idx].pt for idx in idx_pairs[:, 0]])
                pj = np.array([keypoints_list[j][idx].pt for idx in idx_pairs[:, 1]])

                # Convert to homogeneous coordinates
                num_pts = pi.shape[0]
                mi = np.linalg.inv(K) @ np.hstack((pi, np.ones((num_pts, 1)))).T
                mj = np.linalg.inv(K) @ np.hstack((pj, np.ones((num_pts, 1)))).T

                # Pose estimation using quest_ransac
                M, inliers = quest_ransac(mj, mi, ran_thresh)

                if M is None or inliers is None or len(inliers) < min_pts:
                    flag[i] = 0
                    return i, Q, T, flag[i]

                Q[:, j] = M['Q']

                # Recover translation and depth
                mi_inliers = mi[:, inliers]
                mj_inliers = mj[:, inliers]
                num_z_pts = min(max_z_pts, mi_inliers.shape[1])
                tij, z1, z2, R, res = find_trans_depth(mj_inliers[:, :num_z_pts], mi_inliers[:, :num_z_pts], M['Q'])

                # Common scale
                s = np.mean(np.abs(z1))
                T[:, j] = tij / s if s != 0 else tij
        return i, Q, T, flag[i]

    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_uav, task) for task in tasks]
        for future in as_completed(futures):
            i, Q_i, T_i, flag_i = future.result()
            if flag_i == 0:
                flag[i] = 0
            else:
                Qall[:, :, i] = Q_i
                Tall[:, :, i] = T_i

    # Generate output
    Q = np.zeros((n * n, 4))
    T = np.zeros((n * n, 3))

    for i in range(n):
        for j in range(n):
            if i != j:
                idx = i * n + j
                Q[idx, :] = Qall[:, j, i]
                T[idx, :] = Tall[:, j, i]

    # Save data if necessary
    if saveIdx == 1:
        if not os.path.exists('SavedData'):
            os.makedirs('SavedData')
        filename = os.path.join('SavedData', f'Data{itr}.npz')
        np.savez(filename, Q=Q, T=T, flag=flag)

    return Q, T, flag