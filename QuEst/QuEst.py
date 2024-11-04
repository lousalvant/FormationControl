from .QuEst_5Pt_Ver5_2 import quest_5pt_ver5_2
from .QuEst_5Pt_Ver7_8 import quest_5pt_ver7_8
from .QuEst_5Pt_Ver7_8_spd import quest_5pt_ver7_8_spd
from .Helpers.FindTransDepth import find_trans_depth

def quest(m, n):
    """
    QuEst (Quaternion pose Estimation) algorithm.

    Parameters:
    m, n: Homogeneous coordinates of feature points in the first and second frames (3xN matrices).

    Returns:
    sol: A dictionary containing the recovered pose information.
        - sol['Q']: The recovered rotation in quaternions.
        - sol['R']: The recovered rotation as a rotation matrix.
        - sol['T']: The associated translation vectors.
        - sol['Z1']: Depths in the first camera frame.
        - sol['Z2']: Depths in the second camera frame.
    """
    
    # Find the rotation in the form of quaternions
    Q = quest_5pt_ver5_2(m,n)            # Default
    # Q = quest_5pt_ver7_8(m,n)          # More accurate
    # Q = quest_5pt_ver7_8_spd(m,n)      # Faster execution

    # Each column of 'T' is the (unit norm) translation recovered for the corresponding quaternion.
    # 'Z1' and 'Z2' are depths of the feature points in the first and second camera frames.
    T, Z1, Z2, R, Res = find_trans_depth(m, n, Q)

    # Step 3: Store results in a dictionary
    sol = {
        'Q': Q,
        'R': R,
        'T': T,
        'Z1': Z1,
        'Z2': Z2
    }

    return sol
