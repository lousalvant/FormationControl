import numpy as np

def col_avoid(dq, q, dcoll, rcoll):
    # Number of agents
    n = len(q) // 2
    
    # Reshape q and dq into 2D matrices
    qm = np.array(q).reshape(2, n)
    ctrl = np.array(dq).reshape(2, n)

    # Inter-agent distance matrix
    Dc = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            Dc[i, j] = np.linalg.norm(qm[:, i] - qm[:, j])
    Dc = Dc + Dc.T  # Make symmetric

    # Collision avoidance index matrix
    colIdx = Dc < dcoll
    np.fill_diagonal(colIdx, 0)  # Set diagonal to 0, no self-collisions

    # Flag to stop movement if no safe direction is found
    stopFlag = np.zeros(n, dtype=bool)

    # Loop over each UAV to adjust its control vector
    for i in range(n):
        coneAng = []

        # Find angles of collision cones
        for k in range(n):
            if colIdx[i, k]:
                dnb = Dc[i, k]
                vec = qm[:, k] - qm[:, i]
                tht = np.degrees(np.arctan2(vec[1], vec[0]))

                # Vertex half-angle of the collision cone
                if dnb <= dcoll:
                    alp = 90
                else:
                    alp = np.degrees(np.arcsin(rcoll / dnb))

                # Cone boundaries
                thtm = tht - alp
                thtp = tht + alp

                # Adjust angles to be within [-180, 180] range
                if thtm < -180:
                    coneAng.append([thtm + 360, 180])
                    coneAng.append([-180, thtp])
                elif thtp > 180:
                    coneAng.append([-180, thtp - 360])
                    coneAng.append([thtm, 180])
                else:
                    coneAng.append([thtm, thtp])

        # Adjust control vector if necessary
        if np.any(colIdx[i, :]):
            thtC = np.degrees(np.arctan2(ctrl[1, i], ctrl[0, i]))  # Current control vector angle

            # Check if the control vector is within any collision cone
            if any([thtm <= thtC <= thtp for thtm, thtp in coneAng]):
                # Search for feasible directions
                angs = np.arange(-180, 181, 5)  # Candidate directions
                angsIdx = np.ones_like(angs, dtype=bool)

                # Mark angles inside the collision cones
                for thtm, thtp in coneAng:
                    angsIdx &= ~((angs >= thtm) & (angs <= thtp))

                angsFeas = angs[angsIdx]

                if len(angsFeas) == 0:
                    stopFlag[i] = True

                # Find the closest feasible direction
                thtDiff = np.abs(np.degrees(np.arctan2(np.sin(np.radians(thtC - angsFeas)),
                                                      np.cos(np.radians(thtC - angsFeas)))))
                minIdx = np.argmin(thtDiff)
                thtCnew = angsFeas[minIdx]

                # Stop if direction change is too large
                if np.abs(np.degrees(np.arctan2(np.sin(np.radians(thtCnew - thtC)),
                                                np.cos(np.radians(thtCnew - thtC))))) >= 90:
                    stopFlag[i] = True

                # Modify control vector
                if stopFlag[i]:
                    ctrl[:, i] = np.zeros(2)
                else:
                    ctrl[:, i] = np.linalg.norm(ctrl[:, i]) * np.array([np.cos(np.radians(thtCnew)),
                                                                        np.sin(np.radians(thtCnew))])

    # Flatten the modified control vector for output
    u = ctrl.flatten()
    
    return u, n, colIdx, Dc
