# Vision-based formation control of quadrotors in AirSim.

# Notes:
# The simulation environment in UE4 must be running before executing this code.

import airsim
import numpy as np
import time

from FindGains import find_gains
from GetRelativePose import get_relative_pose
from ColAvoid import col_avoid

# Desired formation parameters
# Total number of UAVs
numUAV = 3

# Desired formation (equilateral triangle)
qs = np.array([[0, 4, 2], 
               [0, 0, 3]])

# Adjacency matrix
Adjm = np.array([[0, 1, 1], 
                 [1, 0, 1], 
                 [1, 1, 0]])

# Initial positions of the quads
# Note: This must match the settings file
pos0 = np.zeros((numUAV, 3))
for i in range(numUAV):
    pos0[i, 0] = 0
    pos0[i, 1] = -4 + 4.0 * i
    pos0[i, 2] = 0

# Find formation control gains
A = find_gains(qs, Adjm)
print("Gain matrix calculated.")

# Connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()

for i in range(numUAV):
    name = "UAV" + str(i + 1)
    client.enableApiControl(True, name)
    client.armDisarm(True, name)
print("All UAVs have been initialized.")

# Hover
time.sleep(2)

tout = 3  # Timeout in seconds
spd = 2  # Speed

print("Taking off...")
for i in range(numUAV):
    name = "UAV" + str(i + 1)
    print('Hovering', name)
    client.hoverAsync(vehicle_name=name)
    client.moveToPositionAsync(0, 0, -1, spd, timeout_sec=tout, vehicle_name=name)
print("All UAVs are hovering.")

# Increase altitude
tout = 10.0  # Timeout in seconds
spd = 4.0  # Speed
alt = -20.0  # Altitude

time.sleep(0.5)

for i in range(numUAV):
    name = "UAV" + str(i + 1)
    print('Moving', name)
    client.moveToPositionAsync(0, 0, alt, spd, timeout_sec=tout, vehicle_name=name)
print("UAVs reached desired altitude")

# Formation control loop
dcoll = 1.5  # Collision avoidance activation distance
rcoll = 0.7  # Collision avoidance circle radius
gain = 1.0 / 3  # Control gain
alt = -20.0  # UAV altitude
duration = 0.5  # Max duration for applying input
vmax = 0.1  # Saturation velocity
save = 0  # Set to 1 to save onboard images, otherwise set to 0

# Initial Pause time
time.sleep(0.5)

# Get Image data (Initialization)
imgArray = []
for i in range(numUAV):
    name = f"UAV{i+1}"
    imgs = client.simGetImages([
        airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)], vehicle_name=name)
    img = imgs[0]
    imgArray.extend(img.image_data_uint8)
imgArray = np.array(imgArray, dtype=np.uint8)
imgWidth = img.width
imgHeight = img.height

# Formation control
itr = 0
while True:

    itr += 1
    print("itr =", itr)

    # Get UAV positions for collision avoidance
    q = np.zeros(3 * numUAV)  # Preallocate state vectors
    qo = np.zeros(4 * numUAV)  # Preallocate orientation vectors
    qxy = np.zeros(2 * numUAV)
    for i in range(numUAV):
        name = "UAV" + str(i + 1)

        # Get x-y-z coordinates
        pos = client.simGetGroundTruthKinematics(vehicle_name=name)
        qi = np.array([pos.position.x_val, pos.position.y_val, pos.position.z_val])
        qoi = np.array([pos.orientation.w_val, pos.orientation.x_val, pos.orientation.y_val, pos.orientation.z_val])

        # Add initial coordinates
        qd = qi + pos0[i, :]

        # 3D and 2D state vector
        q[3 * i:3 * i + 3] = qd.copy()
        qxy[2 * i:2 * i + 2] = np.array([qd[0], qd[1]])
        qo[4 * i:4 * i + 4] = qoi.copy()

    # Estimate relative positions using onboard images
    Qm, Tm, flagm = get_relative_pose(imgArray, imgWidth, imgHeight, save, Adjm, itr)
    T = np.asarray(Tm)
    flag = np.asarray(flagm).flatten()

    # Transform recovered coordinates to world frame in order to apply the control.
    # AirSim uses NED (North-East-Down) frame, and the front camera has EDN 
    # (East-Down-North) frame. So we need to swap columns of T.
    Tw = np.array([T[:, 2], T[:, 0], T[:, 1]]).T

    # Calculate distributed control
    dqxy = np.zeros(2 * numUAV)  # Preallocate vectors
    for i in range(numUAV):
        if flag[i] == 1:
            # 3D and 2D state vector
            qi = Tw[i * numUAV: (i + 1) * numUAV, :].flatten()
            qxyi = Tw[i * numUAV: (i + 1) * numUAV, 0:2].flatten()

            # Control
            dqxyi = A[i * 2: i * 2 + 2, :].dot(qxyi)
            dqxy[2 * i:2 * i + 2] = gain * dqxyi
    if save == 1:
        np.save("SavedData/u" + str(itr), dqxy)  # Save control

    # Collision avoidance
    u, _, _, _ = col_avoid(dqxy.tolist(), qxy.tolist(), dcoll, rcoll)

    # Saturate velocity control command
    for i in range(numUAV):
        # Find norm of control vector for each UAV
        ui = u[2 * i:2 * i + 2]
        vel = np.linalg.norm(ui)
        if vel > vmax:
            u[2 * i:2 * i + 2] = (vmax / vel) * u[2 * i:2 * i + 2]
    if save == 1:
        np.save("SavedData/um" + str(itr), u)  # Save modified control

    # Apply control command    
    for i in range(numUAV):
        name = "UAV" + str(i + 1)
        client.moveByVelocityZAsync(u[2 * i], u[2 * i + 1], alt, duration, vehicle_name=name)  # Motion at fixed altitude

    print()

# Terminate
client.reset()
