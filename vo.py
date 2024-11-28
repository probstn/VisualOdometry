import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob

# Function to compute the essential matrix and camera pose
def compute_camera_pose(E, K, prev_points, curr_points):
    """
    Compute rotation (R) and translation (t) from the essential matrix.
    """
    retval, R, t, mask = cv2.recoverPose(E, prev_points, curr_points, cameraMatrix=K)
    return R, t

# Folder with images
image_folder = './image_0/*.png'
# Get list of images from the folder
image_files = sorted(glob.glob(image_folder))

# Camera intrinsic matrix for KITTI dataset (monocular)
K = np.array([[721.5377, 0, 609.5593],
              [0, 721.5377, 172.8540],
              [0, 0, 1]])

# Initialize ORB detector with 2000 keypoints
orb = cv2.ORB_create(nfeatures=5000)

# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Initialize variables
image1 = None
points1 = None
trajectory = [[0, 0]]  # Stores camera positions
R_total = np.eye(3)    # Initial rotation (identity matrix)
t_total = np.zeros((3, 1))  # Initial translation (zero vector)

# Setup live trajectory plot
plt.ion()
fig, ax = plt.subplots()
trajectory_plot, = ax.plot([], [], '-o', label="Camera trajectory")
ax.set_xlim(-250, 250)
ax.set_ylim(-250, 50)
ax.set_xlabel("X-axis")
ax.set_ylabel("Z-axis")
ax.set_title("Live Visual Odometry - Trajectory")
ax.legend()
plt.grid()



# Loop through all the images
for image_path in image_files:
    # Read the current image
    image2 = cv2.imread(image_path)
    
    if image2 is None:
        continue

    # Convert the images to grayscale
    image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    
    # Detect keypoints and compute the descriptors
    if image1 is None:
        image1 = image2_gray
        keypoints1 = orb.detect(image1, None)
        points1 = cv2.KeyPoint_convert(keypoints1)
        continue

    # If tracking points fall below the threshold, recompute ORB keypoints
    if points1 is None or len(points1) < 2500:
        keypoints1 = orb.detect(image1, None)
        points1 = cv2.KeyPoint_convert(keypoints1)

    # Calculate optical flow
    points2, status, err = cv2.calcOpticalFlowPyrLK(image1, image2_gray, points1, None, **lk_params)

    # Select successfully tracked points
    good_new = points2[status.ravel() == 1]
    good_old = points1[status.ravel() == 1]

    # Compute the essential matrix
    if len(good_new) >= 8:  # Minimum points needed for E estimation
        E, mask = cv2.findEssentialMat(good_new, good_old, K, method=cv2.RANSAC, prob=0.999, threshold=2)
        R, t = compute_camera_pose(E, K, good_old, good_new)

        # Update the global pose
        R_total = R @ R_total
        t_total += R_total @ t

        # Add the current position to the trajectory
        trajectory.append([t_total[0, 0], t_total[2, 0]])

        # Update the live trajectory plot
        trajectory_array = np.array(trajectory)
        trajectory_plot.set_data(trajectory_array[:, 0], trajectory_array[:, 1])
        plt.pause(0.01)

    # Draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = map(int, new.ravel())  # Cast to integers
        c, d = map(int, old.ravel())  # Cast to integers
        image2 = cv2.line(image2, (a, b), (c, d), (0, 255, 0), 2)
        image2 = cv2.circle(image2, (a, b), 5, (0, 0, 255), -1)

    # Display the image with the optical flow tracks
    cv2.imshow('Optical Flow', image2)

    # Update points and previous image for the next frame
    points1 = good_new.reshape(-1, 1, 2)
    image1 = image2_gray.copy()

cv2.destroyAllWindows()
