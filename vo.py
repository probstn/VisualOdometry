import cv2
import numpy as np
import glob
import os
import matplotlib.pyplot as plt

# Function to load images from the specified directory
def load_images_from_folder(folder):
    images = sorted(glob.glob(os.path.join(folder, '*.png')))
    return [cv2.imread(img, cv2.IMREAD_GRAYSCALE) for img in images]

# Feature matching function
def feature_matching(img1, img2, detector):
    kp1, des1 = detector.detectAndCompute(img1, None)
    kp2, des2 = detector.detectAndCompute(img2, None)

    # Match descriptors using FLANN
    index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE for SIFT
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Filter matches using Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

    return pts1, pts2, good_matches

# Main visual odometry function
def visual_odometry(image_folder):
    images = load_images_from_folder(image_folder)
    if len(images) < 2:
        print("Not enough images for visual odometry.")
        return

    # Initialize feature detector (ORB/SIFT)
    detector = cv2.SIFT_create()

    # Intrinsic camera matrix (modify for your camera setup)
    K = np.array([[718.856, 0, 607.1928],
                  [0, 718.856, 185.2157],
                  [0, 0, 1]])

    # Initialize pose and trajectory
    trajectory = []
    R, t = np.eye(3), np.zeros((3, 1))

    # Loop through image pairs
    for i in range(300):#len(images) - 1):
        img1, img2 = images[i], images[i + 1]

        # Feature matching
        pts1, pts2, matches = feature_matching(img1, img2, detector)

        # Estimate essential matrix
        E, mask = cv2.findEssentialMat(pts2, pts1, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        _, R1, t1, mask_pose = cv2.recoverPose(E, pts2, pts1, K)

        # Update pose
        R = R @ R1
        t = t + R @ t1

        # Save trajectory
        trajectory.append(t.ravel())

        # Draw matches (optional)
        match_img = cv2.drawMatches(img1, detector.detect(img1, None), img2, detector.detect(img2, None), matches, None)
        cv2.imshow("Matches", match_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Visualize trajectory
    trajectory = np.array(trajectory)
    plt.plot(trajectory[:, 0], trajectory[:, 2], marker='o')
    plt.title("Visual Odometry Trajectory")
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.show()

    cv2.destroyAllWindows()

# Run visual odometry
image_folder = './image_0'  # Specify the path to your image sequence
visual_odometry(image_folder)
