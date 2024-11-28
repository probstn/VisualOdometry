import cv2
import numpy as np
import matplotlib.pyplot as plt

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

# Real-time visual odometry
def real_time_visual_odometry(camera_index=1):
    # Initialize video capture
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
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

    # Variables to store previous frame
    prev_frame = None

    #sleep
    import time
    time.sleep(2)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Convert to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_frame is not None:
            # Feature matching between consecutive frames
            pts1, pts2, matches = feature_matching(prev_frame, gray_frame, detector)

            # Estimate essential matrix
            E, mask = cv2.findEssentialMat(pts2, pts1, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
            _, R1, t1, mask_pose = cv2.recoverPose(E, pts2, pts1, K)

            # Update pose
            R = R @ R1
            t = t + R @ t1

            # Save trajectory
            trajectory.append(t.ravel())

            # Draw matches
            match_img = cv2.drawMatches(prev_frame, detector.detect(prev_frame, None),
                                        gray_frame, detector.detect(gray_frame, None),
                                        matches, None)
            cv2.imshow("Matches", match_img)

        # Update previous frame
        prev_frame = gray_frame

        # Break loop on key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Visualize trajectory
    if trajectory:
        trajectory = np.array(trajectory)
        plt.plot(trajectory[:, 0], trajectory[:, 2], marker='o')
        plt.title("Visual Odometry Trajectory")
        plt.xlabel("X")
        plt.ylabel("Z")
        plt.show()

    cap.release()
    cv2.destroyAllWindows()

# Run real-time visual odometry
real_time_visual_odometry()
