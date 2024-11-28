import numpy as np
import cv2 as cv
import glob

# chessboard properties
images = glob.glob('calibration_img/*.jpeg')
size = (8, 6)

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((size[0] * size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:size[0], 0:size[1]].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

for fname in images:
    print('Processing: ', fname)
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, size, None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        print('Chessboard detected!')
        objpoints.append(objp)

        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        # cv.drawChessboardCorners(img, size, corners2, ret)
        # cv.imshow('img', img)
        # cv.waitKey(500)
    else:
        print('Chessboard not detected!')

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("Camera Matrix (mtx):\n", mtx)
print("Distortion Coefficients (dist):\n", dist)
print("Rotation Vectors (rvecs):\n", rvecs)
print("Translation Vectors (tvecs):\n", tvecs)

img = cv.imread('./calibration_img/IMG_4399.jpeg')
h, w = img.shape[:2]

# Use a higher value for alpha to avoid clipping
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

# undistort using cv.undistort or cv.remap
# Undistort using remap method
mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)
dst = cv.remap(img, mapx, mapy, cv.INTER_CUBIC)  # Use INTER_CUBIC for smoother results

# Crop the image (using ROI) to remove black border
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]

# Save the undistorted result
cv.imwrite('calibresult.png', dst)

# Display the undistorted image
cv.imshow('Undistorted Image', dst)
cv.waitKey(2500)

cv.destroyAllWindows()
