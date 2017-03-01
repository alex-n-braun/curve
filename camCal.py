import numpy as np
import cv2
import glob
from helpers import bgr_rgb
#import matplotlib.pyplot as plt

# findCorners(imggen)
# finds the chessboard corners in a list of chessboard images
# and appends the corresponding points to the list imgpoints,
# while adding the 3d coordinates of those points to the list
# objpoints. For illustration purpose, the imgpoints are drawn 
# onto the chessboard images.
# imggen: generator or list of images to be processed.
# return objpoints: 3d coordinates of the chessboard corners
# return imgpoints: 2d coordinates of the chessboard corners 
#   in the images
# return images: chessboard images with marked corners
def findCorners(imggen):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.
    images = []

    # Step through the list and search for chessboard corners
    for img in imggen:
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw the corners on the input image
            img = bgr_rgb(cv2.drawChessboardCorners(img, (9,6), corners, True))
            images.append(img)
            
    return objpoints, imgpoints, images

def camCal(objpoints, imgpoints, shape):
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, shape,None,None)
    return mtx, dist, rvecs, tvecs

def undistortImg(img, mtx, dist):
    return cv2.undistort(img, mtx, dist, None, mtx)

