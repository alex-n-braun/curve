import cv2

# shortcut for bgr to rgb transformation
def bgr_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# shortcut for bgr to gray transformation
def bgr_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
