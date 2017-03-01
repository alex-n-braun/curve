import cv2

def bgr_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
