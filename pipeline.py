import numpy as np
import cv2
from binaryTransform import mag_thresh
from helpers import *

# normalize the _lower_ part (containing the street) to the range 0...255
def normstreet(channel):
    maxval=np.max(channel[420:,:])
    return 255*(channel/maxval)

def binarypipeline(img):
    hls=bgr_hls(img)
    hsv=bgr_hsv(img)
    # combute binary image based on color threshold on S channel
    hls_bin = np.zeros_like(hls[:,:,2])
    hls_bin[(normstreet(hls[:,:,2]) >= 150) & (normstreet(hls[:,:,2]) <= 240)] = 1
    
    #b_s, s=dirabs_threshold(hls[:,:,2], sobel_kernel=5, thresh=(0.9, 1.1))
    #b_v, v=dirabs_threshold(hsv[:,:,2], sobel_kernel=5, thresh=(0.9, 1.1))
    
    b = np.zeros_like(hls_bin)
    #b[(b_s==1) & (b_v==1)]=1
    
    b_s, s=mag_thresh(hls[:,:,2], sobel_kernel=11, thresh=(80, 255), nrm=normstreet)
    b_v, s=mag_thresh(hsv[:,:,2], sobel_kernel=11, thresh=(60, 255), nrm=normstreet)
    b[(b==1) | (b_s==1) | (b_v==1)] = 1
    
    b[(b==1) | (hls_bin==1)] = 1
    
    return b

# factory for perspective transform from src to dst
def srcdst():
    h=720
    src=np.float32([[596, 450], [686, 450], [1105, h-1], [205, h-1]])
    offset=100
    dst = np.float32([[305, offset], [1005, offset], 
                                         [1005, h-1], 
                                         [305, h-1]])
    return src, dst
    
def warpFactory():
    src, dst=srcdst()
    M=cv2.getPerspectiveTransform(src, dst)
    ## Given src and dst points, calculate the perspective transform matrix
    #M = cv2.getPerspectiveTransform(src, dst)
    return lambda x: cv2.warpPerspective(x, M, (x.shape[1], x.shape[0]))

def unwarpFactory():
    src, dst=srcdst()
    M=cv2.getPerspectiveTransform(dst, src)
    ## Given src and dst points, calculate the perspective transform matrix
    #M = cv2.getPerspectiveTransform(src, dst)
    return lambda x: cv2.warpPerspective(x, M, (x.shape[1], x.shape[0]))
    