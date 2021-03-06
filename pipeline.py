import numpy as np
import cv2
from binaryTransform import mag_thresh, dir_sobel_thresh
from helpers import *
import matplotlib.pyplot as plt

# normalize the _lower_ part (containing the street) to the range 0...255
def normstreet(channel):
    maxval=np.max(channel[420:,:])
    return 255*(channel/maxval)

def binarypipeline(img):
    hls=bgr_hls(img)
    hsv=bgr_hsv(img)
    # combute binary image based on color threshold on S channel
    hls_bin = np.zeros_like(hls[:,:,2])
    hls_bin[(normstreet(hls[:,:,2]) >= 110) & (normstreet(hls[:,:,2]) <= 240)] = 1
    
    #b_s, s=dirabs_threshold(hls[:,:,2], sobel_kernel=5, thresh=(0.9, 1.1))
    #b_v, v=dirabs_threshold(hsv[:,:,2], sobel_kernel=5, thresh=(0.9, 1.1))
    
    b = np.zeros_like(hls_bin)
    #b[(b_s==1) & (b_v==1)]=1
    
    b_s, s=dir_sobel_thresh(hls[:,:,2], sobel_kernel=11, alpha=0, thresh=(40, 255)) #np.arctan(-400/300)
    b_v, s=dir_sobel_thresh(hsv[:,:,2], sobel_kernel=11, alpha=0, thresh=(30, 255))
    #b_s, s=mag_thresh(hls[:,:,2], sobel_kernel=11, thresh=(80, 255), nrm=normstreet)
    #b_v, s=mag_thresh(hsv[:,:,2], sobel_kernel=11, thresh=(60, 255), nrm=normstreet)
    b[(b==1) | (b_s==1) | (b_v==1)] = 1
    
    b[(b==1) | (hls_bin==1)] = 1
    
    return b

# factory for perspective transform from src to dst
def srcdst():
    h=720
    src=np.float32([[596, 450], [686, 450], [1027, h-50], [276, h-50]])
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

# findLanes_windowed() as described in main.ipynb, section "Identifying lane lines"
def findLanes_windowed(warped, sigma=20, nwindows = 9, margin = 100, minpix = 200, ym_per_pix = 3/50, xm_per_pix = 3.7/700):
    # take a histogram of the lower half of the image:
    histogram = np.sum(warped[360:,:], axis=0)
    
    # apply some smoothing. I use a convolution with
    # Gaussian kernel
    s=sigma
    n=np.array(range(3*s), dtype=np.double)
    kernel=np.exp(-((n-1.5*s)/(2*s))**2)
    norm=sum(kernel)
    hc=np.convolve(histogram, np.array(kernel, dtype=np.double)/norm, mode='same')

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(hc.shape[0]/2)
    leftx_base = np.argmax(hc[:midpoint])
    rightx_base = np.argmax(hc[midpoint:]) + midpoint
    
    # Set height of windows
    window_height = np.int(warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = warped.shape[0] - (window+1)*window_height
        win_y_high = warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    # Define conversions in x and y from pixels space to meters
    # as function arguments
    # ym_per_pix = 30/720 # meters per pixel in y dimension
    # xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    # Fit a second order polynomial to each, in meters
    left_fit = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)

    return nonzerox, nonzeroy, left_lane_inds, right_lane_inds, left_fit, right_fit

def findLanes_reuse(warped, left_fit, right_fit, margin = 100, ym_per_pix = 3/50, xm_per_pix = 3.7/700):
    def marginCalc(fit, margin):
        #m=np.ones_like(fit)*margin*xm_per_pix
        #z=np.zeros_like(fit)
        #m=np.maximum(np.minimum(1280*xm_per_pix-fit, m), z)
        #m=np.maximum(np.minimum(fit, m), z)
        return margin*xm_per_pix
        
    # Assume you now have a new warped binary image 
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    nonzero = warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    nonzeroym = nonzeroy*ym_per_pix
    nonzeroxm = nonzerox*xm_per_pix

    lf=left_fit[0]*(nonzeroym**2) + left_fit[1]*nonzeroym + left_fit[2]
    rf=right_fit[0]*(nonzeroym**2) + right_fit[1]*nonzeroym + right_fit[2]
    lm=marginCalc(lf, margin)
    rm=marginCalc(rf, margin)
    left_lane_inds = ((nonzeroxm > (lf - lm)) & 
                      (nonzeroxm < (lf + lm))) 
    right_lane_inds = ((nonzeroxm > (rf - rm)) & 
                       (nonzeroxm < (rf + rm)))  

    # Again, extract left and right line pixel positions in meters
    leftx = nonzeroxm[left_lane_inds]
    lefty = nonzeroym[left_lane_inds] 
    rightx = nonzeroxm[right_lane_inds]
    righty = nonzeroym[right_lane_inds]
    # Fit a second order polynomial to each, in meters
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    return nonzerox, nonzeroy, left_lane_inds, right_lane_inds, left_fit, right_fit


def drawLanes_warped(warped, nonzerox, nonzeroy, left_lane_inds, right_lane_inds, left_fit, right_fit, ym_per_pix = 3/50, xpix_per_m = 700/3.7, col_line=(0, 255, 255)):
    # ym_per_pix and xpix_per_m define conversions in x and y from pixels space to meters
    # as function arguments
    # ym_per_pix = 30/720 # meters per pixel in y dimension
    # xpix_per_m = 700/3.7 # pixels per meter in x dimension

    # Generate x and y values for plotting
    ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0] )
    left_fitx = (left_fit[0]*(ploty*ym_per_pix)**2 + left_fit[1]*(ploty*ym_per_pix) + left_fit[2])*xpix_per_m
    right_fitx = (right_fit[0]*(ploty*ym_per_pix)**2 + right_fit[1]*(ploty*ym_per_pix) + right_fit[2])*xpix_per_m

    if (len(warped.shape)==2):
        out_img=np.dstack((warped, warped, warped))*255
    else:
        out_img=np.uint8(warped)
    
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    pts = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    out_img=cv2.polylines(out_img, np.int32(pts), 0, col_line, 2)
    pts = np.array([np.transpose(np.vstack([right_fitx, ploty]))])
    out_img=cv2.polylines(out_img, np.int32(pts), 0, col_line, 2)
    
    return out_img


# drawLane3d(): Draw lane marking on an image
# img: image to be drawn on
# left_fit, right_fit: coefficients of 2nd order polynomial which have been 
#     computed for the lane lines on the image, for example using
#     findLanes_windowed()
# unwarpFun: function to perform the inverse perspective transform.
#     can be created using unwarpFactory()
# ym_per_pix, xpix_per_m: unit conversions pixels <=> meters
def drawLane3d(img, left_fit, right_fit, unwarpFun, ym_per_pix = 3/50, xpix_per_m = 700/3.7):
    # Create an image to draw the lines on
    color_warp = np.zeros_like(img).astype(np.uint8)
    
    # y values
    ploty = np.linspace(0, 719, num=720)
    # convert to meters
    plotym = ym_per_pix * ploty
    # evaluate polynomial & convert to pixels
    left_fitxm = left_fit[0]*plotym**2 + left_fit[1]*plotym + left_fit[2]
    left_fitx = xpix_per_m * left_fitxm
    right_fitxm = right_fit[0]*plotym**2 + right_fit[1]*plotym + right_fit[2]
    right_fitx = xpix_per_m * right_fitxm

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix
    # see unwarpFactory(), pipeline.py
    newwarp = unwarpFun(color_warp)
    # Combine the result with the original image
    result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)
    return result







