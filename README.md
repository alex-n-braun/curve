
# Advanced Lane Line Finding
---

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.


## Contents of the Submission
---
The submission containes the following files and folders.

1. `README.md`, along with several `output*.png` files containing the graphics. 
2. `main.ipynb`, the notebook containing the documentation and sources. 
3. `helpers.py`, several shortcuts for color space transforms and some more things
4. `binaryTransform.py`, several functions for converting an image to black an white.
5. `pipeline.py`, the final pipeline for processing an image sequence.
6. `project_video_out.mp3`, the processed video.

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
---
Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  


## Writeup / README
---

* Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it! Instead of writing a separate readme, I include everything relevant into this jupyter notebook. This reduces effort & redundancy.


## Camera Calibration
---

### Find Chessboard Corners

* Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in `./camCal.py`, function `findCorners()`, and was derived from `./examples/example.ipynb`.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  


```python
import numpy as np
import glob
import cv2
from camCal import findCorners

# Make a list of calibration images
imageFiles = sorted(glob.glob('./camera_cal/calibration*.jpg'))

imggen = (cv2.imread(fn) for fn in imageFiles)

objpoints, imgpoints, images = findCorners(imggen)

```


```python
import matplotlib.pyplot as plt
%matplotlib inline

# show samples
I=(1, 2, 4, 10, 15, 16)
plt.figure(figsize=(14,6))
for i in range(6):
    plt.subplot(2, 3, 1+i)
    plt.imshow(images[I[i]])
#plt.subplot(2, 3, 2)
#plt.imshow(images[4])
#plt.subplot(2, 3, 3)
#plt.imshow(images[10])
#plt.subplot(2, 3, 4)
#plt.imshow(images[15])
#plt.subplot(2, 3, 5)
#plt.imshow(images[1])
#plt.subplot(2, 3, 6)
#plt.imshow(images[16])

```


![png](output_7_0.png)


### Camera Calibration and Distortion Coefficients
I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained the following result.

The code for this step is contained in ./camCal.py, functions camCal() and undistortImg().



```python
from camCal import camCal, undistortImg
%matplotlib inline

img=images[0]
shape=img.shape[::-1][1:]
mtx, dist, rvecs, tvecs=camCal(objpoints, imgpoints, shape)

image=cv2.imread('./camera_cal/calibration1.jpg')

# show samples
plt.figure(figsize=(14,6))
plt.subplot(1, 2, 1)
plt.text(100, -50, 'Original')
plt.imshow(image)
plt.subplot(1, 2, 2)
plt.text(100, -50, 'Undistorted')
plt.imshow(undistortImg(image, mtx, dist))


```




    <matplotlib.image.AxesImage at 0x7ffa09aa1940>




![png](output_9_1.png)


## Pipeline (test-images)
---
Read in all files from test_images folder


```python
import numpy as np
import glob
import cv2
from camCal import findCorners

# Make a list of calibration images
testImageFiles = sorted(glob.glob('./test_images/*.jpg'))

testImages = [cv2.imread(fn) for fn in testImageFiles]

```

### Distortion Correction
- Provide an example of a distortion-corrected image.

In the following, I apply distortion correction to all sample images provided in the test_images folder. Finally I show three samples.


```python
from helpers import bgr_rgb

undistTestImages=[undistortImg(img, mtx, dist) for img in testImages];

# show samples
plt.figure(figsize=(14,5.5))
I=(0, 3, 6)
plt.subplot(2, 3, 1)
plt.text(-200, 250, 'Original', rotation=90)
plt.subplot(2, 3, 4)
plt.text(-200, 250, 'Undistorted', rotation=90)
for i in range(0,3):
    plt.subplot(2, 3, i+1)
    
    plt.imshow(bgr_rgb(testImages[I[i]]))
    plt.text(100, -50, testImageFiles[I[i]])

    plt.subplot(2, 3, i+4)
    plt.imshow(bgr_rgb(undistTestImages[I[i]]))

```


![png](output_13_0.png)


### Creating a Thresholded Binary Image
- Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image. Provide an example of a binary image result.

#### Gradients
Computing the gradient in x or y direction does not account for the diagonal directions of the left/right lanes. Therefore I define the directed gradient, which computes the gradient in direction of the angle alpha.


```python
from helpers import bgr_gray
from binaryTransform import dir_sobel_thresh

# show samples
I=(0, 6)
plt.figure(figsize=(14,5.5))
for i in (0,1):
    plt.subplot(2, 3, 1+i*3)
    plt.imshow(bgr_rgb(undistTestImages[I[i]]))
    b, s=dir_sobel_thresh(bgr_gray(undistTestImages[I[i]]), sobel_kernel=7, alpha=np.arctan(400/300), thresh=(125, 255))
    plt.subplot(2, 3, 2+i*3)
    plt.imshow(s, cmap='gray')
    plt.subplot(2, 3, 3+i*3)
    plt.imshow(b, cmap='gray')
```


![png](output_16_0.png)


In the first row we can clearly see that the left lane is nicely identified. The gradient is computed in the direction $\alpha=\arctan(400/300)$, where $400/300$ is the approximate slope of the left lane. The right lane can be identified by applying the angle $\alpha=\arctan(-400/300)$. However, we see poor performance in the second row due to poor contrast. Therefore I want to try color space transforms. But first, some more examples applying 'magnitude of the gradient' and 'direction of the gradient'.


```python
from helpers import bgr_gray
from binaryTransform import mag_thresh

# show samples
I=(0, 6)
plt.figure(figsize=(14,5.5))
for i in (0,1):
    plt.subplot(2, 3, 1+i*3)
    plt.imshow(bgr_rgb(undistTestImages[I[i]]))
    b, s=mag_thresh(bgr_gray(undistTestImages[I[i]]), sobel_kernel=7, thresh=(100, 255))
    plt.subplot(2, 3, 2+i*3)
    plt.imshow(s, cmap='gray')
    plt.subplot(2, 3, 3+i*3)
    plt.imshow(b, cmap='gray')
```


![png](output_18_0.png)



```python
from helpers import bgr_gray
from binaryTransform import dir_threshold

# show samples
I=(0, 6)
plt.figure(figsize=(14,5.5))
for i in (0,1):
    plt.subplot(2, 3, 1+i*3)
    plt.imshow(bgr_rgb(undistTestImages[I[i]]))
    b, s=dir_threshold(bgr_gray(undistTestImages[I[i]]), sobel_kernel=15, thresh=(0.9*np.arctan(420/300), 1.1*np.arctan(420/200)))
    plt.subplot(2, 3, 2+i*3)
    plt.imshow(s, cmap='gray')
    plt.subplot(2, 3, 3+i*3)
    plt.imshow(b, cmap='gray')
```


![png](output_19_0.png)



```python
from helpers import bgr_gray
from binaryTransform import dirabs_threshold

# show samples
I=(0, 6)
plt.figure(figsize=(14,5.5))
for i in (0,1):
    plt.subplot(2, 3, 1+i*3)
    plt.imshow(bgr_rgb(undistTestImages[I[i]]))
    b, s=dirabs_threshold(bgr_gray(undistTestImages[I[i]]),  sobel_kernel=25, thresh=(0.9, 1.1))
    plt.subplot(2, 3, 2+i*3)
    plt.imshow(s, cmap='gray')
    plt.subplot(2, 3, 3+i*3)
    plt.imshow(b, cmap='gray')
```


![png](output_20_0.png)


#### Color Space Transforms
As discussed in the course, by just transforming to gray scale, much information may be lost. Instead, here I use color space transfomations in order to preserve valuable information as value, saturation or lightness. The next image sequence displays

1. original image
2. grayscale image
3. saturation channel of the hls transform
4. value channel of the hsv transform

of three selected example images.


```python
from helpers import *

I=(0, 5, 6)
plt.figure(figsize=(14,11.5))
for i in (0,1,2):
    plt.subplot(4, 3, 1+i)
    plt.imshow(bgr_rgb(undistTestImages[I[i]]))
    gray=bgr_gray(undistTestImages[I[i]])
    plt.subplot(4, 3, 4+i)
    plt.imshow(gray, cmap='gray')
    hls=bgr_hls(undistTestImages[I[i]])
    plt.subplot(4, 3, 7+i)
    plt.imshow(hls[:,:,2], cmap='gray')
    hsv=bgr_hsv(undistTestImages[I[i]])
    plt.subplot(4, 3, 10+i)
    plt.imshow(hsv[:,:,2], cmap='gray')
```


![png](output_22_0.png)


The lightness channel yields similar results as the grayscale image, both not very satisfactory. We therefore have a look at the thresholded binaries of S and V channel only:


```python
from helpers import *

I=(0, 5, 6)
plt.figure(figsize=(14,9))
for i in (0,1,2):
    plt.subplot(3, 3, 1+i)
    plt.imshow(bgr_rgb(undistTestImages[I[i]]))
    
    hls=bgr_hls(undistTestImages[I[i]])
    hls_bin = np.zeros_like(hls[:,:,2])
    hls_bin[(hls[:,:,2] >= 110) & (hls[:,:,2] <= 240)] = 1
    plt.subplot(3, 3, 4+i)
    plt.imshow(hls_bin, cmap='gray')
    
    hsv=bgr_hsv(undistTestImages[I[i]])
    hsv_bin = np.zeros_like(hls[:,:,2])
    hsv_bin[(hsv[:,:,2] >= 150) & (hsv[:,:,2] <= 255)] = 1
    plt.subplot(3, 3, 7+i)
    plt.imshow(hsv_bin, cmap='gray')
```


![png](output_24_0.png)


Now let's have a look at the gradients of S and V channel.


```python
from helpers import *

I=(0, 5, 6)
plt.figure(figsize=(14,15))
for i in (0,1,2):
    plt.subplot(5, 3, 1+i)
    plt.imshow(bgr_rgb(undistTestImages[I[i]]))
    
    hls=bgr_hls(undistTestImages[I[i]])
    b_hls, s=dir_sobel_thresh(hls[:,:,2], sobel_kernel=11, alpha=0, thresh=(40, 255)) #np.arctan(-400/300)
    #b_hls, s=mag_thresh(hls[:,:,2], sobel_kernel=11, thresh=(80, 255))
    #b_hls, s=dirabs_threshold(hls[:,:,2], sobel_kernel=15, thresh=(0.9, 1.1))
    plt.subplot(5, 3, 4+i)
    plt.imshow(s, cmap='gray')
    plt.subplot(5, 3, 7+i)
    plt.imshow(b_hls, cmap='gray')
    
    hsv=bgr_hsv(undistTestImages[I[i]])
    b_hsv, s=dir_sobel_thresh(hsv[:,:,2], sobel_kernel=11, alpha=0, thresh=(30, 255))
    #b_hsv, s=mag_thresh(hsv[:,:,2], sobel_kernel=11, thresh=(60, 255))
    #b_hsv, s=dirabs_threshold(hsv[:,:,2], sobel_kernel=15, thresh=(0.9, 1.1))
    plt.subplot(5, 3, 10+i)
    plt.imshow(s, cmap='gray')
    plt.subplot(5, 3, 13+i)
    plt.imshow(b_hsv, cmap='gray')
    
```


![png](output_26_0.png)


#### Final Result for creating a Binary Image
Finally a combination of color threshold on S channel and thresholded gradients on S and V channel is computed using the function `binarypipeline(img)` privided in `pipeline.py`. The final set of parameters like thresholds, as well as the final set of operations that are to be performed, was chosen by trial and error.


```python
from helpers import *
from pipeline import binarypipeline

# show samples
I=(0, 5, 6)
plt.figure(figsize=(14,12))
for i in (0,1,2):
    plt.subplot(3, 2, 1+i*2)
    plt.imshow(bgr_rgb(undistTestImages[I[i]]))
    s=binarypipeline(undistTestImages[I[i]])
    plt.subplot(3, 2, 2+i*2)
    plt.imshow(s, cmap='gray')

```


![png](output_28_0.png)


### Applying Perspective Transform
- Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

I perform the following steps:

1. Choose an image with straight lane lines. Choose four points in a trapezoidal shape (see `srcdst()` in `pipeline.py`)
2. Compute the perspective transformation matrix using the opencv function getPerspectiveTransform (see `warpFactory()`, `unwarpFactory()` in `pipeline.py`)
3. Create sample images applying the perspective transform.


```python
from pipeline import warpFactory, srcdst

src, dst=srcdst()
warpFun=warpFactory()

I=(0, 5, 6)
plt.figure(figsize=(14,12))
for i in (0,1,2):
    plt.subplot(3, 2, 1+i*2)
    rgb=bgr_rgb(undistTestImages[I[i]])
    rgb_poly=np.array(rgb)
    cv2.polylines(rgb_poly, np.int_([src]), 1, (255,0,0), 2)
    plt.imshow(rgb_poly)
    warped = warpFun(rgb)
    cv2.polylines(warped, np.int_([dst]), 1, (255,0,0), 2)
    plt.subplot(3, 2, 2+i*2)
    plt.imshow(warped)


```


![png](output_30_0.png)



```python
dst
```




    array([[  305.,   100.],
           [ 1005.,   100.],
           [ 1005.,   719.],
           [  305.,   719.]], dtype=float32)



The source and destination points are hardcoded in `srcdst()`, `pipeline.py`, with the following values:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 596, 450      | 305, 100      | 
| 686, 450      | 1105, 100     |
| 1105, 719     | 1105, 719     |
| 205,719       | 305, 719      |

By the way, here we see why we should perform the binary transform _before_ applying the perspective transform: gradients get strongly smoothened by the perspective transform. Application of the perspective transform on the binary images yields:


```python
from pipeline import warpFactory, srcdst

src, dst=srcdst()
warpFun=warpFactory()

I=(0, 5, 6)
plt.figure(figsize=(14,12))
for i in (0,1,2):
    plt.subplot(3, 2, 1+i*2)
    rgb=bgr_rgb(undistTestImages[I[i]])
    rgb_poly=np.array(rgb)
    cv2.polylines(rgb_poly, np.int_([src]), 1, (255,0,0), 2)
    plt.imshow(rgb_poly)
    b=binarypipeline(undistTestImages[I[i]])
    warped = warpFun(b)
    #cv2.polylines(warped, np.int_([dst]), 1, (255,0,0), 2)
    plt.subplot(3, 2, 2+i*2)
    plt.imshow(warped, cmap='gray')


```


![png](output_33_0.png)


### Identifying lane lines
- Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial

For further developing the pipeline, I use `straight_lines1.jpg`. I take the following steps, that are described in the lesson. The code is in large parts taken from the lessen.

1. Find the two lanes using a histogram. 
2. Apply windowing to distinguish non-zero lane pixels from non-zero other pixels. Add the pixel positions to a list.
3. Perform a second order polynomial fit to get lane slope and curvature.

Since a lane may be split into two (most probably because of the gradients on the left and right side), I additionaly apply a smoothing based on a convolution, using a Gaussian kernel.


```python
from pipeline import warpFactory

warpFun=warpFactory()
img=undistTestImages[5]
b=binarypipeline(img)
warped = warpFun(b)
plt.imshow(warped, cmap='gray')

# take a histogram of the lower half of the image:
histogram = np.sum(warped[360:,:], axis=0)
plt.plot(histogram)
# oups, there are two left and two right lanes... 
# apply some smoothing. I use a convolution with
# Gaussian kernel
s=20
n=np.array(range(3*s), dtype=np.double)
kernel=np.exp(-((n-1.5*s)/(2*s))**2)
norm=sum(kernel)
hc=np.convolve(histogram, np.array(kernel, dtype=np.double)/norm, mode='same')
plt.plot(hc)

```




    [<matplotlib.lines.Line2D at 0x7ffa09c709b0>]




![png](output_35_1.png)


Next, I find the positions of the lanes by looking for the maximum values in the histogram in the left and the right half.


```python
# Find the peak of the left and right halves of the histogram
# These will be the starting point for the left and right lines
midpoint = np.int(hc.shape[0]/2)
leftx_base = np.argmax(hc[:midpoint])
rightx_base = np.argmax(hc[midpoint:]) + midpoint

```

Preparation for iterating over the image using windows:


```python
# Choose the number of sliding windows
nwindows = 9
# Set height of windows
window_height = np.int(warped.shape[0]/nwindows)
# Identify the x and y positions of all nonzero pixels in the image
nonzero = warped.nonzero()
nonzeroy = np.array(nonzero[0])
nonzerox = np.array(nonzero[1])
# Current positions to be updated for each window
leftx_current = leftx_base
rightx_current = rightx_base
# Set the width of the windows +/- margin
margin = 100
# Set minimum number of pixels found to recenter window
minpix = 200
# Create empty lists to receive left and right lane pixel indices
left_lane_inds = []
right_lane_inds = []

```

Now iterating. Draw the windows on an output image for illustration; this will not be present in the final pipeline. The loop iteratively recenters the windows on the lanes and identifies the relevant pixels inside the windows, appending them to index lists `left_lane_inds` and `right_lane_inds` containing the lane pixels of the whole warped image.


```python
# Create an output image to draw on and  visualize the result
out_img = np.dstack((warped, warped, warped))*255
# Step through the windows one by one
for window in range(nwindows):
    # Identify window boundaries in x and y (and right and left)
    win_y_low = warped.shape[0] - (window+1)*window_height
    win_y_high = warped.shape[0] - window*window_height
    win_xleft_low = leftx_current - margin
    win_xleft_high = leftx_current + margin
    win_xright_low = rightx_current - margin
    win_xright_high = rightx_current + margin
    # Draw the windows on the visualization image
    cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
    cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
    # Identify the nonzero pixels in x and y within the window
    good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
    good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
    # Append these indices to the lists
    left_lane_inds.append(good_left_inds)
    right_lane_inds.append(good_right_inds)
    # If you found > minpix pixels, recenter next window on their mean position
    if len(good_left_inds) > minpix:
        leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
    if len(good_right_inds) > minpix:        
        rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

plt.imshow(out_img)

# Concatenate the arrays of indices
left_lane_inds = np.concatenate(left_lane_inds)
right_lane_inds = np.concatenate(right_lane_inds)

```


![png](output_41_0.png)


Finally get the x and y positions of the nonzero pixels identified by the index lists, and perform a second order polynomial fit. This then allows to extract lane curvature and slope, which will be the next step.


```python
# Extract left and right line pixel positions
leftx = nonzerox[left_lane_inds]
lefty = nonzeroy[left_lane_inds] 
rightx = nonzerox[right_lane_inds]
righty = nonzeroy[right_lane_inds] 

# Fit a second order polynomial to each
left_fit = np.polyfit(lefty, leftx, 2)
right_fit = np.polyfit(righty, rightx, 2)

```

Now I use the code as described in the lecture to draw the lane pixels and the polynomials on the warped image:


```python
# Generate x and y values for plotting
ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0] )
left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
plt.imshow(out_img)
plt.plot(left_fitx, ploty, color='yellow')
plt.plot(right_fitx, ploty, color='yellow')
plt.xlim(0, 1280)
plt.ylim(720, 0)


```




    (720, 0)




![png](output_45_1.png)


That's it for lane finding. The above code is contained in `pipeline.py`, `findLanes_windowed()` (without illustration), and the illustration of the lane finding in `drawLanes_warped()` (without the windows being drawn). There is one more lane line finding function, `findLanes_reuse()`, which makes use of the polynomial coefficients from the previous step, as described in the lecture. The coefficients for the second order polynomial are computed here with a unit conversion from pixels to meters, which we will need lateron for the computation of curvature in meters. Finally I use this code to identify the lanes in three example images:


```python
from pipeline import findLanes_windowed, findLanes_reuse, drawLanes_warped

warpFun=warpFactory()
I=(0, 5, 6)
plt.figure(figsize=(14,12))
for i in (0,1,2):
    plt.subplot(3, 2, 1+i*2)
    rgb=bgr_rgb(undistTestImages[I[i]])
    plt.imshow(rgb)
    b=binarypipeline(undistTestImages[I[i]])
    warped = warpFun(b)
    nonzerox, nonzeroy, left_lane_inds, right_lane_inds, left_fit, right_fit = findLanes_windowed(warped, minpix=200)
    img=drawLanes_warped(warped, nonzerox, nonzeroy, left_lane_inds, right_lane_inds, left_fit, right_fit)
    nonzerox, nonzeroy, left_lane_inds, right_lane_inds, left_fit, right_fit = findLanes_reuse(warped, left_fit, right_fit)
    img=drawLanes_warped(img, nonzerox, nonzeroy, left_lane_inds, right_lane_inds, left_fit, right_fit, col_line=(0, 255, 0))
    plt.subplot(3, 2, 2+i*2)
    plt.imshow(bgr_rgb(img))
    


```


![png](output_47_0.png)


For the third example, stable detection of the right lane line with the given pipeline is not possible. However, reusing the polynomials from the last video frame using `findLanes_reuse()` will improve the situation, as will be demonstrated later.

### Computing Curvature Radius, Vehicle Position, and some more numbers

- Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

Now we already have the polynomial coefficients $A$, $B$ and $C$ returned from the function call `findLanes_windowed()` in the correct units. The $x$ position of the lane is given by

$L(y)=Ay^2+By+C$.

This allows for the direct computation of the lane curvature radius $R$ in meters. The curvature is evaluated at the position of the car, hence `y_eval=720`, for the three example images above.

\begin{equation}
R^\pm = \frac{\left[1+L'(y_{eval})^2\right]^{3/2}}{L''(y_{eval})} = \frac{\left[1+(2Ay_{eval}+B)^2\right]^{3/2}}{2A}, ~~~~~ R=\vert R^\pm \vert
\end{equation}

The values for the unit conversion are chosen as

- 3.7m per 700 pixels in x direction (default value from the lesson, matches very nicely with the straight lanes image)
- 3m per 50 pixels in y direction (estimated from looking at the warped color image of the straight lane, the dashed lane lines are assumed to be 3 meters long).

$R^\pm$ not only  gives the curvature radius, but by means of its sign also the direction.T

In addition, I compute the position $P$ of the vehicle in meters, relative to the lane center. As specified in the rubric, I have to look for the difference of the midpoint of the lane from the center $w/2$ of the image, where $w$ is its width. Therefore I evaluate the polynomials at the position of the car

\begin{equation}
P=\frac w 2 - \frac 1 2 (L_{left}(y_{eval}) + L_{right}(y_{eval})).
\end{equation}



```python
from pipeline import findLanes_windowed

warpFun=warpFactory()
ym_per_pix = 3/50
y_eval=720
y_m=y_eval*ym_per_pix
xm_per_pix = 3.7/700
x_eval=undistTestImages[I[i]].shape[1]/2
x_m=x_eval*xm_per_pix

I=(0, 5, 6)
for i in (0,1,2):
    b=binarypipeline(undistTestImages[I[i]])
    warped = warpFun(b)
    nonzerox, nonzeroy, left_lane_inds, right_lane_inds, left_fit, right_fit = findLanes_windowed(
        warped, ym_per_pix = 3/50, xm_per_pix = 3.7/700)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit[0]*y_m + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_m + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    # Calculate the position of the car. First: position of the two lanes, midpoint:
    pos_left = left_fit[0]*y_m**2 + left_fit[1]*y_m + left_fit[2]
    pos_right = right_fit[0]*y_m**2 + right_fit[1]*y_m + right_fit[2]
    pos_mid = 0.5 * (pos_right + pos_left)
    # difference to center of the image
    pos_car = x_m - pos_mid
    # Now our radius of curvature is in meters
    print('Radius: ', left_curverad, 'm', right_curverad, 'm; Pos. Car: ', pos_car, 'm')

```

    Radius:  15060.0573927 m 110098.844482 m; Pos. Car:  -0.0956835583676 m
    Radius:  1328.98421934 m 1075.94044105 m; Pos. Car:  -0.330563058334 m
    Radius:  449.762353333 m 522.42801703 m; Pos. Car:  -0.101894435509 m


The values for the curvature radius look reasonable; for the straight lanes it could even be infinite, the second one with curvature is in a reasonable range (compare [U.S. government specifications for highway curvature](http://onlinemanuals.txdot.gov/txdotmanuals/rdw/horizontal_alignment.htm#BGBHGEGC)), for the third one the computed radius is too small, which fits with the not so nice performance of the lane detection.

Just for completeness: Distance $D$ of the two lanes at the position of the car is

\begin{equation}
D = L_{right}(y_{eval})-L_{left}(y_{eval}), 
\end{equation}

and we can compute a rotation angle $\alpha$ of the lane against the vertical by first computing the slope of the lane function at the position of the car

$L'(y_{eval})=2Ay_{eval}+B$,

then

\begin{equation}
\alpha = \arcsin(2Ay_{eval}+B).
\end{equation}

We now have 6 numbers: lane distance $D$, car position $P$ relative to the midpoint of the lane, left and right lane curvature $R^\pm_{left}$, $R^\pm_{right}$, and left and right angle $\alpha_{left}$ and $\alpha_{right}$. These numbers can be used to completely determine the two 2nd order polynomials describing the lanes:

\begin{eqnarray}
w-2P&=& L_{right}(y_{eval}) + L_{left}(y_{eval}), \\
D &=& L_{right}(y_{eval})-L_{left}(y_{eval}) \\
\Rightarrow ~~~~~ 
L_{right}(y_{eval}) &=& \frac 1 2 (w-2P+D) = A_{right}y_{eval}^2 + B_{right} y_{eval} + C_{right}, \\
L_{left}(y_{eval}) &=& \frac 1 2 (w-2P-D) = A_{left}y_{eval}^2 + B_{left} y_{eval} + C_{left};
\end{eqnarray}

furthermore, from angles $\alpha$ we can derive

\begin{eqnarray}
L'_{right}(y_{eval}) &=& \sin(\alpha_{right}) = 2A_{right} y_{eval} + B_{right}, \\
L'_{left}(y_{eval}) &=& \sin(\alpha_{left}) = 2A_{left} y_{eval} + B_{left}.
\end{eqnarray}

Finally, we get the second derivative of the lane functions as

\begin{eqnarray}
L''_{right}(y_{eval}) &=& \frac{\left[1+L_{right}'(y_{eval})^2\right]^{3/2}}{R_{right}^\pm}
  = \frac{\left[1+\sin(\alpha_{right})^2\right]^{3/2}}{R_{right}^\pm} = 2A_{right}, \\
L''_{left}(y_{eval}) &=& \frac{\left[1+L_{left}'(y_{eval})^2\right]^{3/2}}{R_{left}^\pm}
  = \frac{\left[1+\sin(\alpha_{left})^2\right]^{3/2}}{R_{left}^\pm} = 2A_{left}.
\end{eqnarray}

Hence,

\begin{eqnarray}
  A &=& \frac{\left[1+\sin(\alpha)^2\right]^{3/2}}{2R^\pm}, ~~ B = \sin(\alpha) - 2A y_{eval},
\end{eqnarray}

for left and right, and
\begin{eqnarray}
  C_{right} &=& \frac 1 2 (w-2P+D) - A_{right}y_{eval}^2 - B_{right} y_{eval}, \\
  C_{left}  &=& \frac 1 2 (w-2P-D) - A_{left}y_{eval}^2 - B_{left} y_{eval}.
\end{eqnarray}

Now, what is the profit of this forward and backward transformation between polynomial coefficients on the one hand and lane distance, car position, angles and curvature radii on the other hand? Well, I dont know yet... at least I have some intuition about the latter, which I do not have that much for the coefficients. There is the idea that for analyzing the video, it might help to have some intuition about the numbers. Neither the angle $\alpha$, nor the radius $R$ should jump, they should be comparable for the left and the right lane line, the distance of the lane lines should be more or less constant, and the car's position relative to the lane lines should be somewhat related to the angle $\alpha$ and to the car's speed... oups, we do not know the latter. And the steering angle or the turning radius of the car would also help. 

Python functions for performing the transformations coefficients <=> parameters can be found in `helpers.py`, called `toParam()` and `toCoeff()`.

### Visualization

- Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

Again I take a code snippet I found in the lessons material. Drawing the lane marking on the image is done in the function `drawLane3d()` in `pipeline.py`.


```python
from pipeline import findLanes_windowed, drawLanes_warped, warpFactory, unwarpFactory, drawLane3d

warpFun = warpFactory()
unwarpFun = unwarpFactory()

I=(0, 5, 6)
plt.figure(figsize=(14,12))
for i in (0,1,2):
    plt.subplot(3, 2, 1+i*2)
    rgb=bgr_rgb(undistTestImages[I[i]])
    plt.imshow(rgb)
    b=binarypipeline(undistTestImages[I[i]])
    warped = warpFun(b)
    nonzerox, nonzeroy, left_lane_inds, right_lane_inds, left_fit, right_fit = findLanes_windowed(warped)
    img = drawLane3d(undistTestImages[I[i]], left_fit, right_fit, unwarpFun)
    plt.subplot(3, 2, 2+i*2)

    plt.imshow(bgr_rgb(img))
    

```


![png](output_53_0.png)


## Pipeline (Video)
---
- Provide a link to your final video output. Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!)

Before processing the video, I define a class to store the properties of the lane lines. Internally, I use lane distance, relative position of the car, left/right curvature radius, left/right angle alpha. The class permits various parameter transformations, including back to the polynomial coefficients (`toCoeff()`). There is as seconde `toCoeff2()`, which transforms to polynomial coefficients of a polynomial with reversed y coordinate (0 is at the bottom of the image).


```python
from helpers import toParam, toCoeff

# Define a class to receive the characteristics of each line detection
class LaneLines():
    def __init__(self, coeff=None, coeff2=None, params=None):
        if coeff2:
            lc2=coeff2[0]; rc2=coeff2[1]
            lc=[lc2[0], -(2*lc2[0]*720*3/50+lc2[1]), lc2[0]*(720*3/50)**2+lc2[1]*720*3/50+lc2[2]]
            rc=[rc2[0], -(2*rc2[0]*720*3/50+rc2[1]), rc2[0]*(720*3/50)**2+rc2[1]*720*3/50+rc2[2]]
            coeff=[lc, rc]
        if coeff:
            pos_car, dist_lanes, alpha_left, alpha_right, left_curverad_pm, right_curverad_pm = toParam(coeff[0], coeff[1])
            self.pos_car=pos_car
            self.dist_lanes=dist_lanes
            self.alpha_left=alpha_left
            self.alpha_right=alpha_right
            self.curverad_left=left_curverad_pm
            self.curverad_right=right_curverad_pm
        elif params:
            self.pos_car=params[0]
            self.dist_lanes=params[1]
            self.alpha_left=params[2]
            self.alpha_right=params[3]
            self.curverad_left=params[4]
            self.curverad_right=params[5]
        else:
            self.pos_car=None
            self.dist_lanes=None
            self.alpha_left=None
            self.alpha_right=None
            self.curverad_left=None
            self.curverad_right=None
            
    def toParam(self):
        return self.pos_car, self.dist_lanes, self.alpha_left, self.alpha_right, self.curverad_left, self.curverad_right
    
    def toCoeff(self):
        return toCoeff(self.pos_car, self.dist_lanes, self.alpha_left, self.alpha_right, self.curverad_left, self.curverad_right)
    
    def toCoeff2(self):
        lc, rc=self.toCoeff()
        lc2=[lc[0], -(2*lc[0]*720*3/50+lc[1]), lc[0]*(720*3/50)**2+lc[1]*720*3/50+lc[2]]
        rc2=[rc[0], -(2*rc[0]*720*3/50+rc[1]), rc[0]*(720*3/50)**2+rc[1]*720*3/50+rc[2]]
        return lc2, rc2
        
    def plausibleNew(self, newLaneLines):
        new_coeff2=np.array(newLaneLines.toCoeff2())
        nlc2=new_coeff2[0]; nrc2=new_coeff2[1]
        old_coeff2=np.array(self.toCoeff2())
        olc2=old_coeff2[0]; orc2=old_coeff2[1]
        
        sigma2=0.1; sigma0=0.001; sigma_alpha=np.pi/6
        wl=np.exp(-0.5*((nlc2[2]-olc2[2])/sigma2)**2) * np.exp(
            -0.5*(nlc2[0]/sigma0)**2) * np.exp( -0.5*(newLaneLines.alpha_left/sigma_alpha)**2 )
        wr=np.exp(-0.5*((nrc2[2]-orc2[2])/sigma2)**2) * np.exp(
            -0.5*(nrc2[0]/sigma0)**2) * np.exp( -0.5*(newLaneLines.alpha_right/sigma_alpha)**2 )
        
        wl=min(0.25, wl); wr=min(0.25, wr)
        plc2=wl*nlc2+(1-wl)*olc2
        prc2=wr*nrc2+(1-wr)*orc2
        plaus_coeff2=np.array([plc2, prc2])
        plausLaneLines=LaneLines(coeff2=plaus_coeff2.tolist())
            
#        if (unplaus_left+unplaus_right>1):
#            print("-----------------------------------------------")
#        print(plausLaneLines.toCoeff())
            
        return plausLaneLines
        
```


```python
b=binarypipeline(undistTestImages[5])
warped = warpFun(b)
nonzerox, nonzeroy, left_lane_inds, right_lane_inds, left_fit5, right_fit5 = findLanes_windowed(warped)
b=binarypipeline(undistTestImages[6])
warped = warpFun(b)
nonzerox, nonzeroy, left_lane_inds, right_lane_inds, left_fit6, right_fit6 = findLanes_windowed(warped)

ll5 = LaneLines(coeff=[left_fit5, right_fit5])
ll6 = LaneLines(coeff=[left_fit6, right_fit6])
llProb = ll5.plausibleNew(ll6)
ll5.__dict__, ll6.__dict__, llProb.__dict__

#lc, rc=llProb.toCoeff()
#img = drawLane3d(undistTestImages[5], lc, rc, unwarpFun)
#plt.imshow(bgr_rgb(img))

```




    ({'alpha_left': -0.00030284136507442852,
      'alpha_right': -0.0093455199692488843,
      'curverad_left': 1328.9842193393802,
      'curverad_right': 1075.9404410527061,
      'dist_lanes': 3.7118800534738692,
      'pos_car': -0.33056305833376998},
     {'alpha_left': 0.010493880939986272,
      'alpha_right': -0.024028924218544353,
      'curverad_left': 449.76235333291095,
      'curverad_right': -522.42801702953955,
      'dist_lanes': 3.9470927311100761,
      'pos_car': -0.10189443550923194},
     {'alpha_left': -0.00028835335836160968,
      'alpha_right': -0.013016058457576024,
      'curverad_left': 1325.5061996127909,
      'curverad_right': 4583.975931287128,
      'dist_lanes': 3.6845791534256525,
      'pos_car': -0.31644793735626964})




```python
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip

class ProcessImage():
    def __init__(self, text_file):
        self.oldLaneLines=None
        self.text_file=text_file
    def __call__(self, image):
        # NOTE: The output you return should be a color image (3 channel) for processing video below
        # TODO: put your pipeline here,
        # you should return the final output (image with lines are drawn on lanes)
        bgr=rgb_bgr(image)
        b=binarypipeline(bgr)
        warped = warpFun(b)
        if self.oldLaneLines:
            lc, rc = self.oldLaneLines.toCoeff()
            try:
                nonzerox, nonzeroy, left_lane_inds, right_lane_inds, left_fit, right_fit = findLanes_reuse(
                    warped, lc, rc)
            except:
                print("exception happened!", file=self.text_file)
                nonzerox, nonzeroy, left_lane_inds, right_lane_inds, left_fit, right_fit = findLanes_windowed(warped)
        else:
            nonzerox, nonzeroy, left_lane_inds, right_lane_inds, left_fit, right_fit = findLanes_windowed(warped)
            
        newLaneLines=LaneLines(coeff=[left_fit, right_fit])
        if self.oldLaneLines:
            probLaneLines=self.oldLaneLines.plausibleNew(newLaneLines)
            lc, rc=probLaneLines.toCoeff()
            if ((abs(lc[0])>0.001) | (abs(rc[0])>0.001)):
                print("unplausible high curvature!", file=self.text_file)
                nonzerox, nonzeroy, left_lane_inds, right_lane_inds, left_fit, right_fit = findLanes_windowed(warped)
                newLaneLines=LaneLines(coeff=[left_fit, right_fit])
                lc, rc=newLaneLines.toCoeff()
                if ((abs(lc[0])>0.0015) != (abs(rc[0])>0.0015)):
                    pos_car,dist_lanes,alpha_left,alpha_right,curverad_left,curverad_right=newLaneLines.toParam()
                    if (abs(lc[0])>0.0015):
                        alpha_left=alpha_right
                        curverad_left=curverad_right
                    else:
                        alpha_right=alpha_left
                        curverad_right=curverad_left
                    newLaneLines=LaneLines(params=[pos_car,dist_lanes,alpha_left,alpha_right,curverad_left,curverad_right])
                    
                probLaneLines=self.oldLaneLines.plausibleNew(newLaneLines)
                        
        else:
            probLaneLines=newLaneLines
            
        lc, rc=probLaneLines.toCoeff()
        result = drawLane3d(bgr, lc, rc, unwarpFun)
        statimg=drawLanes_warped(warped, nonzerox, nonzeroy, left_lane_inds, right_lane_inds, lc, rc)
        lc2, rc2 = probLaneLines.toCoeff2()
        print(lc2, rc2, file=self.text_file)
        pos_car,dist_lanes,alpha_left,alpha_right,curverad_left,curverad_right=probLaneLines.toParam()
        curverad=0.5*(curverad_left+curverad_right)
        
        statimg=cv2.resize(statimg, None, fx=1/3, fy=1/3)
        y_offset=50; x_offset=result.shape[1]-statimg.shape[1]-80
        result[y_offset:y_offset+statimg.shape[0], x_offset:x_offset+statimg.shape[1]] = statimg

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(result,"R = {0}".format(curverad),(50,100), font, 1,(255,255,255),2)
        cv2.putText(result,"pos = {0}".format(pos_car),(50,150), font, 1,(255,255,255),2)

        self.oldLaneLines=probLaneLines
        return bgr_rgb(result)

video_output = 'project_video_out.mp4'
clip1 = VideoFileClip("project_video.mp4")

text_file = open("Output.txt", "w")
process_image=ProcessImage(text_file)
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
%time white_clip.write_videofile(video_output, audio=False)
text_file.close()

```

    [MoviePy] >>>> Building video project_video_out.mp4
    [MoviePy] Writing video project_video_out.mp4


    100%|█████████▉| 1260/1261 [03:54<00:00,  5.71it/s]


    [MoviePy] Done.
    [MoviePy] >>>> Video ready: project_video_out.mp4 
    
    CPU times: user 12min 43s, sys: 1.63 s, total: 12min 44s
    Wall time: 3min 54s


## Discussion
---

- Briefly discuss any problems / issues you faced in your implementation of this project. Where will your pipeline likely fail? What could you do to make it more robust?

Problems really started when processing the video. It was really hard to figure out how I could make the detection stable in tricky situations, like shadows or dirt on the street. I gained much stability by introducing the `findLanes_reuse()` function, which looks for the lane line close to the polynomial from the last step. Furthermore, some weighted filtering of new and old coefficients helped. Furthermore, it was very important to increase the `minpix` parameter for the `findLanes_windowed()` function in order to make it more robust against noise.

For the video processing, I also implemented some plausibility checks. I inspected the numbers and found it highly unlikely, that the highest polynomial order coefficient exceeds the value 0.001. In that case I used to re-initialize by reverting back to the `findLanes_windowed()`. If the problem remained, I took curvature radius and alpha from the opposite lane lane, if there were plausible values to be found. Finally I reduced the lower threshold for the saturation component, which made the lane lines more prominent under bad lighning conditions.

Due to the hand-craftet parameters, like `minpixels`, saturation threshold, and gradient thresholds, it is very likely that the pipeline will fail under different lighning/weather conditions, and in case there appear different other gradients on the street as in the case of `challenge_video.mp4`: here, the conditions on the street introduce additional vertical gradients. Furthermore, the plausibiliy checks restrict the lane curvature, which will make it impossible to run the pipeline on streets with small curvature radius.

My feeling about possible improvements is that it is absolute essential to improve the image preprocessing. Highlighting the lane line pixels accurately is the key, 


```python
from IPython.display import HTML
HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(video_output))
```





<video width="960" height="540" controls>
  <source src="project_video_out.mp4">
</video>





```python

```
