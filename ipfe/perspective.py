#!/usr/bin/env python

import cv2
import numpy as np
import math
from skimage import io

# https://code.google.com/p/pythonxy/source/browse/src/python/OpenCV/DOC/samples/python2/squares.py?spec=svn.xy-27.cd6bf12fae7ae496d581794b32fd9ac75b4eb366&repo=xy-27&r=cd6bf12fae7ae496d581794b32fd9ac75b4eb366
def angle_cos(p0, p1, p2):
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )


def rank(square, width, height):
  formatted = np.array([[s] for s in square], np.int32)
  x,y,wid,hei = cv2.boundingRect(formatted)
  max_distance_from_center = math.sqrt(((width / 2))**2 + ((height / 2))**2)
  distance_from_center = math.sqrt(((x + wid / 2) - (width / 2))**2 + ((y + hei / 2) - (height / 2))**2)

  height_above_horizontal = (height / 2) - y if y + hei > height / 2 else hei
  width_left_vertical = (width / 2) - x if x + wid > width / 2 else wid
  horizontal_score = abs(float(height_above_horizontal) / hei - 0.5) * 2
  vertical_score = abs(float(width_left_vertical) / wid - 0.5) * 2

  if cv2.contourArea(formatted) / (width * height) > 0.98:
    return 5 # max rank possible otherwise - penalize boxes that are the whole image heavily
  else:
    bounding_box = np.array([[[x,y]], [[x,y+hei]], [[x+wid,y+hei]], [[x+wid,y]]], dtype = np.int32)
    # every separate line in this addition has a max of 1
    return (distance_from_center / max_distance_from_center +
      cv2.contourArea(formatted) / cv2.contourArea(bounding_box) +
      cv2.contourArea(formatted) / (width * height) +
      horizontal_score +
      vertical_score)


def find_rectangles(image):
    # OpenCV follows BGR order, while the API follows RGB order.  Since using
    # OpenCV methods, function, we need to flip the colors dimension from
    # BGR to RGB (using only NumPy indexing)
    img = image[:,:,::-1]
    height = img.shape[0]
    width = img.shape[1]
    squares = []
    all_contours = []

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)) # matrix of ones
    for gray in cv2.split(img):
      dilated = cv2.dilate(src = gray, kernel = kernel, anchor = (-1,-1))
      blured = cv2.medianBlur(dilated, 7)
      # Shrinking followed by expanding can be used for removing isolated noise pixels
      # another way to think of it is "enlarging the background"
      # http://www.cs.umb.edu/~marc/cs675/cvs09-12.pdf
      small = cv2.pyrDown(blured, dstsize = (width / 2, height / 2))
      oversized = cv2.pyrUp(small, dstsize = (width, height))
      # after seeing utility of later thresholds (non 0 threshold results)
      # try instead to loop through and change thresholds in the canny filter
      # also might be interesting to store the contours in different arrays for display to color them according
      # to the channel that they came from
      for thrs in xrange(0, 255, 26):
        if thrs == 0:
          edges = cv2.Canny(oversized, threshold1 = 0, threshold2 = 50, apertureSize = 3)
          next_ = cv2.dilate(src = edges, kernel = kernel, anchor = (-1,-1))
        else:
          retval, next_ = cv2.threshold(gray, thrs, 255, cv2.THRESH_BINARY)

        image_, contours, hierarchy = cv2.findContours(next_, mode = cv2.RETR_LIST, method = cv2.CHAIN_APPROX_SIMPLE)
        # how are the contours sorted? outwards to inwards? would be interesting to do a PVE
        # sort of thing where the contours within a contour (and maybe see an elbow plot of some sort)
        for cnt in contours:
          all_contours.append(cnt)
          cnt_len = cv2.arcLength(cnt, True)
          cnt = cv2.approxPolyDP(cnt, 0.02*cnt_len, True)
          if len(cnt) == 4 and cv2.contourArea(cnt) > 1000 and cv2.isContourConvex(cnt):
            cnt = cnt.reshape(-1, 2)
            max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in xrange(4)])
            if max_cos < 0.1:
              squares.append(cnt)
    sorted_squares = sorted(squares, key=lambda square: rank(square,width,height))
    return sorted_squares



def straighten_rect(image, pts):
    """
    Given a set of pts that describe a polygon determine the parameters
    needed to transform the polygon to a rectangle.
    """
    # OpenCV follows BGR order, while the API follows RGB order.  Since using
    # OpenCV methods, function, we need to flip the colors dimension from
    # RGB to BGR (using only NumPy indexing)
    img = image[:,:,::-1]

    # We have our contour pts, we need to determine
    # the top-left, top-right, bottom-right, and bottom-left
    # points so that we can later warp the image -- we'll start
    # by reshaping our contour to be our finals and initializing
    # our output rectangle in top-left, top-right, bottom-right,
    # and bottom-left order
    #pts = sorted_squares[0]   #screenCnt.reshape(4, 2)
    rect = np.zeros((4, 2), dtype = "float32")

    # the top-left point has the smallest sum whereas the
    # bottom-right has the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # compute the difference between the points -- the top-right
    # will have the minumum difference and the bottom-left will
    # have the maximum difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # multiply the rectangle by the original ratio
    # rect *= ratio

    # now that we have our rectangle of points, let's compute
    # the width of our new image
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[0] - bl[0]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[0] - tl[0]) ** 2))

    # ...and now for the height of our new image
    heightA = np.sqrt(((tr[1] - br[1]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[1] - bl[1]) ** 2) + ((tl[1] - bl[1]) ** 2))

    # take the maximum of the width and height values to reach
    # our final dimensions
    maxWidth = max(int(widthA), int(widthB))
    maxHeight = max(int(heightA), int(heightB))

    # construct our destination points which will be used to
    # map the screen to a top-down, "birds eye" view
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")

    # calculate the perspective transform matrix and warp
    # the perspective to grab the screen
    M = cv2.getPerspectiveTransform(rect, dst)
    warp_img = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
    # Flip BGR to RGB (using only NumPy indexing)
    warp = warp_img[:,:,::-1]
    return M, maxWidth, maxHeight, warp


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    import sys
    # Loading image
    if len(sys.argv) == 2:
        filename = sys.argv[1] # for drawing purposes
    else:
        print "No input image given! \n"

    img = io.imread(filename,)
    sorted_squares = find_squares(img)
    M, maxWidth, maxheight, warp = straighten_square(img, sorted_squares[0])

    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].imshow(img)
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[1].imshow(warp)
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    plt.show()
