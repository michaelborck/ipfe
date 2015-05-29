#!/usr/bin/env python
from __future__ import division

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import io
from skimage.color import rgb2gray
from skimage.measure import label
from skimage.measure import regionprops
from skimage.segmentation import clear_border
from skimage.morphology import erosion, dilation, opening, closing, white_tophat
from skimage.morphology import black_tophat, skeletonize, convex_hull_image
from skimage.morphology import disk, square
from skimage import img_as_float, img_as_uint


def local_max_region(image, response, threshold, shape):
    img_copy = image
    detected = rgb2gray(response)
    height, width = detected.shape
    sm = ndimage.gaussian_filter(detected, sigma=5)
    bw = sm > threshold  #70
    selem = shape   #disk(15)
    dilated = dilation(bw, selem)

    # to be safe, remove any artifacts connect to border
    cleared = dilated.copy()
    clear_border(cleared)

    # label the remain regions (should be "signs")
    label_image = label(cleared)
    borders = np.logical_xor(dilated, cleared)
    label_image[borders] = -1

    patches = []
    # plot the brighest points over original image
    region_num = 0
    for region in regionprops(label_image):
        # find center of region
        minr, minc, maxr, maxc = region.bbox
        #check aspect ratio
        aspect_ratio = (maxc - minc) / (maxr - minr)
        #print aspect_ratio
        if (aspect_ratio > 1.3) or (aspect_ratio < 0.3):
            continue
        if region.area < 900: # or region.area > 2000:
            continue
        patches.append(img_copy[minr:maxr,minc:maxc].copy())
        #centr = minr + (maxr-minr)/2
        #centc = minc + (maxc-minc)/2
    return patches

if __name__ == '__main__':
    # Get list of files to process, crude/basic command line processing
    if len(sys.argv) < 3:
        print "Usage: %s image response" % (sys.argv[0])
        sys.exit()
    image = io.imread(sys.argv[1])
    response = io.imread(sys.argv[2])
    patches = segment_signs(image, response)
    for patch in patches:
        plt.imshow(patch)
        plt.show()
