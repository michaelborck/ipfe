import os
import numpy as np
import urllib
from cStringIO import StringIO
from skimage import io, img_as_float
from skimage.transform import resize
import matplotlib.pyplot as plt


def url_to_img(image_url):
    """
    Read an image form a a URL and converts it to an image
    """
    img = None
    img_file = urllib.urlopen(image_url)
    img = io.imread(StringIO(img_file.read()))
    return img


def get_imlist(path):
    """
    Returns a list of filenames for all jpg images in a directory.
    """
    return [os.path.join(path, f) for f in os.listdir(path)
            if f.endswith('.jpg')]


def compute_average(imlist):
    """
    Compute the average of a list of images.
    """
    # open first image and make into array of type float
    averageim = img_as_float(io.imread(imlist[0]))
    size = averageim.shape   # make all images the same size as first

    for imname in imlist[1:]:
        try:
            averageim += resize(img_as_float(io.imread(imname)), size)
        except:
            print imname + "...skipped"
    averageim /= len(imlist)
    # return average as uint8
    return np.array(averageim, 'uint8')


def plot_points(image, filtered_coords):
    """
    Plots corners found in image.
    """
    plt.figure()
    plt.gray()
    plt.imshow(image)
    plt.plot([p[1] for p in filtered_coords],
             [p[0] for p in filtered_coords], '*')
    plt.axis('off')
    plt.show()


def overlay_points(im, filtered_coords):
    """
    Overlays corners found in image.
    """
    if im.max():
        im = im.astype(float) / float(im.max())
    radius = 3
    for cy, cx in filtered_coords:
        y, x = np.ogrid[-radius: radius, -radius: radius]
        index = x ** 2 + y ** 2 <= radius ** 2
        im[cy - radius:cy + radius, cx - radius:cx + radius][index] = 1
    return im


def appendimages(im1, im2):
    """
    Return a new image that appends the two images side-by-side.
    """

    # select the image with the fewest rows and fill in enough empty rows
    rows1 = im1.shape[0]
    rows2 = im2.shape[0]

    if rows1 < rows2:
        im1 = np.concatenate((im1, np.zeros((rows2 - rows1, im1.shape[1]))),
                             axis=0)
    elif rows1 > rows2:
        im2 = np.concatenate((im2, np.zeros((rows1 - rows2, im2.shape[1]))),
                             axis=0)
    # if none of these cases they are equal, no filling needed.
    return np.concatenate((im1, im2), axis=1)
