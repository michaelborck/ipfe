"""Additional test images.

For more images, see

 - http://sipi.usc.edu/database/database.php

"""
import numpy as np
from scipy import ndimage
from skimage.io import imread
import os.path as _osp

data_dir = _osp.abspath(_osp.dirname(__file__)) + "/data"


def load(f):
    """Load an image file located in the data directory.

    Parameters
    ----------
    f : string
        File name.

    Returns
    -------
    img : ndarray
        Image loaded from skimage.data_dir.
    """
    return imread(_osp.join(data_dir, f))


def stinkbug():
    """24-bit RGB PNG image of a stink bug

    Notes
    -----
    This image was downloaded from Matplotlib tutorial
    <http://matplotlib.org/users/image_tutorial.html>

    No known copyright restrictions, released into the public domain.

    """
    return load("stinkbug.png")


def nuclei():
    """Gray-level "cell nuclei" image used for blob processing.

    Notes
    -----
    This image was downloaded from Pythonvision Basic Tutorial
    <http://pythonvision.org/media/files/images/dna.jpeg>

    No known copyright restrictions, released into the public domain.

    """
    return load("nuclei.png")


def baboon():
    """Baboon.

    Notes
    -----
    This image was downloaded from the Signal and Image Processing Institute at the
    University of Southern California.
    <http://sipi.usc.edu/database/database.php?volume=misc&image=11#top>

    No known copyright restrictions, released into the public domain.

    """
    return load("baboon.png")


def city_depth():
    """City depth map.

    Notes
    -----
    No known copyright restrictions, released into the public domain.

    """
    return load("city_depth.png")


def city():
    """City.

    Notes
    -----
    This image was generate form the Earthmine datas set of Perth, Australia.

    This image is copyright earthmine.

    """
    return load("city.png")


def aerial():
    """Aerial Sna Diego (shelter Island).

    Notes
    -----
    This image was downloaded from the Signal and Image Processing Institute at the
    University of Southern California.
    <http://sipi.usc.edu/database/database.php?volume=aerials&image=10#top>

    No known copyright restrictions, released into the public domain.

    """
    return load("aerial.png")


def peppers():
    """Peppers.

    Notes
    -----
    This image was downloaded from the Signal and Image Processing Institute at the
    University of Southern California.
    <http://sipi.usc.edu/database/database.php?volume=misc&image=11#top>

    No known copyright restrictions, released into the public domain.

    """
    return load("peppers.png")


def sign():
    """Sign.
    """
    return load("sign.png")


def microstructure(l=256):
    """Synthetic binary data: binary microstructure with blobs.

    Parameters
    ----------

    l: int, optional
        linear size of the returned image

    Code fragment from scikit-image Medial axis skeletonization see:
    <http://scikit-image.org/docs/dev/auto_examples/plot_medial_transform.html>

    No known copyright restrictions, released into the public domain.

    """
    n = 5
    x, y = np.ogrid[0:l, 0:l]
    mask = np.zeros((l, l))
    generator = np.random.RandomState(1)
    points = l * generator.rand(2, n ** 2)
    mask[(points[0]).astype(np.int), (points[1]).astype(np.int)] = 1
    mask = ndimage.gaussian_filter(mask, sigma=l / (4. * n))
    return mask


def noisy_blobs(noise=True):
    """Synthetic binary data: four circles with noise.

    Parameters
    ----------

    noise: boolean, optional
        include noise in image

    Code fragment from 'Image manipulation and processing using Numpy and Scipy'
    <http://www.tp.umu.se/~nylen/fnm/pylect/advanced/image_processing/index.html>

    No known copyright restrictions, released into the public domain.
    """

    np.random.seed(1)
    n = 10
    l = 256
    blob = np.zeros((l, l))
    points = l * np.random.random((2, n ** 2))
    blob[(points[0]).astype(np.int), (points[1]).astype(np.int)] = 1
    blob = ndimage.gaussian_filter(blob, sigma=l / (4. * n))
    if noise:
        mask = (blob > blob.mean()).astype(np.float)
        mask += 0.1 * blob
        blob = mask + 0.2 * np.random.randn(*mask.shape)
    return blob


def noisy_circles(noise=True):
    """Synthetic binary data: four circles with noise.

    Parameters
    ----------

    noise: boolean, optional
        include noise in image


    Code fragment from 'Image manipulation and processing using Numpy and Scipy'
    <http://www.tp.umu.se/~nylen/fnm/pylect/advanced/image_processing/index.html>

    No known copyright restrictions, released into the public domain.
    """

    l = 100
    x, y = np.indices((l, l))

    center1 = (28, 24)
    center2 = (40, 50)
    center3 = (67, 58)
    center4 = (24, 70)

    radius1, radius2, radius3, radius4 = 16, 14, 15, 14

    circle1 = (x - center1[0]) ** 2 + (y - center1[1]) ** 2 < radius1 ** 2
    circle2 = (x - center2[0]) ** 2 + (y - center2[1]) ** 2 < radius2 ** 2
    circle3 = (x - center3[0]) ** 2 + (y - center3[1]) ** 2 < radius3 ** 2
    circle4 = (x - center4[0]) ** 2 + (y - center4[1]) ** 2 < radius4 ** 2

    # 4 circles
    image = circle1 + circle2 + circle3 + circle4

    if noise:
        image = image.astype(float)
        image += 1 + 0.2 * np.random.randn(*image.shape)

    return image


def noisy_square(noise=True):
    """Synthetic binary data: square with noise.

    Parameters
    ----------

    noise: boolean, optional
        include noise in image

    Code fragment from scikit-image 'Canny edge detector'. see:
    <http://scikit-image.org/docs/dev/auto_examples/plot_canny.html>

    Also in 'Image manipulation and processing using Numpy and Scipy'
    <http://www.tp.umu.se/~nylen/fnm/pylect/advanced/image_processing/index.html>

    No known copyright restrictions, released into the public domain.

    """
    # Generate noisy image of a square
    im = np.zeros((128, 128))
    im[32:-32, 32:-32] = 1

    im = ndimage.rotate(im, 15, mode='constant')
    im = ndimage.gaussian_filter(im, 4)
    im += 0.2 * np.random.random(im.shape)

    if noise:
        im = im.astype(float)
        im += 0.2 * np.random.random(im.shape)

    return im


def mri():
    """MRI.

    Notes
    -----
    This image was created/downloaded from the Matplotlib using the following:

    dfile= cbook.get_sample_data('s1045.ima', asfileobj=False)
    im = np.fromstring(file(dfile, 'rb').read(), np.uint16).astype(float)
    im.shape = 256, 256
    im2 = im.astype(float)/float(im.max())
    imsave('mri.png',im2)

    Code frament from 'pylab_examples example code: mri_with_eeg.py'
    <http://matplotlib.org/examples/pylab_examples/mri_with_eeg.html>

    No known copyright restrictions, released into the public domain.

    """
    return load("mri.png")


def cells():
    """Cells

    From in 'Image manipulation and processing using Numpy and Scipy' section
    2.6.6 Measuring object properties
    <http://www.tp.umu.se/~nylen/fnm/pylect/advanced/image_processing/index.html>

    No known copyright restrictions, released into the public domain.

    """
    np.random.seed(1)
    n = 10
    l = 256
    im = np.zeros((l, l))
    points = l * np.random.random((2, n ** 2))
    im[(points[0]).astype(np.int), (points[1]).astype(np.int)] = 1
    im = ndimage.gaussian_filter(im, sigma=l / (4. * n))
    return im


def cross():
    """Synthetic binary data showing a cross

    Code fragment from scikit-image:
    <http://scikit-image.org/docs/dev/auto_examples/plot_hough_transform.html#example-plot-hough-transform-py>

    No known copyright restrictions, released into the public domain.

    """
    image = np.zeros((100, 100))
    idx = np.arange(25, 75)
    image[idx[::-1], idx] = 255
    image[idx, idx] = 255
    return image


def misc():
    """Synthetic binary data

    Code fragment from scikit-image:
    <http://scikit-image.org/docs/0.7.0/api/skimage.transform.hough_transform.html>

    No known copyright restrictions, released into the public domain.

    """
    image = np.zeros((100, 150), dtype=bool)
    image[30, :] = 1
    image[:, 65] = 1
    image[35:45, 35:50] = 1
    for i in range(90):
        image[i, i] = 1
    image += np.random.random(image.shape) > 0.95
    return image


def random(same=False):
    """Synthetic binary data

    Released into the public domain.

    """
    # Generate standardized random data
    if same:
        np.random.seed(seed=1234)
    else:
        np.random.seed()
    return np.random.randint(0, 255, size=(256, 256))


def overlapping_circles():
    """Synthetic binary data

    Generate a binary image with two overlapping circles

    Released into the public domain.

    """
    x, y = np.indices((80, 80))
    x1, y1, x2, y2 = 28, 28, 44, 52
    r1, r2 = 16, 20
    mask_circle1 = (x - x1) ** 2 + (y - y1) ** 2 < r1 ** 2
    mask_circle2 = (x - x2) ** 2 + (y - y2) ** 2 < r2 ** 2
    image = np.logical_or(mask_circle1, mask_circle2)
    return image
