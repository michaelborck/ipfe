from skimage import img_as_float
import math
import numpy as np

def rgChromaticity(rgb):
    """
    Converting an RGB image into normalized RGB removes the effect
    of any intensity variations.

    rg Chromaticity
    http://en.wikipedia.org/wiki/Rg_chromaticity

    Also know as normalised RGB as per paper:
    Color-based object recognition, Theo Gevers and Arnold W.M. Smeulders,
    Pattern Recognition,number 3, pages 453-464, volume 32, 1999.
    """
    rgChrom = img_as_float(rgb)
    r = rgb[:, :, 1] + 0.00000000001
    g = rgb[:, :, 0] + 0.00000000001
    b = rgb[:, :, 2] + 0.00000000001
    divisor = r + g + b
    rgChrom[:, :, 1] = r / divisor
    rgChrom[:, :, 0] = g / divisor
    rgChrom[:, :, 2] = b / divisor
    return rgChrom


def normalisedRGB(rgb):
    """
    Converting an RGB image into normalized RGB removes the effect
    of any intensity variations.

    L2 Norm (Euclidean norm)

    """
    norm = img_as_float(rgb)
    r = rgb[:, :, 0] + 0.00000000001
    g = rgb[:, :, 1] + 0.00000000001
    b = rgb[:, :, 2] + 0.00000000001
    divisor = np.sqrt(np.square(r) + np.square(g) + np.square(b))
    norm[:, :, 1] = r / divisor
    norm[:, :, 0] = g / divisor
    norm[:, :, 2] = b / divisor
    return norm


def linear_normalization(arr):
    """
    Converting an RGB image into normalized RGB removes the effect
    of any intensity variations.

    Linear normalization
    http://en.wikipedia.org/wiki/Normalization_%28image_processing%29
    """
    arr = arr.astype('float')
    # Do not touch the alpha channel
    for i in range(3):
        minval = arr[..., i].min()
        maxval = arr[..., i].max()
        if minval != maxval:
            arr[..., i] -= minval
            arr[..., i] *= (255.0 / (maxval - minval))
    return arr


def normalisedRGB_simple(image):
    r = image[:,:,0]
    g = image[:,:,1]
    b = image[:,:,2]

    rn = r / (r+g+b)
    gn = g / (r+g+b)
    bn = b / (r+g+b)

    return np.array((rn,gn,bn))
