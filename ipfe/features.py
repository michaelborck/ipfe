import numpy as np
import fast.fast9 as fast9
import mahatos as mh
from scipy import ndimage
import scipy.stats
import warnings
from skimage import img_as_float, img_as_int
from skimage.color import rgb2gray
from skimage import io
from skimage.feature import local_binary_pattern, corner_harris
from skimage.feature import corner_peaks, hog, daisy, canny
from skimage.filters import gabor_filter, sobel
from skimage.transform import resize
import mahatos as mh


def hist_intersection(histA, histB):
    """ Calcuates the intersection of two histograms.

    If two normalised histograms are the same then the sum of the intersection
    will be one.

    Assumes histograms are normalised.

    Parameters
    ----------
    histA: 1D numpy array
        normalised array where the sum of elements equals 1
    histB: 1D numpy array
        normalised array where the sum of elements equals 1

    Returns
    -------
    similarity: number
        Range 0-1. With similar -> 1
    """
    if histA == None or histB == None:
        return 0
    if len(histA) !=  len(histB): # Histogram same size
        return 0
    if histA.ndim != 1: # Must be single dimension histogrsm
        return 0
    return np.sum([min(a,b) for (a,b) in zip(histA,histB)])





def edge_orientation_histogram(image):
    """
    Implements the Mpeg7 Descriptor.  As I need to compare image and
    image patches which may be of a different size I have modified the
    original implementation to produce a single histogram descriptor for
    the entire region as a probabliity density function.

    Parameters
    ----------
    image: array
       the image, patch region to be described

    Returns
    -------
    eoh : array
       returns a histogram of five regions: 0, 45, 90, 135, none
    """
    # Define the Sobel edge filters for the 5 types of edges
    sobel = (
        np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]),
        np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
        np.array([[2, 2, -1], [2, -1, -1], [-1, -1, -1]]),
        np.array([[-1, 2, 2], [-1, -1, 2], [-1, -1, -1]]),
        np.array([[-1, 0, 1], [0, 0, 0], [1, 0, -1]]))

    # the size of the image
    ys = image.shape[0]
    xs = image.shape[1]

    # The image has to be in gray scale (intensities)
    if image.ndim == 3:
        image = rgb2gray(image)

    # Build a new matrix of the same size of the image
    # and 5 dimensions to save the gradients
    response = np.zeros((ys, xs, len(sobel)))

    # iterate over the posible directions
    for i, kernel in enumerate(sobel):
        # apply the sobel mask
        response[:, :, i] = ndimage.convolve(image, kernel)

    # find the index of the max gradient of the five  directions
    orientations = np.max(response, axis=2)

    # detect the edges using the default parameters
    edges = canny(image)

    # So we know where the edges are (binary image)
    # The canny edge detector thresholds the edges
    # Multiply against the types of orientations detected
    # by the Sobel masks and we have the orientations
    # of the strongest edges (where strongest was determine
    # by the canny edge detector
    edge_orientations = edges * orientations

    eoh, bins = np.histogram(edge_orientations.flatten(),
                             bins=5, density=True)
    # represent all the histograms on one vector
    return eoh.flatten()


def patch_HoG(image, size=None):
    """
    Extract Histogram of Oriented Gradients (HOG) for a given image.

    Parameters
    -----------
    image : (M, N) ndarray
        Input image (greyscale).
    size: tupple (m, n)
        Optionally resize image from MxN to mxn

    Returns
    -------
    newarr : ndarray
            HOG for the image as a 1D (flattened) array.
    hog_image : ndarray
        A visualisation of the HOG image.
    """

    img = rgb2gray(image)
    if size is not None:
        img = resize(img, size)
    fd, hog_img = hog(img, orientations=8, pixels_per_cell=(16, 16),
                      cells_per_block=(1, 1), visualise=True)
    hog_img = resize(hog_img, (image.shape[0], image.shape[1]))
    return fd.flatten(), hog_img


def patch_Daisy(image, size=None):
    """
    Extract DAISY feature descriptors densely for the given image.

    Parameters
    ----------
    image: (M, N) ndarray
        Input image.
    size:  tupple (m, n) optionally resize image from MxN to mxn

    Returns
    -------
    descs : ndarray
        Grid of DAISY descriptors for the given image as an array.
        See scikit image Daisy for more information
    descs_img : (M, N, 3) array
        Visualization of the DAISY descriptors.
    """
    img = rgb2gray(image)
    if size is not None:
        img = resize(img, size)
    fd, daisy_image = daisy(img, step=40, radius=15, rings=2,
                            histograms=6, orientations=8,
                            visualize=True)
    daisy_image = resize(daisy_image, image.shape)
    return fd.flatten(), daisy_image


def patch_fd(image, size=None):
    """
    Extract 1D array of image pixels as the descriptor

    Parameters
    ----------
    image : (M, N) ndarray
        Input image.

    size:  tupple (m, n) optionally resize image from MxN to mxn

    Returns
    -------
    desc : ndarray
        1D array of the image pixels
    image : ndarray
        image patch (may have resized image)
    """
    patch = resize(img_as_float(image), size)
    return patch.flatten(), patch


def patch_LBP(image, size=None):
    """
    Extract Local binary Pattern feature descriptor for the given image.

    Parameters
    ----------
    image : (M, N) ndarray
        Input image.
    size :  tupple (m, n) optionally resize image from MxN to mxn

    Returns
    -------
    histogram : array
        Probability density histogram of LBP image.
    image : ndarray
        LBP image
    """
    img = img_as_float(rgb2gray(image))
    if size is not None:
        img = resize(img, size)
    # settings for LBP
    radius = 2
    n_points = 8 * radius
    METHOD = 'uniform'
    lbp = local_binary_pattern(img, n_points, radius, METHOD)
    n_bins = 18  # lbp.max() + 1
    fd, _ = np.histogram(lbp, bins=n_bins, density=True)
    return fd, lbp


def patch_SURF(image, size=None):
    """
    Extract SIFT feature descriptor for the given image.

    Parameters
    ----------
    image: (M, N) ndarray
        Input image.
    size:  tupple (m, n) optionally resize image from MxN to mxn

    Returns
    -------
    density: float
        proportion of SIFT keypoint in image
    locs: array
        list of keypoint locations
    desrc : ndarray
        SIFT descriptor
    """
    img = img_as_int(rgb2gray(image))
    if size is not None:
        img = resize(img, size)
    return mh.features.surf.dense(image,spacing=16)


def patch_FAST(image, size=None):
    """
    Extract FAST keypoints and density measure.

    Parameters
    ----------
    image : (M, N) ndarray
        Input image.
    size :  tupple (m, n) optionally resize image from MxN to mxn

    Returns
    -------
    corners : ndarray
        locations of FAST keypoints
    density : float
        proportion of FAST keypoint in image
    """
    if size is not None:
        image = resize(image, size)
    (corners, scores) = fast9.detect(image, 20)
    return corners, len(corners) / float(image.shape[0] * image.shape[1])


def patch_Harris(image, size=None):
    """
    Extract Harris keypoints and density measure.

    Parameters
    ----------
    image : (M, N) ndarray
        Input image.

    size : tupple (m, n) optionally resize image from MxN to mxn

    Returns
    -------
    response : ndarray
         Harris response image
    corners : ndarray
         locations of Harris keypoints
    density: float
         proportion of Harris keypoint in image
    """
    if size is not None:
        image = resize(image, size)
    response = corner_harris(image)
    if response.sum():
        corners = corner_peaks(response)
        density = len(corners) / float(image.shape[0] * image.shape[1])
    else:
        corners = np.asarray([])
        density = 0
    return response, corners, density


def patch_Gabor(image, angle=0, size=None, feature=False):
    """
    Extract Gabor response, mean and std.

    Parameters
    ----------
    image : (M, N) ndarray
        Input image.

    angle : float
        Degree of Gabor filter.

    size : tupple (m, n) optionally resize image from MxN to mxn

    Returns
    -------
    real : ndarray
       "real" Gabor response (imag ignored)

    mean : float
        mean of gabor response
    std : float
        variance of gabor response
    """
    if size is not None:
        image = resize(image, size)
    real, _ = gabor_filter(image, 0.25, theta=np.deg2rad(angle))
    if feature:
        mean = np.mean(real)
        std = np.std(real)
        return real, mean, std
    else:
        return real


def edginess_sobel(image):
    '''Measure the "edginess" of an image

    image should be a 2d numpy array (an image)

    Returns a floating point value which is higher the "edgier" the image is.
    '''
    edges = sobel(rgb2gray(image))
    edges = edges.ravel()
    return np.sqrt(np.dot(edges, edges))


def texture_haralick(image):
    '''Compute features for an image

    Parameters
    ----------
    im : ndarray

    Returns
    -------
    fs : ndarray
        1-D array of features
    '''
    im = im.astype(np.uint8)
    return np.mean(mh.features.haralick(image),0)


def chist(im):
    '''Compute color histogram of input image
    Parameters
    ----------
    im : ndarray
        should be an RGB image
    Returns
    -------
    c : ndarray
        1-D array of histogram values
    '''

    # Downsample pixel values:
    im = im // 64

    # We can also implement the following by using np.histogramdd
    # im = im.reshape((-1,3))
    # bins = [np.arange(5), np.arange(5), np.arange(5)]
    # hist = np.histogramdd(im, bins=bins)[0]
    # hist = hist.ravel()

    # Separate RGB channels:
    r,g,b = im.transpose((2,0,1))

    pixels = 1 * r + 4 * g + 16 * b
    hist = np.bincount(pixels.ravel(), minlength=64)
    hist = hist.astype(float)
    return np.log1p(hist)


def edge_density(image, sigma=20, size=None):
    """
    Extract density of edges within a region.  Uses canny
    edge detector to calculate edge image

    Paramters
    ---------
    image : (M, N) ndarray
        Input image.
    sigma : float
        Smoothing for canny edge detector
    size : tupple (m, n) optionally resize image from MxN to mxn

    Returns
    -------
    edges : ndarray
        Canny edge may
    density : float
        proportion of Harris keypoint in image
    """
    if size is not None:
        image = resize(image, size)
    edges = img_as_int(canny(rgb2gray(image)), sigma)
    return edges, np.sum(edges.flatten()) / float(np.prod(edges.shape))


def _skew(image):
    """
    Internal function to handle warning messages of numpy.stats.skew()
    """
    warnings.filterwarnings('error')
    try:
        return scipy.stats.skew(image.flatten())
    except Exception:
        return 0


def _kurtosis(image):
    """
    Internal function to handle warning messages of numpy.stats.kurtosis()
    """
    warnings.filterwarnings('error')
    try:
        return scipy.stats.kurtosis(image.flatten())
    except Exception:
        return 0


def describe(image, size=None):
    """
    Create statistical descriptors of an image.

    Parameters
    ----------
    image : (M, N) ndarray
        Input image.
    size : tupple (m, n) optionally resize image from MxN to mxn

    Returns
    -------
    mean : float
        mean of image
    std: float
        variance of image
    skew : float
        skew of image
    kurtosis : float
        kurtosis of image
    energy : float
        energy of image
    entropy : float
        entropy of image
    """
    def histogram(L):
        d = {}
        for x in L:
            if x in d:
                d[x] += 1
            else:
                d[x] = 1
        return d
    if size is not None:
        image = resize(image, size)
    mean = np.mean(image.flatten())
    variance = np.var(image.flatten(), ddof=1)
    sk = _skew(image)
    kurt = _kurtosis(image)
    hist = histogram(image.flatten())
    npHist = np.asarray(hist.values())
    prob = npHist / float(image.shape[0] * image.shape[1])
    energy = np.sum(prob ** 2)
    entropy = np.sum(prob * np.log(prob)) * -1
    return mean, variance, sk, kurt, energy, entropy


def compute_kernel_features(image, kernels):
    """
    Convlove kernel with image.

    Parameters
    ----------
    image: (M, N) ndarray
        Input image.
    kernels: list
        list of mxn kernels

    Returns
    -------
    feats: list
        mean and variance of each kernel
    response: list of arrays
        list of each kernel response
    """
    response = np.zeros((image.shape[0], image.shape[1],
                         len(kernels)), dtype=np.double)
    feats = np.zeros((len(kernels), 2), dtype=np.double)
    for k, kernel in enumerate(kernels):
        print kernel
        response[:, : k] = ndimage.convolve(image, kernel, mode='wrap')
        feats[k, 0] = response[:, :k].mean()
        feats[k, 1] = response[:, :k].var()
    return feats, response


def get_simple_descriptors(image, keypoints, wid=5):
    """
    For each point return pixel values around the point using a neighbourhood
    of width 2*wid+1. (Assume points are extracted with min_distance > wid).

    Parameters
    ----------
    image: (M, N) ndarray
        Input image.
    keypoints: list
        list of locations to extract pixel patch
    width: int
        size of patch to extract

    Returns
    -------
    desc: list
        list of pixel values around a point, for each point

    """
    desc = []
    for coords in keypoints:
        patch = image[coords[0] - wid:coords[0] + wid + 1,
                      coords[1] - wid:coords[1] + wid + 1].flatten()
        desc.append(patch)
    return desc
