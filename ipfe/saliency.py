#!/usr/bin/python
import os
import numpy as np
import math
from scipy import fftpack, ndimage, misc
import skimage as si


def edge_based(image):
    """
    A simple method for detecting salient regions
    Paul L. Rosin

    Abstract
    A simple method for detecting salient regions in images is proposed. It
    requires only edge detection, threshold decomposition, the distance
    transform, and thresholding. Moreover, it avoids the need for setting any
    parameter values. Experiments show that the resulting regions are
    relatively coarse, but overall the method is surprisingly effective, and
    has the benefit of easy implementation. Quantitative tests were carried out
    on Liu et al.'s dataset of 5000 images. Although the ratings of our simple
    method were not as good as their approach which involved an extensive
    training stage, they were comparable to several other popular methods from
    the literature. Further tests on Kootstra and Schomaker's dataset of 99
    images also showed promising results.

    Publication
    Paul L. Rosin
    A simple method for detecting salient regions
    Pattern Recognition, 2009, 42, 11, 2363 - 2371.

    Args:
        image: numpy array
          RGB image as an array

    Returns:
        sm: numpy array
           Normalised salicency map

    Need to write proper python wrapper.  For no will only
    is binary edge_based_saliency_map is in users path
    """

    si.io.imsave('edge.png', image)
    os.system('/opt/local/bin/convert edge.png edge.pgm')
    os.system('/Users/michael/bin/salient_regions -i edge.pgm \
              -o overlay.pgm -m mask.pgm -s smedge.pgm ')
              #-o overlay.pgm -s smedge.pgm ')
    os.system('/opt/local/bin/convert -negate smedge.pgm sm.pgm')
    sm = si.io.imread('sm.pgm')
    #sm = si.io.imread('smedge.pgm')
    sm *= 255.0 / sm.max()  # Normalise image
    os.system('rm -rf edge.pgm sm.pgm smedge.pgm overlay.pgm mask.pgm')
    #os.system('rm -rf edge.pgm sm.pgm smedge.pgm overlay.pgm')
    return sm


def frequency_tuned(image):
    """
    Frequency-tuned Salient Region Detection
    Radhakrishna Achanta, Sheila Hemami, Francisco Estrada, Sabine Susstrunk

    Abstract
    Detection of visually salient image regions is useful for applications like
    object segmentation, adaptive compression, and object recognition. In this
    paper, we introduce a method for salient region detection that outputs full
    resolution saliency maps with well-defined boundaries of salient objects.
    These boundaries are preserved by retaining substantially more frequency
    content from the original image than other existing techniques. Our method
    exploits features of color and luminance, is simple to implement, and is
    computationally efficient. We compare our algorithm to five
    state-of-the-art salient region detection methods with a frequency domain
    analysis, ground truth, and a salient object segmentation application. Our
    method outperforms the five algorithms both on the ground truth evaluation
    and on the segmentation task by achieving both higher precision and better
    recall.

    Reference and PDF
    R. Achanta, S. Hemami, F. Estrada and S. Susstrunk, Frequency-tuned Salient
    Region Detection, IEEE International Conference on Computer Vision and
    Pattern Recognition (CVPR), 2009.

    URL: http://ivrg.epfl.ch/supplementary_material/RK_CVPR09/

    Args:
        image: numpy array
          RGB image as an array

    Returns:
        sm: numpy array
           Normalised salicency map
    """

    if len(image.shape) != 3:
        #print 'Warning processing non standard image'
        image = si.color.gray2rgb(image)

    # Convert to CIE Lab colour space

    image = ndimage.gaussian_filter(image, sigma=3)
    image = si.color.rgb2lab(image)

    # Get each channel

    l = image[:, :, 0]
    a = image[:, :, 1]
    b = image[:, :, 2]

    # LAB image average

    lm = np.mean(l)
    am = np.mean(a)
    bm = np.mean(b)

    # Compute the saliency map

    sm = (l - lm) ** 2 + (a - am) ** 2 + (b - bm) ** 2

    # Normalise saliency map

    sm *= 255 / sm.max()
    return sm


def luminance_and_colour(image):
    """
    Salient Region Detection and Segmentation
    Radhakrishna Achanta

    Abstract
    Detection of salient image regions is useful for applications like image
    segmentation, adaptive compression, and region-based image retrieval. In
    this paper we present a novel method to determine salient regions in images
    using low-level features of luminance and color. The method is fast, easy
    to implement and generates high quality saliency maps of the same size and
    resolution as the input image.We demonstrate the use of the algorithm in
    the segmentation of semantically meaningful whole objects from digital
    images.

    Publication
    R. Achanta, F. Estrada, P. Wils and S. Susstrunk, Salient Region Detection
    and Segmentation, International Conference on Computer Vision Systems
    (ICVS '08), Vol. 5008, Springer Lecture Notes in Computer Science,
    pp. 66-75, 2008.

    ivrg.epfl.ch/~achanta/SalientRegionDetection/SalientRegionDetection.html

    Args:
        image: numpy array
          RGB image as an array

    Returns:
        sm: numpy array
           Normalised salicency map
    """

    if len(image.shape) != 3:
        #print 'Warning processing non standard image'
        image = si.color.gray2rgb(image)

    # Convert to CIE Lab colour space

    image = ndimage.gaussian_filter(image, sigma=3)
    image = si.color.rgb2lab(image)

    # Get each channel

    l = image[:, :, 0]
    a = image[:, :, 1]
    b = image[:, :, 2]

    height = image.shape[0]
    width = image.shape[1]
    mindim = min(width, height)
    sm = np.zeros((height, width))  # empty saliency map
    off1 = mindim / 2
    off2 = mindim / 4
    off3 = mindim / 8
    for j in range(1, height):
        y11 = max(1, j - off1)
        y12 = min(j + off1, height - 1)
        y21 = max(1, j - off2)
        y22 = min(j + off2, height - 1)
        y31 = max(1, j - off3)
        y32 = min(j + off3, height - 1)
        for k in range(1, width):
            x11 = max(1, k - off1)
            x12 = min(k + off1, width - 1)
            x21 = max(1, k - off2)
            x22 = min(k + off2, width - 1)
            x31 = max(1, k - off3)
            x32 = min(k + off3, width - 1)
            lm1 = np.mean(l[y11:y12, x11:x12])
            am1 = np.mean(a[y11:y12, x11:x12])
            bm1 = np.mean(b[y11:y12, x11:x12])
            lm2 = np.mean(l[y21:y22, x21:x22])
            am2 = np.mean(a[y21:y22, x21:x22])
            bm2 = np.mean(b[y21:y22, x21:x22])
            lm3 = np.mean(l[y31:y32, x31:x32])
            am3 = np.mean(a[y31:y32, x31:x32])
            bm3 = np.mean(b[y31:y32, x31:x32])

            # Compute conspucity values

            cv1 = (l[j][k] - lm1) ** 2 + (a[j][k] - am1) ** 2 \
                + (b[j][k] - bm1) ** 2
            cv2 = (l[j][k] - lm2) ** 2 + (a[j][k] - am2) ** 2 \
                + (b[j][k] - bm2) ** 2
            cv3 = (l[j][k] - lm3) ** 2 + (a[j][k] - am3) ** 2 \
                + (b[j][k] - bm3) ** 2

        # Combine conspucity to create Saliency map

        sm[j][k] = cv1 + cv2 + cv3

    # Normalise image

    sm *= 255.0 / sm.max()  # Normalise image
    return sm


def maximum_symmetric_surround(image):
    """
    Saliency Detection using Maximum Symmetric Surround
    Radhakrishna Achanta and Sabine Susstrunk


    Abstract
    Detection of visually salient image regions is useful for applications like
    object segmentation, adaptive compression, and object recognition.
    Recently, full-resolution salient maps that retain well-defined boundaries
    have attracted attention. In these maps, boundaries are preserved by
    retaining substantially more frequency content from the original image than
    older techniques. However, if the salient regions comprise more than half
    the pixels of the image, or if the background is complex, the background
    gets highlighted instead of the salient object. In this paper, we introduce
    a method for salient region detection that retains the advantages of full
    resolution saliency maps with well-defined boundaries while overcoming
    their shortcomings. Our method exploits features of color and luminance, is
    simple to implement and is computationally efficient. We compare our novel
    algorithm to six state-of-the-art salient region detection methods using
    publicly available ground truth. Our method outperforms the six algorithms
    by achieving both higher precision and better recall. We also show
    application of our saliency maps in an automatic salient object
    segmentation scheme using graph-cuts.

    Reference
    Radhakrishna Achanta and Sabine Susstrunk, Saliency Detection using Maximum
    Symmetric Surround, International Conference on Image Processing (ICIP),
    Hong Kong, September 2010.

    URL: http://ivrg.epfl.ch/supplementary_material/RK_ICIP2010/

    Args:
        image: numpy array
          RGB image as an array

    Returns:
        sm: numpy array
           Normalised salicency map
    """

    if len(image.shape) != 3:
        #print 'Warning processing non standard image'
        image = si.color.gray2rgb(image)

    # Convert to CIE Lab colour space

    image = ndimage.gaussian_filter(image, sigma=3)
    image = si.color.rgb2lab(image)

    # Get each channel

    l = image[:, :, 0]
    a = image[:, :, 1]
    b = image[:, :, 2]

    height = image.shape[0]
    width = image.shape[1]
    sm = np.zeros((height, width))
    for j in range(1, height):  # Start from lab(1,1)
        yo = min(j, height - j)
        for k in range(1, width):
            xo = min(k, width - k)
            lm = np.mean(l[max(0, j - yo):min(j + yo, height - 1),
                         max(0, k - xo):min(k + xo, width - 1)])
            am = np.mean(a[max(0, j - yo):min(j + yo, height - 1),
                         max(0, k - xo):min(k + xo, width - 1)])
            bm = np.mean(b[max(0, j - yo):min(j + yo, height - 1),
                         max(0, k - xo):min(k + xo, width - 1)])
            sm[j][k] = (l[j][k] - lm) ** 2 + (a[j][k] - am) ** 2 \
                + (b[j][k] - bm) ** 2

    # Save Saliency Map for later processing

    sm *= 255.0 / sm.max()  # Normalise image
    return sm


def spectral_residual(image):
    """
    Saliency Detection: A Spectral Residual Approach
    Xiaodi Hou and Liqing Zhang

    Abstract
    The ability of human visual system to detect visual saliency is
    extraordinarily fast and reliable. However, computational modeling of this
    basic intelligent behavior still remains a challenge. This paper presents a
    simple method for the visual saliency detection. Our model is independent
    of features, categories, or other forms of prior knowledge of the objects.
    By analyzing the log-spectrum of an input image, we extract the spectral
    residual of an image in spectral domain, and propose a fast method to
    construct the corresponding saliency map in spatial domain. We test this
    model on both natural pictures and artificial images such as psychological
    patterns. The result indicate fast and robust saliency detection of our
    method.

    Publication
    Xiaodi Hou and Liqing Zhang Saliency Detection: A Spectral Residual
    Approach Journal Computer Vision and Pattern Recognition, IEEE Computer
    Society Conference 2007

    www.klab.caltech.edu/~xhou/projects/spectralResidual/spectralresidual.html

    Args:
        image: numpy array
          RGB image as an array

    Returns:
        sm: numpy array
           Normalised salicency map
    """

    if len(image.shape) == 3:
        image = matlab_rgb2gray(image)

    image = misc.imresize(image, 64.0 / image.shape[1])  # Scale Image
    image = si.img_as_float(image)
    fft = fftpack.fft2(image)
    logAmplitude = np.log(np.abs(fft))
    phase = np.angle(fft)
    avgLogAmp = ndimage.uniform_filter(logAmplitude, size=3, mode="nearest")
    spectralResidual = logAmplitude - avgLogAmp
    sm = np.abs(fftpack.ifft2(np.exp(spectralResidual + 1j * phase))) ** 2
    sm = ndimage.gaussian_filter(sm, sigma=2.5)
    #sm = ndimage.correlate(sm, fgau_mat(), mode='nearest')
    return sm


def phase_map(image):
    """
    Spatio-temporal Saliency Detection Using Phase Spectrum of
    Quaternion Fourier transform.  Chenlei Guo, Qi Ma and Liming Zhang

    The above paper shows that the saliency map can be produce using
    just the phase compoment of the FFT, reducing computation by 1/3,
    with no significant difference in the saliency map.

    Args:
        image: numpy array
          RGB image as an array

    Returns:
        sm: numpy array
           Normalised salicency map
    """
    if len(image.shape) == 3:
        image = si.color.rgb2gray(image)
        #image = matlab_rgb2gray(image)

    fft = fftpack.fft2(image)
    phase = np.angle(fft)
    sm = np.abs(fftpack.ifft2(np.exp(1j * phase))) ** 2
    sm = ndimage.gaussian_filter(sm, sigma=5)
    return sm


def fgau_mat():
    """
    Load filter form MATLAB
    """
    from scipy.io import loadmat
    fgau = loadmat('/Users/michael/PhD/MATLAB/matlab.mat')
    return fgau['gau']


def fgaussian(size, sigma):
    """
    Calculate a simple gaussian
    """
    m, n = size
    h, k = m // 2, n // 2
    x, y = np.mgrid[-h:h, -k:k]
    c = 1 / ((2 * math.pi) * sigma * sigma)
    return c * np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))


def makeGaussian(size, fwhm=3, center=None):
    """ Make a square gaussian kernel.

    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / fwhm ** 2)


def matlab_rgb2gray(rgb):  # Same as MATLAB
    """
    Attempt to get colour consistency between MATLAB and scikit image
    """
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray
