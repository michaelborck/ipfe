import numpy as np
import skimage as si
import util


def planarity(normals):
    """
    Calculate the planarity of array of surface normals.
    Planarity is 1 - dot product of k-neighbourhod normals

    Args:
        normals: array of tuples (x,y,z)
           Array of surface normals

    Returns:
        response: array
            the planarity measure of each element in normals

        mean: float
            mean of response

       variance: float
           variance of response
    """
    response = np.zeros_like(normals[:, :, 0])
    row,  col, _ = normals.shape
    for r in range(1,  row - 1):
        for c in range(1,   col - 1):
            sum_d = np.dot(normals[r, c], normals[r - 1, c - 1])
            sum_d += np.dot(normals[r, c], normals[r - 1, c])
            sum_d += np.dot(normals[r, c], normals[r - 1, c + 1])
            sum_d += np.dot(normals[r, c], normals[r, c - 1])
            sum_d += np.dot(normals[r, c], normals[r, c + 1])
            sum_d += np.dot(normals[r, c], normals[r + 1, c - 1])
            sum_d += np.dot(normals[r, c], normals[r + 1, c])
            sum_d += np.dot(normals[r, c], normals[r + 1, c + 1])
            response[r, c] = 1 - (sum_d / 8.0)
    return response, np.mean(response), np.std(response)


def surface_normal(Z):
    """
    Calculate the Surface normals.

    Equation (2) in "Computation of Surface Curvature from Range Images
    Using Geometrically Intrinsic Weights"*, T. Kurita and P. Boulanger, 1992.

    Args:
        Z: array
           depth map

    Returns:
        N: array
           the surface normal values
    """
    Zy, Zx = np.gradient(Z)
    N = (-Zx, -Zy, 1) / np.sqrt(1 + (Zx ** 2) + (Zy ** 2))
    return N


def mean_curvature(Z):
    """
    Calculate the mean curvature.

    Equation (3) in "Computation of Surface Curvature from Range Images
    Using Geometrically Intrinsic Weights"*, T. Kurita and P. Boulanger, 1992.

    Args:
        Z: array
           depth map

    Returns:
        H: array
           the mean curvature values
    """
    Zy, Zx = np.gradient(Z)
    Zxy, Zxx = np.gradient(Zx)
    Zyy, _ = np.gradient(Zy)
    H = (Zx ** 2 + 1) * Zyy - 2 * Zx * Zy * Zxy + (Zy ** 2 + 1) * Zxx
    H = -H / (2 * (Zx ** 2 + Zy ** 2 + 1) ** (1.5))
    return H


def gaussian_curvature(Z):
    """
    Calculate the Gaussian curvature.

    Equation (4) in "Computation of Surface Curvature from Range Images
    Using Geometrically Intrinsic Weights"*, T. Kurita and P. Boulanger, 1992.

    Args:
        Z: array
           depth map

    Returns:
        K: array
           the gaussian curvature values
    """
    Zy, Zx = np.gradient(Z)
    Zxy, Zxx = np.gradient(Zx)
    Zyy, _ = np.gradient(Zy)
    K = (Zxx * Zyy - (Zxy ** 2)) / (1 + (Zx ** 2) + (Zy ** 2)) ** 2
    return K


def curvature(Z):
    """
    Calculate the mean curvature.
    Calculate the Gaussian curvature.
    Calculate the principal curvature.

    Equation (3) & (4) in "Computation of Surface Curvature from Range Images
    Using Geometrically Intrinsic Weights"*, T. Kurita and P. Boulanger, 1992.

    Args:
        Z: array
           depth map

    Returns:
        K: array
           the gaussian curvature values
    """
    Zy, Zx = np.gradient(Z)
    Zxy, Zxx = np.gradient(Zx)
    Zyy, _ = np.gradient(Zy)

    # Mean Curvature
    H = (Zx ** 2 + 1) * Zyy - 2 * Zx * Zy * Zxy + (Zy ** 2 + 1) * Zxx
    H = -H / (2 * (Zx ** 2 + Zy ** 2 + 1) ** (1.5))

    # Gaussian Curvature
    K = (Zxx * Zyy - (Zxy ** 2)) / (1 + (Zx ** 2) + (Zy ** 2)) ** 2

    # Principal Curvatures
    Pmax = H + np.sqrt(H ** 2 - K)
    Pmin = H - np.sqrt(H ** 2 - K)
    return H, K, Pmax, Pmin


def histogram_of_surface_normals(image):
    """
    Calculate the Histogram of Surface Normals (HoSN)

    Parameters
    ----------
    image: array
       depth map

    Returns
    -------
    response: array
        array of angles same size of image
    HoSN: array
       response as histogram over 5 bins
    """
    angles, normals = pixel_angle_to_plane(image)
    #hosn, bin = np.histogram(np.degrees(angles), bins=4, density=True)
    hosn, bin = np.histogram(np.degrees(angles), density=True)
    return angles, hosn


def pixel_angle_to_plane(image, plane=[0, 0, 1], size=3):
    """
    For each pixel in the image:
    1. Fit a plane to surrounding pixels
    2. Find normal to the plane
    3. Orientation = fitted .dot. plane

    Parameters
    ----------
    image : ndarry
        depth map
    plane : tupple (x,y,z)
        plane to to calculat between
    size : int
        k-neighbourhood size

    Returns
    -------
    angle_to_plane : array
        array of angles same size of image
    normals : list
        list of normals calculated
    """
    if image.ndim > 3:
        image = si.color.rgb2gray(image)
        #raise ValueError("Currently only supports grey-level images")

    rows, cols = image.shape
    angle_to_plane = np.zeros_like(image)
    normals = np.zeros((image.shape[0], image.shape[1], 3))
    for r in range((size - 1), rows - (size - 1)):  # allow for boarder pixels
        for c in range((size - 1), cols - (size - 1)):
            box = image[r - (size - 1):r + (size - 1), c - (size - 1):
                        c + (size - 1)]
            normal = np.array(util.vector.fit_plane_SVD(box))
            normals[r, c, 0] = normal[0]
            normals[r, c, 1] = normal[1]
            normals[r, c, 2] = normal[2]
            angle_to_plane[r, c] = util.vector.angle(normal, plane)
    return angle_to_plane, normals


def in_front_of(depth):
    """
    Calculates the "in-front-ness" of pixels with its k-neighbourhod

    Parameters
    ----------
    depth : array
       depth map

    Returns
    -------
    in_front_image : array
       image response
    in_front_mean : array
       array of in_front mean value
    in_front_std : array
       array of in_front std value
    """
    #in_front_image = np.zeros_like(depth)
    in_front_mean = np.zeros_like(depth)
    in_front_std = np.zeros_like(depth)
    row,  col = depth.shape
    for r in range(1,  row - 1):
        for c in range(1,   col - 1):
            hood = np.array((depth[r - 1, c - 1], depth[r - 1, c],
                             depth[r - 1, c + 1], depth[r, c - 1],
                             depth[r, c + 1], depth[r + 1, c - 1],
                             depth[r + 1, c], depth[r + 1, c + 1]))
            nh_mean = np.mean(hood)
            if depth[r, c] < nh_mean:
                depth_diff = [np.abs(depth[r, c] - x) for x in hood]
                in_front_mean[r, c] = np.mean(depth_diff)
                in_front_std[r, c] = np.std(depth_diff)
    #return np.mean(in_front_mean), np.mean(in_front_std)
    return in_front_mean, in_front_std


def histogram_of_depth_difference(image, x0, y0, x1, y1, cell_size=3):
    """
    Calcuate the histogram of depth difference between a cell and
    every other cell in the image.

    Parameters
    ----------
    image :  ndarray
       Image under consideration
    x0,y0,x1,y1 : int
       Region of interest
    cell_size : int
       fixed cell size.

    Returns
    -------
    ldp : array
       Image response of HoDD
    hist : array
       Histogram of image response across 10 bins.
    mean : float
       means of HoDD
    variance : float
       means of HoDD
    """
    print "Inside HoDD"
    avgD = []
    ii = si.transform.integral_image(image)
    for row in range(y0 + cell_size - 1, y1, cell_size):
        for col in range(x0 + cell_size - 1, x1, cell_size):
            avgD.append(si.transform.integrate(ii, row - cell_size, col -
                        cell_size, row, col) / (cell_size * cell_size * 1.0))
    a = np.asarray(avgD)
    diff = []
    for i in range(len(a) - 1):
        for j in range(i + 1, len(avgD) - 1):
            diff.append((a[j] - a[i]))
    ldp = np.asarray(diff)
    if ldp.sum():
        hist, _ = np.histogram(ldp, density=True)
        mean = np.mean(ldp)
        std = np.std(ldp)
    else:
        hist = np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        mean = 0
        std = 0
    return ldp, hist, mean, std
