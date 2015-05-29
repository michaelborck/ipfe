import scipy.optimize
import numpy as np
import math


def fit_plane_LTSQ(XYZ):
    """
    Using np.linalg.lstsq
    Fits a plane to a point cloud,
    Where Z = aX + bY + c   --- #1
    Rearanging #1 gives: aX + bY -Z +c =0
    Normal = (a,b,-1)
    """
    [rows, cols] = XYZ.shape
    G = np.ones((rows, 3))
    G[:, 0] = XYZ[:, 0]  # X
    G[:, 1] = XYZ[:, 1]  # Y
    Z = XYZ[:, 2]
    (a, b, c), resid, rank, s = np.linalg.lstsq(G, Z)
    return unit_vector((a, b, -1))


def fit_plane_SVD(XYZ):
    """
    Fit a plane using np.linalg.svd
    Set up constraint equations of the form  AB = 0,
    where B is a column vector of the plane coefficients
    in the form b(1)*X + b(2)*Y +b(3)*Z + b(4) = 0.
    """
    [rows, cols] = XYZ.shape
    p = (np.ones((rows, 1)))
    AB = np.hstack([XYZ, p])
    [u, d, v] = np.linalg.svd(AB, 0)
    return unit_vector(v[3, :][0:3])     # Solution is last column of v.


def fit_plane_Eigen(XYZ):
    """
    Fit a plane using np.linalg.eig
    """
    average = sum(XYZ) / XYZ.shape[0]
    covariant = np.cov(XYZ - average)
    eigenvalues, eigenvectors = np.linalg.eig(covariant)
    want = eigenvectors[:, eigenvalues.argmax()]
    (c, a, b) = want[3:6]  # Do not understand! Why [3:6]? Why (c,a,b)?
    return unit_vector(np.array([a, b, c]))


def fit_plane_Solve(XYZ):
    """
    Fit a plane using np.linalg.solve
    "Ordinary" Least Squares
    """
    X = XYZ[:, 0]
    Y = XYZ[:, 1]
    Z = XYZ[:, 2]
    npts = len(X)
    A = np.array([[sum(X * X), sum(X * Y), sum(X)],
                  [sum(X * Y), sum(Y * Y), sum(Y)],
                  [sum(X), sum(Y), npts]])
    B = np.array([[sum(X * Z), sum(Y * Z), sum(Z)]])
    return unit_vector(np.linalg.solve(A, B.T).ravel())


def fit_plane_Optimize(XYZ):
    """
    Fit a plane using scipy.optimse.leastsq
    """
    def residiuals(parameter, f, x, y):
        return [(f[i] - model(parameter, x[i], y[i])) for i in range(len(f))]

    def model(parameter, x, y):
        a, b, c = parameter
        return a * x + b * y + c

    X = XYZ[:, 0]
    Y = XYZ[:, 1]
    Z = XYZ[:, 2]
    p0 = [1., 1., 1.]  # initial guess
    return unit_vector(scipy.optimize.leastsq(residiuals,
                                              p0, args=(Z, X, Y))[0][0:3])


def unit_vector(u):
    """
    Normalise vector u
    """
    norm = np.linalg.norm(u)
    if norm > 0:
        return u / norm
    return u


def angle(u, v):
    """
    Calculate the anger between two vectors
    """
    u_u = unit_vector(u)
    v_u = unit_vector(v)
    angle = math.acos(np.dot(u_u, v_u))
    if math.isnan(angle):   # if not a number then....
        if (u_u == v_u).all():
            angle = 0.0
        else:
            angle = np.pi
    return angle


def _alpha_wrtX(u):
    """
    Internal function to calculate alpha angle
    """
    a, b, c = u
    return math.atan(c / a)


def _gamma_wrtY(u):
    """
    Internal function to calculate gamma angle
    """
    a, b, c = u
    return math.atan(c / b)
