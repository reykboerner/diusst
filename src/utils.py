import numpy as np

def _lapl_central(array, deriv1, deriv2):
    """
    1D Laplacian (second spatial derivative) as central finite differences on stretched mesh.

    Arguments
    ----------
    array   (1D np.array): input vector
    deriv1  (1D np.array): first stretched grid derivative evaluated at all grid points
    deriv2  (1D np.array): second stretched grid derivative evaluated at all grid points

    Input array must have len(array) = ngrid + 2 (ghost point at top and bottom).
    (See eqs. (5.29) and (5.30) of thesis.)
    """

    # Initialize output vector
    lapl = np.zeros(len(array))
    # use forward difference for surface point
    lapl[1] = (array[2]-array[1]) * (deriv1[1]**2 + 0.5 * deriv2[1])
    # use second order central difference for all other points
    lapl[2:-1] = (array[3:] - 2*array[2:-1] + array[1:-2]) * deriv1[2:-1]**2 + (array[3:]-array[1:-2])*0.5*deriv2[2:-1]

    return lapl

def _grad_central(array, deriv1):
    """
    1D gradient (first spatial derivative) as central finite differences on stretched mesh.

    Arguments
    ----------
    array   (1D np.array): input vector
    deriv1  (1D np.array): first stretched grid derivative evaluated at all grid points

    Input array must have len(array) = ngrid + 2 (ghost point at top and bottom).
    (See eqs. (5.29) and (5.30) of thesis.)
    """

    grad = np.zeros(len(array))
    grad[1:-1] = (array[:-2] - array[2:]) * deriv1[1:-1] / 2
    return grad

def _grad_bckward(array, deriv1):
    """
    1D gradient (first spatial derivative) as backward finite differences on stretched mesh.

    Arguments
    ----------
    array   (1D np.array): input vector
    deriv1  (1D np.array): first stretched grid derivative evaluated at all grid points

    Input array must have len(array) = ngrid + 2 (ghost point at top and bottom).
    (See eqs. (5.29) and (5.30) of thesis.)
    """

    grad = np.zeros(len(array))
    grad[1:-1] = (array[1:-1] - array[:-2]) * deriv1[1:-1]
    return grad
