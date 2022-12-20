import numpy as np
import uncertainties as unc
#from funcy import compose
from scipy.integrate import quad, nquad
#from quadpy import quad
from pandas import DataFrame

def ufloat_from_dist(arr, correlated=False):
    """Generates a ufloat (or uarray) from a distribution (or array of distributions)."""
    axis = np.ndim(arr) - 1

    means = np.mean(arr, axis=axis)

    if axis == 0:
        return unc.unumpy.uarray(means, np.std(arr, axis=axis)) * 1

    if correlated:
        cov = np.cov(arr)
        return np.array(unc.correlated_values(means, cov))
    
    return unc.unumpy.uarray(means, np.std(arr, axis=axis))

def dist_from_ufloat(ufloats, n_samples=10000):
    """Generates a normal distribution (or array of distributions) from a ufloat (or an array-like of ufloats)."""
    # Handles the case of ufloats being a single ufloat, or an array-like with just one of them
    if np.size(ufloats) == 1:
        x = np.reshape(ufloats, 1).item()
        return np.random.normal(x.n, x.s, n_samples)

    # Otherwise, handles correlation between these
    means = unc.unumpy.nominal_values(ufloats)
    cov_matrix = unc.covariance_matrix(ufloats)
    dists = np.random.multivariate_normal(means, cov_matrix, n_samples).T
    return dists

def quad_v(func, **kwargs):
    """
    Integrates func using scipy.integrate.quad when kwargs are vectorized.
    
    Arguments:
    func {function} -- the function to integrate.
    kwargs {dict} -- a dictionary of kwargs whose values are array-likes.

    Example Usage:
    quad_v(lambda x: x**2, a=[1, 2, 3], b=[4, 5, 6])
    """

    vec_length = len(list(kwargs.values())[0])

    kwargs = DataFrame(kwargs).to_dict('records')

    int_vals = np.zeros(vec_length)
    #int_errs = np.zeros(vec_length)

    for i in range(vec_length):
        int_val, _ = quad(func, **kwargs[i])
        int_vals[i] = int_val
        #int_errs[i] = int_err
    return int_vals#, int_errs

def nquad_v(func, **kwargs):
    """
    Integrates func using scipy.integrate.nquad when kwargs are vectorized.
    
    Arguments:
    func {function} -- the function to integrate.
    kwargs {dict} -- a dictionary of kwargs whose values are array-likes.

    Example Usage:
    nquad_v(lambda x, y: x * y, ranges=[[(0, 1), (1, 2)], [(1, 3), (0, 4)]])
    """

    vec_length = len(list(kwargs.values())[0])

    kwargs = DataFrame(kwargs).to_dict('records')

    int_vals = np.zeros(vec_length)
    #int_errs = np.zeros(vec_length)

    for i in range(vec_length):
        int_val, _ = nquad(func, **kwargs[i])
        int_vals[i] = int_val
        #int_errs[i] = int_err
    return int_vals#, int_errs
