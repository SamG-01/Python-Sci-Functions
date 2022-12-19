import numpy as np
import uncertainties as unc
#from funcy import compose
from scipy.integrate import quad, nquad
#from quadpy import quad
from pandas import DataFrame

def dist_ufloat(arr):
    """Converts an array_like arr of dimension 1 or 2 to its mean(s) and uncertainty(s)."""
    axis = np.ndim(arr) - 1

    result = unc.unumpy.uarray(
        np.mean(arr, axis=axis),
        np.std(arr, axis=axis)    
    )
    
    if np.ndim(result) == 0:
        result = result.item()
    return result

def normal_distribution(ufloat, n_samples=10000):
    """Converts a ufloat to the equivalent normal distribution."""
    return np.random.normal(ufloat.n, ufloat.s, n_samples)

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
