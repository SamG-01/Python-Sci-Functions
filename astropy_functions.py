import numpy as np
#from astropy import units as u
from astropy import uncertainty as unc

def measure(n, s, n_samples=10000):
    """Shortcut for a measurement n with error s."""
    if type(n) == np.ndarray:
        if not type(s) == np.ndarray:
            s = np.array([s] * len(n))
        return np.array(
            [unc.normal(N, std=S, n_samples=n_samples) for N, S in zip(n, s)], dtype=object
        )
    return unc.normal(n, std=s, n_samples=n_samples)

def units(arr):
    """Returns the units of each QuantityDistribution in an array."""
    return np.array([x.unit for x in arr])

def values(arr):
    """Returns the values of each QuantityDistribution in an array."""
    return np.array([x/u for x,u in zip(arr, units(arr))])
