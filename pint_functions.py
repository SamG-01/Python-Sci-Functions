"""
Functions for the pint and uncertainties packages.
"""

import numpy as np
from scipy.stats import norm

import uncertainties as unc
import uncertainties.unumpy as unp

def to_quantity(measure, ureg):
    """Converts a measurement to a Quantity with a ufloat."""
    u = measure.units
    x = measure.magnitude
    return ureg.Quantity(unc.ufloat(x.n, x.s), u)

def to_quantity_array(arr):
    """Converts an array of Measurements into a Quantity with a uarray as its magnitude."""
    mag = np.array([x.magnitude for x in arr])
    units = [x.units for x in arr]

    if len(set(units)) != 1:
        raise ValueError("Units of arr do not match")
    return mag * units[0]

def values(arr):
    """Returns the values of a Quantity whose magnitude is an array of ufloats."""
    return unp.nominal_values(arr) * arr.units

def errors(arr):
    """Returns the values of a Quantity whose magnitude is an array of ufloats."""
    return unp.std_devs(arr) * arr.units

def split_measurement(arr, units=True):
    """Splits an array of ufloats (possibly with units) into its values and errors."""
    if units:
        return values(arr), errors(arr)
    return unp.nominal_values(arr), unp.std_devs(arr)

def confidence_interval(arr, units=False, confidence=0.99, means=False):
    """Returns the lower and upper bounds of a Quantity array within a given confidence interval."""
    z = norm.ppf(1-(1-confidence)/2)
    x, dx = split_measurement(arr, units)
    output = [x - z*dx, x + z*dx]
    if means:
        output.append(x)
    return output
