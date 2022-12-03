from uncertainties import ufloat
import uncertainties.unumpy as unp
import numpy as np

def to_quantity(measure, ureg):
    """Converts a measurement to a Quantity with a ufloat."""
    u = measure.units
    x = measure.magnitude
    return ureg.Quantity(ufloat(x.n, x.s), u)

def to_quantity_array(arr):
    """Converts an array of Measurements into a Quantity with a uarray."""
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
