import sympy as sp
from sympy.physics import units

def tex(expr):
    """Prints an expression in LaTeX form"""
    print(sp.latex(expr))

def unit_value(expr, unit=1):
    """Converts an expression to the given unit(s), and outputs the scaling factor associated with the resultant quantity."""
    expr = units.convert_to(expr, unit)
    return expr.as_coeff_Mul()[0]

def normalize(f, x, key, just_constant=False):
    """Normalizes a function f(x) such that evaluating the key function returns one. The normalization constant C is assumed to be positive.
    
    Examples
    ===========
    The key could be the norm on L^2; C is chosen so that f (e.g. a wavefunction) has unit norm.
    The key could be a sum of probabilities; C is chosen so that the probabilities (f) are normalized.
    """
    C = sp.symbols("C", positive=True)
    C = sp.solve(key(lambda x: C * f(x)) - 1, C)[0]
    if just_constant:
        return C
    return C * f(x)

def f_inner(f, g, bounds):
    """Computes the L^2 inner product between f and g over the bounds."""
    x = bounds[0]
    return sp.integrate(sp.conjugate(f(x)) * g(x), bounds)

def f_norm(f, bounds):
    """Computes the L^2 norm of f over the bounds."""
    return sp.sqrt(f_inner(f, f, bounds))

def convert_function(expr, variables):
    """Transforms an expression in terms of some variables into a function of said variables.
    
    Examples
    ===========
    >>> f = convert_function(x**2 - y**3, [x, y])
    >>> f(x, y)
    x**2 - y**3
    >>> f(z, 2)
    z**2 - 8
    >>> f(3, t)
    9 - t**3
    """
    def f(*new_vars):
        nonlocal expr
        old_len, new_len = len(variables), len(new_vars)
        if old_len != new_len:
            raise TypeError(f"Function takes {old_len} positional arguments but {new_len} were given")
        return expr.subs({x: x_new for x, x_new in zip(variables, new_vars)})
    return f
