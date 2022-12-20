import sympy as sp
from sympy.physics.units import convert_to

def tex(expr):
    """Prints an expression in LaTeX form."""
    print(sp.latex(expr))

def unit_value(quantity, units=1, return_float=False):
    """Returns the numerical value of a quantity in the given units."""
    n = convert_to(quantity, units)/units
    if return_float:
        n = float(n)
    return n

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
