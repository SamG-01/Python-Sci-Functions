"""
In the Hamiltonian formalism of classical mechanics, the conjugate momentum for each generalized coordinate q_k(t) is given by p_k(t) := ∂L/(∂q_k'(t)), where L is the Lagrangian of the System. The Hamiltonian is then the Legendre Transform of the Lagrangian L(q, q', t); that is, H(q, p, t) := p • q' - L(q, p, t), where q and p are the collections of each q_k and p_k. Hamilton's Equations are then given by ∂H/∂p_k = q_k' and ∂H/∂q_k = -p_k' for each k, and describe the equations of motion of the system.
"""
import sympy as sp
t = sp.Symbol("t")

def conjugate_momenta(L, q):
    """Returns the list of conjugate momenta given L(q, q', t) and q."""
    return { sp.Function("p_{}".format(q_k))(t): sp.diff(L, q_k(t).diff(t)) for k, q_k in enumerate(q) }

def hamiltonian(L, q, return_p=True):
    """Returns the Hamiltonian H(q, p, t). Also returns p if return_p is True."""
    p = conjugate_momenta(L, q)

    p_eq = [p_k - p_k_val for p_k, p_k_val in p.items()]
    temp = [ sp.solve(p_eq[k], q_k(t).diff(t), dict=True)[0] for k, q_k in enumerate(q) ]
    q_dot = { k: v for solution in temp for k,v in solution.items() }

    H = -L
    for q_k_dot, p_k in zip(q_dot, p): H = p_k*q_dot[q_k_dot] + H.subs(q_k_dot, q_dot[q_k_dot])
    H = sp.simplify(H)

    # if you are using conjugate_momenta(L, q) separately, you can let return_p be false. otherwise, keep it as true
    if return_p:
        return H, list(p.keys())
    return H

def hamilton_equations(H, p, q):
    """Returns Hamilton's equations for each q_k, p_k given H, p, and q."""
    eqs = []
    for p_k, q_k in zip(p, [q_k(t) for q_k in q]):
        eqs += [sp.Eq( q_k.diff(t) , H.diff(p_k) ), sp.Eq( p_k.diff(t), -H.diff(q_k) )]
    return eqs

def poisson_bracket(f, g, p, q):
    """Evaluates the Poisson bracket of two functions f, g."""
    return sum(f.diff(q_k) * g.diff(p_k) - f.diff(p_k) * g.diff(q_k) for q_k, p_k in zip(q, p))

"""Tests to try

m, l, omega_g = sp.symbols("m l omega_g")
phi, theta = sp.Function("phi"), sp.Function("theta")

# Polar Pendulum
L = m/2 * l**2 * phi(t).diff(t)**2 + m*omega_g**2 *sp.cos(phi(t))
q = [phi]

# Spherical Pendulum
#L = m/2 * l**2 * (theta(t).diff(t)**2 + phi(t).diff(t)**2 * sp.sin(theta(t))**2) - m * l*omega_g**2 * l * sp.cos(theta(t))
#q = [theta, phi]

sp.pprint( conjugate_momenta(L, q) )
sp.pprint( H := hamiltonian(L, q) )
sp.pprint( hamilton_equations(*H, q) )
"""
