"""Simple 1st order PDE IVP

Note that an upper-case 'Y' is used to represent the Greek psi from
the original equation.

The equation is defined on the domain [0, 1].

The analytical form of the equation is:
    G(x, y, Y, dY/dx, dY/dy) = dY/dx + dY/dy - x - y = 0

with initial conditions:

Y(0, y) = 0
Y(x, 0) = 0

This equation has the analytical solution for the supplied initial conditions:

Ya(x, y) = x*y

with analytical partial derivatives:

dYa/dx = y
dYa/dy = x
"""


def G(xy, Y, delY):
    """Code for differential equation"""
    (x, y) = xy
    (dY_dx, dY_dy) = delY
    return dY_dx + dY_dy - x - y


def f0(xy):
    """Initial condition at x=0"""
    return 0


def g0(xy):
    """Initial condition at y=0"""
    return 0


bc = [[f0, None], [g0, None]]


def df0_dx(xy):
    """Derivatve of f(0, y) wrt x"""
    return 0


def df0_dy(xy):
    """Derivatve of f(0, y) wrt y"""
    return 0


def dg0_dx(xy):
    """Derivative of g0(x, 0) wrt x"""
    return 0


def dg0_dy(xy):
    """Derivative of g0(x, 0) wrt y"""
    return 0


delbc = [[[df0_dx, df0_dy], [None, None]],
         [[dg0_dx, dg0_dy], [None, None]]]


def dG_dY(xy, Y, delY):
    """Derivative of G wrt Y"""
    return 0


def dG_ddYdx(xy, Y, delY):
    """Derivative of G wrt dY/dx"""
    return 1


def dG_ddYdy(xy, Y, delY):
    """Derivative of G wrt dY/dy"""
    return 1


dG_ddelY = [dG_ddYdx, dG_ddYdy]


def Ya(xy):
    """Analytical solution"""
    (x, y) = xy
    return x*y


def dYa_dx(xy):
    """Derivative of analytical solution wrt x"""
    (x, y) = xy
    return y


def dYa_dy(xy):
    """Derivative of analytical solution wrt y"""
    (x, y) = xy
    return x


delYa = [dYa_dx, dYa_dy]
