"""
Example 1 for 1st order PDE IVP

Note that an upper-case 'Y' is used to represent the Greek psi from
the original equation.

The equation is defined on the domain [0,1].

The analytical form of the equation is:
    G((x, y), Y, (dY/dx, dY/dy)) = dY/dy + 2*dY/dx = 0

with initial condition:

Y((0, y)) = sin(y)
Y((x, 0)) = sin(-x/2)

This equation has the analytical solution for the supplied initial conditions:

Ya((x, y)) = -sin((x - 2*y)/2)

with analytical Jacobian:

dYa/dx = -cos((x - 2*y)/2)/2
dYa/dy = cos((x - 2*y)/2])

Reference:

This example is taken from Mathematica documentation at:
https://reference.wolfram.com/language/howto/SolveAPartialDifferentialEquation.html
"""


from math import cos, sin
import numpy as np


def G(xy, Y, delY):
    """Code for differential equation"""
    (x, y) = xy
    (dY_dx, dY_dy) = delY
    return dY_dy + 2*dY_dx


def f0(xy):
    """Initial condition at x=0"""
    (x, y) = xy
    return sin(y)


def f1(xy):
    """No boundary condition is specified at x=1."""
    return None


def g0(xy):
    """Initial condition at y=0"""
    (x, y) = xy
    return sin(-x/2)


def g1(xy):
    """No boundary condition is specified at y=1."""
    return None


bc = [[f0, f1], [g0, g1]]


def df0_dx(xy):
    """Derivatve of f(0, y) wrt x"""
    return 0


def df0_dy(xy):
    """Derivatve of f(0, y) wrt y"""
    (x, y) = xy
    return cos(y)


def df1_dx(xy):
    """No boundary condition is specified at x=1."""
    return None


def df1_dy(xy):
    """No boundary condition is specified at x=1."""
    return None


def dg0_dx(xy):
    """Derivative of g0(x, 0) wrt x"""
    (x, y) = xy
    return -cos(-x/2)/2


def dg0_dy(xy):
    """Derivative of g0(x, 0) wrt y"""
    return 0


def dg1_dx(xy):
    """No boundary condition is specified at y=1."""
    return None


def dg1_dy(xy):
    """No boundary condition is specified at y=1."""
    return None


delbc = [[[df0_dx, df0_dy], [df1_dx, df1_dy]],
         [[dg0_dx, dg0_dy], [dg1_dx, dg1_dy]]]


def dG_dY(xy, Y, delY):
    """Derivative of G wrt Y"""
    return 0


def dG_ddYdx(xy, Y, delY):
    """Derivative of G wrt dY/dx"""
    return 2


def dG_ddYdy(xy, Y, delY):
    """Derivative of G wrt dY/dy"""
    return 1


dG_ddelY = [dG_ddYdx, dG_ddYdy]


def Ya(xy):
    """Analytical solution"""
    (x, y) = xy
    return -sin((x - 2*y)/2)


def dYa_dx(xy):
    """Derivative of analytical solution wrt x"""
    (x, y) = xy
    return -cos((x - 2*y)/2)/2


def dYa_dy(xy):
    """Derivative of analytical solution wrt y"""
    (x, y) = xy
    return cos((x - 2*y)/2)


delYa = [dYa_dx, dYa_dy]


if __name__ == '__main__':

    # Values to use for testing.
    x_test = [0.4, 0.5]
    m = len(x_test)
    Y_test = 0.6
    delY_test = [0.7, 0.8]

    # Reference values for tests
    G_ref = 2.2
    bc_ref = [[0.479426, None], [-0.198669, None]]
    delbc_ref = [
        [[0, 0.877583], [None, None]],
        [[-0.490033, 0], [None, None]]
    ]
    dG_dY_ref = 0
    dG_ddelY_ref = [2, 1]
    Ya_ref = 0.29552
    delYa_ref = [-0.477668, 0.955336]

    print('Testing differential equation.')
    assert np.isclose(G(x_test, Y_test, delY_test), G_ref)

    print('Testing boundary conditions.')
    for j in range(m):
        assert np.isclose(bc[j][0](x_test), bc_ref[j][0])
        assert bc[j][1](x_test) is bc_ref[j][1]  # None since not used.

    print('Testing boundary condition gradients.')
    for j in range(m):
        for jj in range(m):
            assert np.isclose(delbc[j][0][jj](x_test), delbc_ref[j][0][jj])
            assert delbc[j][1][jj](x_test) is delbc_ref[j][1][jj]  # None

    print('Testing derivative of differential equation wrt solution.')
    assert np.isclose(dG_dY(x_test, Y_test, delY_test), dG_dY_ref)

    print('Testing derivative of differential equation wrt gradient '
          'components.')
    for j in range(m):
        assert np.isclose(dG_ddelY[j](x_test, Y_test, delY_test),
                          dG_ddelY_ref[j])

    print('Testing analytical solution.')
    assert np.isclose(Ya(x_test), Ya_ref)

    print('Testing analytical gradient.')
    for j in range(m):
        assert np.isclose(delYa[j](x_test), delYa_ref[j])
