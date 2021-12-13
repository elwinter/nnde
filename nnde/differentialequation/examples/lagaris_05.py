from math import exp


def G(xy, Y, delY, del2Y):
    (x, y) = xy
    (d2Y_dx2, d2Y_dy2) = del2Y
    return d2Y_dx2 + d2Y_dy2 - exp(-x)*(x - 2 + y**3 + 6*y)


def f0(xy):
    (x, y) = xy
    return y**3


def f1(xy):
    (x, y) = xy
    return (1 + y**3)*exp(-1)


def g0(xy):
    (x, y) = xy
    return x*exp(-x)


def g1(xy):
    (x, y) = xy
    return exp(-x)*(x + 1)


# Gather the boundary condition functions in a single array.
bc = [[f0, f1], [g0, g1]]


def df0_dx(xy):
    return 0


def df0_dy(xy):
    (x, y) = xy
    return 3*y^2


def df1_dx(xy):
    return 0


def df1_dy(xy):
    (x, y) = xy
    return 3*y**2*exp(-1)


def dg0_dx(xy):
    (x, y) = xy
    return exp(-x)*(1 - x)


def dg0_dy(xy):
    return 0


def dg1_dx(xy):
    (x, y) = xy
    return -x*exp(-x)


def dg1_dy(xy):
    return 0


# Gather the gradient functions into a single array.
delbc = [[[df0_dx, df0_dy], [df1_dx, df1_dy]],
         [[dg0_dx, dg0_dy], [dg1_dx, dg1_dy]]]


def d2f0_dx2(xy):
    return 0


def d2f0_dy2(xy):
    (x, y) = xy
    return 6*y


def d2f1_dx2(xy):
    return 0


def d2f1_dy2(xy):
    (x, y) = xy
    return 6*y*exp(-1)


def d2g0_dx2(xy):
    (x, y) = xy
    return exp(-x)*(x - 2)


def d2g0_dy2(xy):
    return 0


def d2g1_dx2(xy):
    (x, y) = xy
    return exp(-x)*(x - 1)


def d2g1_dy2(xy):
    return 0


# Gather the functions for the Laplacian components into a single array.
del2bc = [[[d2f0_dx2, d2f0_dy2], [d2f1_dx2, d2f1_dy2]],
          [[d2g0_dx2, d2g0_dy2], [d2g1_dx2, d2g1_dy2]]]


def dG_dY(xy, Y, delY, del2Y):
    return 0


# NOTE: Should be named dG_ddY_dx() but that name will not import!
def dG_dY_dx(xy, Y, delY, del2Y):
    return 0


# NOTE: Should be named dG_ddY_dt() but that name will not import!
def dG_dY_dy(xy, Y, delY, del2Y):
    return 0


# Gather the derivatives into a single array.
dG_ddelY = [dG_dY_dx, dG_dY_dy]


# NOTE: Should be named dG_dd2Y_dx2() but that name will not import!
def dG_d2Y_dx2(xy, Y, delY, del2Y):
    return 1


# NOTE: Should be named dG_dd2Y_dt2() but that name will not import!
def dG_d2Y_dy2(xy, Y, delY, del2Y):
    return 1


# Gather the derivatives into a single array.
dG_ddel2Y = [dG_d2Y_dx2, dG_d2Y_dy2]


def A(xy):
    (x, y) = xy
    return (
        (1 - x)*y**3
        + x*(1 + y**3)*exp(-1)
        + (1 - y)*x*(exp(-x) - exp(-1))
        + y*(exp(-x)*(x + 1) - (1 - x + 2*x*exp(-1)))
    )


def delA(xy):
    (x, y) = xy
    dA_dx = y - y**3 - exp(-x)*(x + y - 1) + y*(y**2 - 1)*exp(-1)
    dA_dy = exp(-x) - (x - 1)*(3*y**2 - 1) + x*(3*y**2 - 1)*exp(-1)
    return [dA_dx, dA_dy]


def del2A(xy):
    (x, y) = xy
    d2A_dx2 = exp(-x)*(x + y - 2)
    d2A_dy2 = 6*(exp(1)*(1 - x) + x)*y*exp(-1)
    return [d2A_dx2, d2A_dy2]


def Ya(xy):
    (x, y) = xy
    return exp(-x)*(x + y**3)


def dYa_dx(xy):
    (x, y) = xy
    return -exp(-x)*(x + y**3 - 1)


def dYa_dy(xy):
    (x, y) = xy
    return 3*exp(-x)*y**2


# Gather the analytical gradient functions into a single array.
delYa = [dYa_dx, dYa_dy]


def d2Ya_dx2(xy):
    (x, y) = xy
    return exp(-x)*(x + y**3 - 2)


def d2Ya_dy2(xy):
    (x, y) = xy
    return 6*exp(-x)*y


# Gather the Laplacian component functions into a single array.
del2Ya = [d2Ya_dx2, d2Ya_dy2]
