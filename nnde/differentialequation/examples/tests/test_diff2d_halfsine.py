"""Tests for the diff2d_halfsine module."""


from math import cos, exp, pi, sin
import unittest

import numpy as np

from nnde.differentialequation.examples.diff2d_halfsine import (
    D,
    G,
    f0, f1, g0, g1, Y0, bc,
    df0_dx, df0_dy, df0_dt, df1_dx, df1_dy, df1_dt,
    dg0_dx, dg0_dy, dg0_dt, dg1_dx, dg1_dy, dg1_dt,
    dY0_dx, dY0_dy, dY0_dt, delbc,
    d2f0_dx2, d2f0_dy2, d2f0_dt2, d2f1_dx2, d2f1_dy2, d2f1_dt2,
    d2g0_dx2, d2g0_dy2, d2g0_dt2, d2g1_dx2, d2g1_dy2, d2g1_dt2,
    d2Y0_dx2, d2Y0_dy2, d2Y0_dt2, del2bc,
    dG_ddY_dx, dG_ddY_dy, dG_ddY_dt, dG_ddelY,
    dG_dd2Y_dx2, dG_dd2Y_dy2, dG_dd2Y_dt2, dG_ddel2Y,
    A, delA, del2A,
    Ya,
    dYa_dx, dYa_dy, dYa_dt, delYa,
    d2Ya_dx2, d2Ya_dy2, d2Ya_dt2, del2Ya,
)


# Grid points for testing.
xx = np.linspace(0, 1, 11)
yy = np.linspace(0, 1, 11)
tt = np.linspace(0, 1, 11)


class TestBuilder(unittest.TestCase):
    """Tests for the diff2d_halfsine_increase module."""

    def test_G(self):
        """Test the differential equation."""
        Yt = 2
        delYt = (3, 4, 5)
        del2Yt = (5, 6, 7)
        for x in xx:
            for y in yy:
                for t in tt:
                    xyt = (x, y, t)
                    G_ref = delYt[2] - D*(del2Yt[0] + del2Yt[1])
                    self.assertAlmostEqual(G(xyt, Yt, delYt, del2Yt), G_ref)

    def test_f0(self):
        """Test boundary condition at (x, y, t) = (0, y, t)."""
        f0_ref = 0  # For all inputs.
        for x in xx:
            for y in yy:
                for t in tt:
                    xyt = (x, y, t)
                    self.assertAlmostEqual(f0(xyt), f0_ref)

    def test_f1(self):
        """Test boundary condition at (x, y, t) = (1, y, t)."""
        f1_ref = 0  # For all inputs.
        for x in xx:
            for y in yy:
                for t in tt:
                    xyt = (x, y, t)
                    self.assertAlmostEqual(f1(xyt), f1_ref)

    def test_g0(self):
        """Test boundary condition at (x, y, t) = (x, 0, t)."""
        g0_ref = 0  # For all inputs.
        for x in xx:
            for y in yy:
                for t in tt:
                    xyt = (x, y, t)
                    self.assertAlmostEqual(g0(xyt), g0_ref)

    def test_g1(self):
        """Test boundary condition at (x, y, t) = (x, 1, t)."""
        g1_ref = 0  # For all inputs.
        for x in xx:
            for y in yy:
                for t in tt:
                    xyt = (x, y, t)
                    self.assertAlmostEqual(g1(xyt), g1_ref)

    def test_Y0(self):
        """Test boundary condition at (x, y, t) = (x, y, 0)."""
        for x in xx:
            for y in yy:
                for t in tt:
                    xyt = (x, y, t)
                    Y0_ref = sin(pi*x)*sin(pi*y)
                    self.assertAlmostEqual(Y0(xyt), Y0_ref)

    def test_bc(self):
        """Test that the boundary conditions are continuous at corners."""
        xyt = (0, 0, 0)
        self.assertAlmostEqual(bc[0][0](xyt), bc[1][0](xyt))
        self.assertAlmostEqual(bc[0][0](xyt), bc[2][0](xyt))
        xyt = (1, 0, 0)
        self.assertAlmostEqual(bc[0][1](xyt), bc[1][0](xyt))
        self.assertAlmostEqual(bc[0][1](xyt), bc[2][0](xyt))
        xyt = (0, 1, 0)
        self.assertAlmostEqual(bc[0][0](xyt), bc[1][1](xyt))
        self.assertAlmostEqual(bc[0][0](xyt), bc[2][0](xyt))
        xyt = (1, 1, 0)
        self.assertAlmostEqual(bc[0][1](xyt), bc[1][1](xyt))
        self.assertAlmostEqual(bc[0][1](xyt), bc[2][0](xyt))

    def test_df0_dx(self):
        """Test the first derivative of f0 wrt x."""
        df0_dx_ref = 0  # For all inputs.
        for x in xx:
            for y in yy:
                for t in tt:
                    xyt = (x, y, t)
                    self.assertAlmostEqual(df0_dx(xyt), df0_dx_ref)

    def test_df0_dy(self):
        """Test the first derivative of f0 wrt y."""
        df0_dy_ref = 0  # For all inputs.
        for x in xx:
            for y in yy:
                for t in tt:
                    xyt = (x, y, t)
                    self.assertAlmostEqual(df0_dy(xyt), df0_dy_ref)

    def test_df0_dt(self):
        """Test the first derivative of f0 wrt t."""
        df0_dt_ref = 0  # For all inputs.
        for x in xx:
            for y in yy:
                for t in tt:
                    xyt = (x, y, t)
                    self.assertAlmostEqual(df0_dt(xyt), df0_dt_ref)

    def test_df1_dx(self):
        """Test the first derivative of f1 wrt x."""
        df1_dx_ref = 0  # For all inputs.
        for x in xx:
            for y in yy:
                for t in tt:
                    xyt = (x, y, t)
                    self.assertAlmostEqual(df1_dx(xyt), df1_dx_ref)

    def test_df1_dy(self):
        """Test the first derivative of f1 wrt y."""
        df1_dy_ref = 0  # For all inputs.
        for x in xx:
            for y in yy:
                for t in tt:
                    xyt = (x, y, t)
                    self.assertAlmostEqual(df1_dy(xyt), df1_dy_ref)

    def test_df1_dt(self):
        """Test the first derivative of f1 wrt t."""
        df1_dt_ref = 0  # For all inputs.
        for x in xx:
            for y in yy:
                for t in tt:
                    xyt = (x, y, t)
                    self.assertAlmostEqual(df1_dt(xyt), df1_dt_ref)

    def test_dg0_dx(self):
        """Test the first derivative of g0 wrt x."""
        dg0_dx_ref = 0  # For all inputs.
        for x in xx:
            for y in yy:
                for t in tt:
                    xyt = (x, y, t)
                    self.assertAlmostEqual(dg0_dx(xyt), dg0_dx_ref)

    def test_dg0_dy(self):
        """Test the first derivative of g0 wrt y."""
        dg0_dy_ref = 0  # For all inputs.
        for x in xx:
            for y in yy:
                for t in tt:
                    xyt = (x, y, t)
                    self.assertAlmostEqual(dg0_dy(xyt), dg0_dy_ref)

    def test_dg0_dt(self):
        """Test the first derivative of g0 wrt t."""
        dg0_dt_ref = 0  # For all inputs.
        for x in xx:
            for y in yy:
                for t in tt:
                    xyt = (x, y, t)
                    self.assertAlmostEqual(dg0_dt(xyt), dg0_dt_ref)

    def test_dg1_dx(self):
        """Test the first derivative of g1 wrt x."""
        dg1_dx_ref = 0  # For all inputs.
        for x in xx:
            for y in yy:
                for t in tt:
                    xyt = (x, y, t)
                    self.assertAlmostEqual(dg1_dx(xyt), dg1_dx_ref)

    def test_dg1_dy(self):
        """Test the first derivative of g1 wrt y."""
        dg1_dy_ref = 0  # For all inputs.
        for x in xx:
            for y in yy:
                for t in tt:
                    xyt = (x, y, t)
                    self.assertAlmostEqual(dg1_dy(xyt), dg1_dy_ref)

    def test_dg1_dt(self):
        """Test the first derivative of g1 wrt t."""
        dg1_dt_ref = 0  # For all inputs.
        for x in xx:
            for y in yy:
                for t in tt:
                    xyt = (x, y, t)
                    self.assertAlmostEqual(dg1_dt(xyt), dg1_dt_ref)

    def test_dY0_dx(self):
        """Test the first derivative of Y0 wrt x."""
        for x in xx:
            for y in yy:
                for t in tt:
                    xyt = (x, y, t)
                    dY0_dx_ref = pi*cos(pi*x)*sin(pi*y)
                    self.assertAlmostEqual(dY0_dx(xyt), dY0_dx_ref)

    def test_dY0_dy(self):
        """Test the first derivative of Y0 wrt y."""
        for x in xx:
            for y in yy:
                for t in tt:
                    xyt = (x, y, t)
                    dY0_dy_ref = pi*sin(pi*x)*cos(pi*y)
                    self.assertAlmostEqual(dY0_dy(xyt), dY0_dy_ref)

    def test_dY0_dt(self):
        """Test the first derivative of Y0 wrt t."""
        dY0_dt_ref = 0  # For all inputs.
        for x in xx:
            for y in yy:
                for t in tt:
                    xyt = (x, y, t)
                    self.assertAlmostEqual(dY0_dt(xyt), dY0_dt_ref)

    def test_delbc(self):
        """Test the boundary condition gradients as an array."""
        self.assertIs(delbc[0][0][0], df0_dx)
        self.assertIs(delbc[0][0][1], df0_dy)
        self.assertIs(delbc[0][0][2], df0_dt)
        self.assertIs(delbc[0][1][0], df1_dx)
        self.assertIs(delbc[0][1][1], df1_dy)
        self.assertIs(delbc[0][1][2], df1_dt)
        self.assertIs(delbc[1][0][0], dg0_dx)
        self.assertIs(delbc[1][0][1], dg0_dy)
        self.assertIs(delbc[1][0][2], dg0_dt)
        self.assertIs(delbc[1][1][0], dg1_dx)
        self.assertIs(delbc[1][1][1], dg1_dy)
        self.assertIs(delbc[1][1][2], dg1_dt)
        self.assertIs(delbc[2][0][0], dY0_dx)
        self.assertIs(delbc[2][0][1], dY0_dy)
        self.assertIs(delbc[2][0][2], dY0_dt)

    def test_d2f0_dx2(self):
        """Test the 2nd derivative of f0 wrt x."""
        d2f0_dx2_ref = 0  # For all inputs.
        for x in xx:
            for y in yy:
                for t in tt:
                    xyt = (x, y, t)
                    self.assertAlmostEqual(d2f0_dx2(xyt), d2f0_dx2_ref)

    def test_d2f0_dy2(self):
        """Test the 2nd derivative of f0 wrt y."""
        d2f0_dy2_ref = 0  # For all inputs.
        for x in xx:
            for y in yy:
                for t in tt:
                    xyt = (x, y, t)
                    self.assertAlmostEqual(d2f0_dy2(xyt), d2f0_dy2_ref)

    def test_d2f0_dt2(self):
        """Test the 2nd derivative of f0 wrt t."""
        d2f0_dt2_ref = 0  # For all inputs.
        for x in xx:
            for y in yy:
                for t in tt:
                    xyt = (x, y, t)
                    self.assertAlmostEqual(d2f0_dt2(xyt), d2f0_dt2_ref)

    def test_d2f1_dx2(self):
        """Test the 2nd derivative of f1 wrt x."""
        d2f1_dx2_ref = 0  # For all inputs.
        for x in xx:
            for y in yy:
                for t in tt:
                    xyt = (x, y, t)
                    self.assertAlmostEqual(d2f1_dx2(xyt), d2f1_dx2_ref)

    def test_d2f1_dy2(self):
        """Test the 2nd derivative of f1 wrt y."""
        d2f1_dy2_ref = 0  # For all inputs.
        for x in xx:
            for y in yy:
                for t in tt:
                    xyt = (x, y, t)
                    self.assertAlmostEqual(d2f1_dy2(xyt), d2f1_dy2_ref)

    def test_d2f1_dt2(self):
        """Test the 2nd derivative of f1 wrt t."""
        d2f1_dt2_ref = 0  # For all inputs.
        for x in xx:
            for y in yy:
                for t in tt:
                    xyt = (x, y, t)
                    self.assertAlmostEqual(d2f1_dt2(xyt), d2f1_dt2_ref)

    def test_d2g0_dx2(self):
        """Test the 2nd derivative of g0 wrt x."""
        d2g0_dx2_ref = 0  # For all inputs.
        for x in xx:
            for y in yy:
                for t in tt:
                    xyt = (x, y, t)
                    self.assertAlmostEqual(d2g0_dx2(xyt), d2g0_dx2_ref)

    def test_d2g0_dy2(self):
        """Test the 2nd derivative of g0 wrt y."""
        d2g0_dy2_ref = 0  # For all inputs.
        for x in xx:
            for y in yy:
                for t in tt:
                    xyt = (x, y, t)
                    self.assertAlmostEqual(d2g0_dy2(xyt), d2g0_dy2_ref)

    def test_d2g0_dt2(self):
        """Test the 2nd derivative of g0 wrt t."""
        d2g0_dt2_ref = 0  # For all inputs.
        for x in xx:
            for y in yy:
                for t in tt:
                    xyt = (x, y, t)
                    self.assertAlmostEqual(d2g0_dt2(xyt), d2g0_dt2_ref)

    def test_d2g1_dx2(self):
        """Test the 2nd derivative of g1 wrt x."""
        d2g1_dx2_ref = 0  # For all inputs.
        for x in xx:
            for y in yy:
                for t in tt:
                    xyt = (x, y, t)
                    self.assertAlmostEqual(d2g1_dx2(xyt), d2g1_dx2_ref)

    def test_d2g1_dy2(self):
        """Test the 2nd derivative of g1 wrt y."""
        d2g1_dy2_ref = 0  # For all inputs.
        for x in xx:
            for y in yy:
                for t in tt:
                    xyt = (x, y, t)
                    self.assertAlmostEqual(d2g1_dy2(xyt), d2g1_dy2_ref)

    def test_d2g1_dt2(self):
        """Test the 2nd derivative of g1 wrt t."""
        d2g1_dt2_ref = 0  # For all inputs.
        for x in xx:
            for y in yy:
                for t in tt:
                    xyt = (x, y, t)
                    self.assertAlmostEqual(d2g1_dt2(xyt), d2g1_dt2_ref)

    def test_d2Y0_dx2(self):
        """Test the 2nd derivative of Y0 wrt x."""
        for x in xx:
            for y in yy:
                for t in tt:
                    xyt = (x, y, t)
                    d2Y0_dx2_ref = -pi**2*sin(pi*x)*sin(pi*y)
                    self.assertAlmostEqual(d2Y0_dx2(xyt), d2Y0_dx2_ref)

    def test_d2Y0_dy2(self):
        """Test the 2nd derivative of Y0 wrt y."""
        for x in xx:
            for y in yy:
                for t in tt:
                    xyt = (x, y, t)
                    d2Y0_dy2_ref = -pi**2*sin(pi*x)*sin(pi*y)
                    self.assertAlmostEqual(d2Y0_dy2(xyt), d2Y0_dy2_ref)

    def test_d2Y0_dt2(self):
        """Test the 2nd derivative of Y0 wrt t."""
        d2Y0_dt2_ref = 0  # For all inputs.
        for x in xx:
            for y in yy:
                for t in tt:
                    xyt = (x, y, t)
                    self.assertAlmostEqual(d2Y0_dt2(xyt), d2Y0_dt2_ref)

    def test_del2bc(self):
        """Test the boundary condition Laplacian as an array."""
        self.assertIs(del2bc[0][0][0], d2f0_dx2)
        self.assertIs(del2bc[0][0][1], d2f0_dy2)
        self.assertIs(del2bc[0][0][2], d2f0_dt2)
        self.assertIs(del2bc[0][1][0], d2f1_dx2)
        self.assertIs(del2bc[0][1][1], d2f1_dy2)
        self.assertIs(del2bc[0][1][2], d2f1_dt2)
        self.assertIs(del2bc[1][0][0], d2g0_dx2)
        self.assertIs(del2bc[1][0][1], d2g0_dy2)
        self.assertIs(del2bc[1][0][2], d2g0_dt2)
        self.assertIs(del2bc[1][1][0], d2g1_dx2)
        self.assertIs(del2bc[1][1][1], d2g1_dy2)
        self.assertIs(del2bc[1][1][2], d2g1_dt2)
        self.assertIs(del2bc[2][0][0], d2Y0_dx2)
        self.assertIs(del2bc[2][0][1], d2Y0_dy2)
        self.assertIs(del2bc[2][0][2], d2Y0_dt2)

    def test_dG_ddY_dx(self):
        """Test the derivative of G wrt dY/dx."""
        dG_ddY_dx_ref = 0  # For all inputs.
        Y = 0.2
        delY = [0.3, 0.4, 0.5]
        del2Y = [0.5, 0.6, 0.7]
        for x in xx:
            for y in yy:
                for t in tt:
                    xyt = (x, y, t)
                    self.assertAlmostEqual(
                        dG_ddY_dx(xyt, Y, delY, del2Y),
                        dG_ddY_dx_ref)

    def test_dG_ddY_dy(self):
        """Test the derivative of G wrt dY/dy."""
        dG_ddY_dy_ref = 0  # For all inputs.
        Y = 0.2
        delY = [0.3, 0.4, 0.5]
        del2Y = [0.5, 0.6, 0.7]
        for x in xx:
            for y in yy:
                for t in tt:
                    xyt = (x, y, t)
                    self.assertAlmostEqual(
                        dG_ddY_dy(xyt, Y, delY, del2Y),
                        dG_ddY_dy_ref)

    def test_dG_ddY_dt(self):
        """Test the derivative of G wrt dY/dt."""
        dG_ddY_dt_ref = 1  # For all inputs.
        Y = 0.2
        delY = [0.3, 0.4, 0.5]
        del2Y = [0.5, 0.6, 0.7]
        for x in xx:
            for y in yy:
                for t in tt:
                    xyt = (x, y, t)
                    self.assertAlmostEqual(
                        dG_ddY_dt(xyt, Y, delY, del2Y),
                        dG_ddY_dt_ref)

    def test_dG_ddelY(self):
        """Test the derivatives of G wrt gradients."""
        dG_dY_dx_ref = 0  # For all inputs.
        dG_dY_dy_ref = 0  # For all inputs.
        dG_dY_dt_ref = 1  # For all inputs.
        dG_ddelY_ref = [dG_dY_dx_ref, dG_dY_dy_ref, dG_dY_dt_ref]
        Y = 0.2
        delY = [0.3, 0.4, 0.5]
        del2Y = [0.5, 0.6, 0.7]
        for x in xx:
            for y in yy:
                for t in tt:
                    xyt = (x, y, t)
                    for i in range(3):
                        self.assertAlmostEqual(
                            dG_ddelY[i](xyt, Y, delY, del2Y),
                            dG_ddelY_ref[i])

    def test_dG_dd2Y_dx2(self):
        """Test the derivative of G wrt d2Y/dx2."""
        dG_dd2Y_dx2_ref = -D  # For all inputs.
        Y = 0.2
        delY = [0.3, 0.4, 0.5]
        del2Y = [0.5, 0.6, 0.7]
        for x in xx:
            for y in yy:
                for t in tt:
                    xyt = (x, y, t)
                    self.assertAlmostEqual(
                        dG_dd2Y_dx2(xyt, Y, delY, del2Y),
                        dG_dd2Y_dx2_ref)

    def test_dG_dd2Y_dy2(self):
        """Test the derivative of G wrt d2Y/dy2."""
        dG_dd2Y_dy2_ref = -D  # For all inputs.
        Y = 0.2
        delY = [0.3, 0.4, 0.5]
        del2Y = [0.5, 0.6, 0.7]
        for x in xx:
            for y in yy:
                for t in tt:
                    xyt = (x, y, t)
                    self.assertAlmostEqual(
                        dG_dd2Y_dy2(xyt, Y, delY, del2Y),
                        dG_dd2Y_dy2_ref)

    def test_dG_dd2Y_dt2(self):
        """Test the derivative of G wrt d2Y/dt2."""
        dG_dd2Y_dt2_ref = 0  # For all inputs.
        Y = 0.2
        delY = [0.3, 0.4, 0.5]
        del2Y = [0.5, 0.6, 0.7]
        for x in xx:
            for y in yy:
                for t in tt:
                    xyt = (x, y, t)
                    self.assertAlmostEqual(
                        dG_dd2Y_dt2(xyt, Y, delY, del2Y),
                        dG_dd2Y_dt2_ref)

    def test_dG_ddel2Y(self):
        """Test the derivatives of G wrt Laplacians."""
        dG_d2Y_dx2_ref = -D  # For all inputs.
        dG_d2Y_dy2_ref = -D  # For all inputs.
        dG_d2Y_dt2_ref = 0  # For all inputs.
        dG_ddel2Y_ref = [dG_d2Y_dx2_ref, dG_d2Y_dy2_ref, dG_d2Y_dt2_ref]
        Y = 0.2
        delY = [0.3, 0.4, 0.5]
        del2Y = [0.5, 0.6, 0.7]
        for x in xx:
            for y in yy:
                for t in tt:
                    xyt = (x, y, t)
                    for i in range(3):
                        self.assertAlmostEqual(
                            dG_ddel2Y[i](xyt, Y, delY, del2Y),
                            dG_ddel2Y_ref[i])

    def test_A(self):
        """Test the optimized boundary condition function."""
        for x in xx:
            for y in yy:
                for t in tt:
                    xyt = (x, y, t)
                    A_ref = (1 - t)*sin(pi*x)*sin(pi*y)
                    self.assertAlmostEqual(A(xyt), A_ref)

    def test_delA(self):
        """Test the optimized boundary condition function gradient."""
        delA_ref = [None, None, None]
        for x in xx:
            for y in yy:
                for t in tt:
                    xyt = (x, y, t)
                    delA_ref[0] = pi*(1 - t)*cos(pi*x)*sin(pi*y)
                    delA_ref[1] = pi*(1 - t)*sin(pi*x)*cos(pi*y)
                    delA_ref[2] = -sin(pi*x)*sin(pi*y)
                    for i in range(2):
                        self.assertAlmostEqual(delA(xyt)[i], delA_ref[i])

    def test_del2A(self):
        """Test the optimized boundary condition function Laplacians."""
        del2A_ref = [None, None, None]
        for x in xx:
            for y in yy:
                for t in tt:
                    xyt = (x, y, t)
                    del2A_ref[0] = -pi**2*(1 - t)*sin(pi*x)*sin(pi*y)
                    del2A_ref[1] = -pi**2*(1 - t)*sin(pi*x)*sin(pi*y)
                    del2A_ref[2] = 0
                    for i in range(2):
                        self.assertAlmostEqual(del2A(xyt)[i], del2A_ref[i])

    def test_Ya(self):
        """Test the analytical solution."""
        for x in xx:
            for y in yy:
                for t in tt:
                    xyt = (x, y, t)
                    Ya_ref = exp(-2*pi**2*D*t)*sin(pi*x)*sin(pi*y)
                    self.assertAlmostEqual(Ya(xyt), Ya_ref)

    def test_dYa_dx(self):
        """Test the first derivative of the analytical solution wrt x."""
        for x in xx:
            for y in yy:
                for t in tt:
                    xyt = (x, y, t)
                    dYa_dx_ref = pi*exp(-2*pi**2*D*t)*cos(pi*x)*sin(pi*y)
                    self.assertAlmostEqual(dYa_dx(xyt), dYa_dx_ref)

    def test_dYa_dy(self):
        """Test the first derivative of the analytical solution wrt y."""
        for x in xx:
            for y in yy:
                for t in tt:
                    xyt = (x, y, t)
                    dYa_dy_ref = pi*exp(-2*pi**2*D*t)*sin(pi*x)*cos(pi*y)
                    self.assertAlmostEqual(dYa_dy(xyt), dYa_dy_ref)

    def test_dYa_dt(self):
        """Test the first derivative of the analytical solution wrt t."""
        for x in xx:
            for y in yy:
                for t in tt:
                    xyt = (x, y, t)
                    dYa_dt_ref = -2*pi**2*D*exp(-2*pi**2*D*t)*sin(pi*x)*sin(pi*y)
                    self.assertAlmostEqual(dYa_dt(xyt), dYa_dt_ref)

    def test_delYa(self):
        """Test the analytical solution gradient."""
        for x in xx:
            for y in yy:
                for t in tt:
                    xyt = (x, y, t)
                    self.assertAlmostEqual(delYa[0](xyt), dYa_dx(xyt))
                    self.assertAlmostEqual(delYa[1](xyt), dYa_dy(xyt))
                    self.assertAlmostEqual(delYa[2](xyt), dYa_dt(xyt))

    def test_d2Ya_dx2(self):
        """Test the second derivative of the analytical solution wrt x."""
        for x in xx:
            for y in yy:
                for t in tt:
                    xyt = (x, y, t)
                    d2Ya_dx2_ref = -pi**2*exp(-2*pi**2*D*t)*sin(pi*x)*sin(pi*y)
                    self.assertAlmostEqual(d2Ya_dx2(xyt), d2Ya_dx2_ref)

    def test_d2Ya_dy2(self):
        """Test the second derivative of the analytical solution wrt y."""
        for x in xx:
            for y in yy:
                for t in tt:
                    xyt = (x, y, t)
                    d2Ya_dy2_ref = -pi**2*exp(-2*pi**2*D*t)*sin(pi*x)*sin(pi*y)
                    self.assertAlmostEqual(d2Ya_dy2(xyt), d2Ya_dy2_ref)

    def test_d2Ya_dt2(self):
        """Test the second derivative of the analytical solution wrt t."""
        for x in xx:
            for y in yy:
                for t in tt:
                    xyt = (x, y, t)
                    d2Ya_dt2_ref = 4*pi**4*D**2*exp(-2*pi**2*D*t)*sin(pi*x)*sin(pi*y)
                    self.assertAlmostEqual(d2Ya_dt2(xyt), d2Ya_dt2_ref)

    def test_del2A(self):
        """Test the optimized boundary condition function Laplacians."""
        del2A_ref = [None, None, None]
        for x in xx:
            for y in yy:
                for t in tt:
                    xyt = (x, y, t)
                    self.assertAlmostEqual(del2Ya[0](xyt), d2Ya_dx2(xyt))
                    self.assertAlmostEqual(del2Ya[1](xyt), d2Ya_dy2(xyt))
                    self.assertAlmostEqual(del2Ya[2](xyt), d2Ya_dt2(xyt))


if __name__ == '__main__':
    unittest.main()
