"""Tests for the diff3d_halfsine_increase module."""


from math import cos, pi, sin
import unittest

import numpy as np

from nnde.differentialequation.examples.diff3d_halfsine_increase import (
    a, D,
    G,
    f0, f1, g0, g1, h0, h1, Y0, bc,
    df0_dx, df0_dy, df0_dz, df0_dt, df1_dx, df1_dy, df1_dz, df1_dt,
    dg0_dx, dg0_dy, dg0_dz, dg0_dt, dg1_dx, dg1_dy, dg1_dz, dg1_dt,
    dh0_dx, dh0_dy, dh0_dz, dh0_dt, dh1_dx, dh1_dy, dh1_dz, dh1_dt,
    dY0_dx, dY0_dy, dY0_dz, dY0_dt, delbc,
    d2f0_dx2, d2f0_dy2, d2f0_dz2, d2f0_dt2, d2f1_dx2, d2f1_dy2, d2f1_dz2,
    d2f1_dt2,
    d2g0_dx2, d2g0_dy2, d2g0_dz2, d2g0_dt2, d2g1_dx2, d2g1_dy2, d2g1_dz2,
    d2g1_dt2,
    d2h0_dx2, d2h0_dy2, d2h0_dz2, d2h0_dt2, d2h1_dx2, d2h1_dy2, d2h1_dz2,
    d2h1_dt2,
    d2Y0_dx2, d2Y0_dy2, d2Y0_dz2, d2Y0_dt2, del2bc,
    dG_ddY_dx, dG_ddY_dy, dG_ddY_dz, dG_ddY_dt, dG_ddelY,
    dG_dd2Y_dx2, dG_dd2Y_dy2, dG_dd2Y_dz2, dG_dd2Y_dt2, dG_ddel2Y,
    A, delA, del2A,
)


# Grid points for testing.
n = 11
xx = np.linspace(0, 1, n)
yy = np.linspace(0, 1, n)
zz = np.linspace(0, 1, n)
tt = np.linspace(0, 1, n)


class TestBuilder(unittest.TestCase):
    """Tests for the diff3d_halfsine_increase module."""

    def test_G(self):
        """Test the differential equation."""
        Yt = 2
        delYt = (3, 4, 5, 6)
        del2Yt = (5, 6, 7, 8)
        for x in xx:
            for y in yy:
                for z in zz:
                    for t in tt:
                        xyzt = (x, y, z, t)
                        G_ref = (
                            delYt[3] - D*(del2Yt[0] + del2Yt[1] + del2Yt[2]))
                        self.assertAlmostEqual(G(xyzt, Yt, delYt, del2Yt),
                                               G_ref)

    def test_f0(self):
        """Test boundary condition at (x, y, z, t) = (0, y, z, t)."""
        f0_ref = 0  # For all inputs.
        for x in xx:
            for y in yy:
                for z in zz:
                    for t in tt:
                        xyzt = (x, y, z, t)
                        self.assertAlmostEqual(f0(xyzt), f0_ref)

    def test_f1(self):
        """Test boundary condition at (x, y, z, t) = (1, y, z, t)."""
        for x in xx:
            for y in yy:
                for z in zz:
                    for t in tt:
                        xyzt = (x, y, z, t)
                        f1_ref = a*t*sin(pi*y)*sin(pi*z)
                        self.assertAlmostEqual(f1(xyzt), f1_ref)

    def test_g0(self):
        """Test boundary condition at (x, y, z, t) = (x, 0, z, t)."""
        g0_ref = 0  # For all inputs.
        for x in xx:
            for y in yy:
                for z in zz:
                    for t in tt:
                        xyzt = (x, y, z, t)
                        self.assertAlmostEqual(g0(xyzt), g0_ref)

    def test_g1(self):
        """Test boundary condition at (x, y, z, t) = (x, 1, z, t)."""
        g1_ref = 0  # For all inputs.
        for x in xx:
            for y in yy:
                for z in zz:
                    for t in tt:
                        xyzt = (x, y, z, t)
                        self.assertAlmostEqual(g1(xyzt), g1_ref)

    def test_h0(self):
        """Test boundary condition at (x, y, z, t) = (x, y, 0, t)."""
        h0_ref = 0  # For all inputs.
        for x in xx:
            for y in yy:
                for z in zz:
                    for t in tt:
                        xyzt = (x, y, z, t)
                        self.assertAlmostEqual(h0(xyzt), h0_ref)

    def test_h1(self):
        """Test boundary condition at (x, y, z, t) = (x, y, 1, t)."""
        h1_ref = 0  # For all inputs.
        for x in xx:
            for y in yy:
                for z in zz:
                    for t in tt:
                        xyzt = (x, y, z, t)
                        self.assertAlmostEqual(h1(xyzt), h1_ref)

    def test_Y0(self):
        """Test boundary condition at (x, y, z, t) = (x, y, z, 0)."""
        for x in xx:
            for y in yy:
                for z in zz:
                    for t in tt:
                        xyzt = (x, y, z, t)
                        Y0_ref = sin(pi*x)*sin(pi*y)*sin(pi*z)
                        self.assertAlmostEqual(Y0(xyzt), Y0_ref)

    def test_bc(self):
        """Test that the boundary conditions are continuous at corners."""
        ((f0, f1), (g0, g1), (h0, h1), (Y0, Y1_ignore)) = bc
        xyzt = (0, 0, 0, 0)
        self.assertAlmostEqual(f0(xyzt), g0(xyzt))
        self.assertAlmostEqual(f0(xyzt), h0(xyzt))
        self.assertAlmostEqual(f0(xyzt), Y0(xyzt))
        xyzt = (0, 0, 1, 0)
        self.assertAlmostEqual(f0(xyzt), g0(xyzt))
        self.assertAlmostEqual(f0(xyzt), h1(xyzt))
        self.assertAlmostEqual(f0(xyzt), Y0(xyzt))
        xyzt = (0, 1, 0, 0)
        self.assertAlmostEqual(f0(xyzt), g1(xyzt))
        self.assertAlmostEqual(f0(xyzt), h0(xyzt))
        self.assertAlmostEqual(f0(xyzt), Y0(xyzt))
        xyzt = (0, 1, 1, 0)
        self.assertAlmostEqual(f0(xyzt), g1(xyzt))
        self.assertAlmostEqual(f0(xyzt), h1(xyzt))
        self.assertAlmostEqual(f0(xyzt), Y0(xyzt))
        xyzt = (1, 0, 0, 0)
        self.assertAlmostEqual(f1(xyzt), g0(xyzt))
        self.assertAlmostEqual(f1(xyzt), h0(xyzt))
        self.assertAlmostEqual(f1(xyzt), Y0(xyzt))
        xyzt = (1, 0, 1, 0)
        self.assertAlmostEqual(f1(xyzt), g0(xyzt))
        self.assertAlmostEqual(f1(xyzt), h1(xyzt))
        self.assertAlmostEqual(f1(xyzt), Y0(xyzt))
        xyzt = (1, 1, 0, 0)
        self.assertAlmostEqual(f1(xyzt), g1(xyzt))
        self.assertAlmostEqual(f1(xyzt), h0(xyzt))
        self.assertAlmostEqual(f1(xyzt), Y0(xyzt))
        xyzt = (1, 1, 1, 0)
        self.assertAlmostEqual(f1(xyzt), g1(xyzt))
        self.assertAlmostEqual(f1(xyzt), h1(xyzt))
        self.assertAlmostEqual(f1(xyzt), Y0(xyzt))

    def test_df0_dx(self):
        """Test the first derivative of f0 wrt x."""
        df0_dx_ref = 0  # For all inputs.
        for x in xx:
            for y in yy:
                for z in zz:
                    for t in tt:
                        xyzt = (x, y, z, t)
                        self.assertAlmostEqual(df0_dx(xyzt), df0_dx_ref)

    def test_df0_dy(self):
        """Test the first derivative of f0 wrt y."""
        df0_dy_ref = 0  # For all inputs.
        for x in xx:
            for y in yy:
                for z in zz:
                    for t in tt:
                        xyzt = (x, y, z, t)
                        self.assertAlmostEqual(df0_dy(xyzt), df0_dy_ref)

    def test_df0_dz(self):
        """Test the first derivative of f0 wrt z."""
        df0_dz_ref = 0  # For all inputs.
        for x in xx:
            for y in yy:
                for z in zz:
                    for t in tt:
                        xyzt = (x, y, z, t)
                        self.assertAlmostEqual(df0_dz(xyzt), df0_dz_ref)

    def test_df0_dt(self):
        """Test the first derivative of f0 wrt t."""
        df0_dt_ref = 0  # For all inputs.
        for x in xx:
            for y in yy:
                for z in zz:
                    for t in tt:
                        xyzt = (x, y, z, t)
                        self.assertAlmostEqual(df0_dt(xyzt), df0_dt_ref)

    def test_df1_dx(self):
        """Test the first derivative of f1 wrt x."""
        df1_dx_ref = 0  # For all inputs.
        for x in xx:
            for y in yy:
                for z in zz:
                    for t in tt:
                        xyzt = (x, y, z, t)
                        self.assertAlmostEqual(df1_dx(xyzt), df1_dx_ref)

    def test_df1_dy(self):
        """Test the first derivative of f1 wrt y."""
        for x in xx:
            for y in yy:
                for z in zz:
                    for t in tt:
                        xyzt = (x, y, z, t)
                        df1_dy_ref = a*pi*t*cos(pi*y)*sin(pi*z)
                        self.assertAlmostEqual(df1_dy(xyzt), df1_dy_ref)

    def test_df1_dz(self):
        """Test the first derivative of f1 wrt z."""
        for x in xx:
            for y in yy:
                for z in zz:
                    for t in tt:
                        xyzt = (x, y, z, t)
                        df1_dz_ref = a*pi*t*sin(pi*y)*cos(pi*z)
                        self.assertAlmostEqual(df1_dz(xyzt), df1_dz_ref)

    def test_df1_dt(self):
        """Test the first derivative of f1 wrt t."""
        df1_dt_ref = 0  # For all inputs.
        for x in xx:
            for y in yy:
                for z in zz:
                    for t in tt:
                        xyzt = (x, y, z, t)
                        df1_dt_ref = a*sin(pi*y)*sin(pi*z)
                        self.assertAlmostEqual(df1_dt(xyzt), df1_dt_ref)

    def test_dg0_dx(self):
        """Test the first derivative of g0 wrt x."""
        dg0_dx_ref = 0  # For all inputs.
        for x in xx:
            for y in yy:
                for z in zz:
                    for t in tt:
                        xyzt = (x, y, z, t)
                        self.assertAlmostEqual(dg0_dx(xyzt), dg0_dx_ref)

    def test_dg0_dy(self):
        """Test the first derivative of g0 wrt y."""
        dg0_dy_ref = 0  # For all inputs.
        for x in xx:
            for y in yy:
                for z in zz:
                    for t in tt:
                        xyzt = (x, y, z, t)
                        self.assertAlmostEqual(dg0_dy(xyzt), dg0_dy_ref)

    def test_dg0_dz(self):
        """Test the first derivative of g0 wrt z."""
        dg0_dz_ref = 0  # For all inputs.
        for x in xx:
            for y in yy:
                for z in zz:
                    for t in tt:
                        xyzt = (x, y, z, t)
                        self.assertAlmostEqual(dg0_dy(xyzt), dg0_dz_ref)

    def test_dg0_dt(self):
        """Test the first derivative of g0 wrt t."""
        dg0_dt_ref = 0  # For all inputs.
        for x in xx:
            for y in yy:
                for z in zz:
                    for t in tt:
                        xyzt = (x, y, z, t)
                        self.assertAlmostEqual(dg0_dt(xyzt), dg0_dt_ref)

    def test_dg1_dx(self):
        """Test the first derivative of g1 wrt x."""
        dg1_dx_ref = 0  # For all inputs.
        for x in xx:
            for y in yy:
                for z in zz:
                    for t in tt:
                        xyzt = (x, y, z, t)
                        self.assertAlmostEqual(dg1_dx(xyzt), dg1_dx_ref)

    def test_dg1_dy(self):
        """Test the first derivative of g1 wrt y."""
        dg1_dy_ref = 0  # For all inputs.
        for x in xx:
            for y in yy:
                for z in zz:
                    for t in tt:
                        xyzt = (x, y, z, t)
                        self.assertAlmostEqual(dg1_dy(xyzt), dg1_dy_ref)

    def test_dg1_dz(self):
        """Test the first derivative of g1 wrt z."""
        dg1_dz_ref = 0  # For all inputs.
        for x in xx:
            for y in yy:
                for z in zz:
                    for t in tt:
                        xyzt = (x, y, z, t)
                        self.assertAlmostEqual(dg1_dy(xyzt), dg1_dz_ref)

    def test_dg1_dt(self):
        """Test the first derivative of g1 wrt t."""
        dg1_dt_ref = 0  # For all inputs.
        for x in xx:
            for y in yy:
                for z in zz:
                    for t in tt:
                        xyzt = (x, y, z, t)
                        self.assertAlmostEqual(dg1_dt(xyzt), dg1_dt_ref)

    def test_dY0_dx(self):
        """Test the first derivative of Y0 wrt x."""
        for x in xx:
            for y in yy:
                for z in zz:
                    for t in tt:
                        xyzt = (x, y, z, t)
                        dY0_dx_ref = pi*cos(pi*x)*sin(pi*y)*sin(pi*z)
                        self.assertAlmostEqual(dY0_dx(xyzt), dY0_dx_ref)

    def test_dY0_dy(self):
        """Test the first derivative of Y0 wrt y."""
        for x in xx:
            for y in yy:
                for z in zz:
                    for t in tt:
                        xyzt = (x, y, z, t)
                        dY0_dy_ref = pi*sin(pi*x)*cos(pi*y)*sin(pi*z)
                        self.assertAlmostEqual(dY0_dy(xyzt), dY0_dy_ref)

    def test_dY0_dz(self):
        """Test the first derivative of Y0 wrt z."""
        for x in xx:
            for y in yy:
                for z in zz:
                    for t in tt:
                        xyzt = (x, y, z, t)
                        dY0_dz_ref = pi*sin(pi*x)*sin(pi*y)*cos(pi*z)
                        self.assertAlmostEqual(dY0_dz(xyzt), dY0_dz_ref)

    def test_dY0_dt(self):
        """Test the first derivative of Y0 wrt t."""
        dY0_dt_ref = 0  # For all inputs.
        for x in xx:
            for y in yy:
                for z in zz:
                    for t in tt:
                        xyzt = (x, y, z, t)
                        self.assertAlmostEqual(dY0_dt(xyzt), dY0_dt_ref)

    def test_delbc(self):
        """Test the boundary condition gradients as an array."""
        self.assertIs(delbc[0][0][0], df0_dx)
        self.assertIs(delbc[0][0][1], df0_dy)
        self.assertIs(delbc[0][0][2], df0_dz)
        self.assertIs(delbc[0][0][3], df0_dt)
        self.assertIs(delbc[0][1][0], df1_dx)
        self.assertIs(delbc[0][1][1], df1_dy)
        self.assertIs(delbc[0][1][2], df1_dz)
        self.assertIs(delbc[0][1][3], df1_dt)
        self.assertIs(delbc[1][0][0], dg0_dx)
        self.assertIs(delbc[1][0][1], dg0_dy)
        self.assertIs(delbc[1][0][2], dg0_dz)
        self.assertIs(delbc[1][0][3], dg0_dt)
        self.assertIs(delbc[1][1][0], dg1_dx)
        self.assertIs(delbc[1][1][1], dg1_dy)
        self.assertIs(delbc[1][1][2], dg1_dz)
        self.assertIs(delbc[1][1][3], dg1_dt)
        self.assertIs(delbc[2][0][0], dh0_dx)
        self.assertIs(delbc[2][0][1], dh0_dy)
        self.assertIs(delbc[2][0][2], dh0_dz)
        self.assertIs(delbc[2][0][3], dh0_dt)
        self.assertIs(delbc[2][1][0], dh1_dx)
        self.assertIs(delbc[2][1][1], dh1_dy)
        self.assertIs(delbc[2][1][2], dh1_dz)
        self.assertIs(delbc[2][1][3], dh1_dt)
        self.assertIs(delbc[3][0][0], dY0_dx)
        self.assertIs(delbc[3][0][1], dY0_dy)
        self.assertIs(delbc[3][0][2], dY0_dz)
        self.assertIs(delbc[3][0][3], dY0_dt)
        self.assertIsNone(delbc[3][1][0])
        self.assertIsNone(delbc[3][1][1])
        self.assertIsNone(delbc[3][1][2])
        self.assertIsNone(delbc[3][1][3])

    def test_d2f0_dx2(self):
        """Test the 2nd derivative of f0 wrt x."""
        d2f0_dx2_ref = 0  # For all inputs.
        for x in xx:
            for y in yy:
                for z in zz:
                    for t in tt:
                        xyzt = (x, y, z, t)
                        self.assertAlmostEqual(d2f0_dx2(xyzt), d2f0_dx2_ref)

    def test_d2f0_dy2(self):
        """Test the 2nd derivative of f0 wrt y."""
        d2f0_dy2_ref = 0  # For all inputs.
        for x in xx:
            for y in yy:
                for z in zz:
                    for t in tt:
                        xyzt = (x, y, z, t)
                        self.assertAlmostEqual(d2f0_dy2(xyzt), d2f0_dy2_ref)

    def test_d2f0_dz2(self):
        """Test the 2nd derivative of f0 wrt z."""
        d2f0_dz2_ref = 0  # For all inputs.
        for x in xx:
            for y in yy:
                for z in zz:
                    for t in tt:
                        xyzt = (x, y, z, t)
                        self.assertAlmostEqual(d2f0_dz2(xyzt), d2f0_dz2_ref)

    def test_d2f0_dt2(self):
        """Test the 2nd derivative of f0 wrt t."""
        d2f0_dt2_ref = 0  # For all inputs.
        for x in xx:
            for y in yy:
                for z in zz:
                    for t in tt:
                        xyzt = (x, y, z, t)
                        self.assertAlmostEqual(d2f0_dt2(xyzt), d2f0_dt2_ref)

    def test_d2f1_dx2(self):
        """Test the 2nd derivative of f1 wrt x."""
        d2f1_dx2_ref = 0  # For all inputs.
        for x in xx:
            for y in yy:
                for z in zz:
                    for t in tt:
                        xyzt = (x, y, z, t)
                        self.assertAlmostEqual(d2f1_dx2(xyzt), d2f1_dx2_ref)

    def test_d2f1_dy2(self):
        """Test the 2nd derivative of f1 wrt y."""
        for x in xx:
            for y in yy:
                for z in zz:
                    for t in tt:
                        xyzt = (x, y, z, t)
                        d2f1_dy2_ref = -a*pi**2*t*sin(pi*y)*sin(pi*z)
                        self.assertAlmostEqual(d2f1_dy2(xyzt), d2f1_dy2_ref)

    def test_d2f1_dz2(self):
        """Test the 2nd derivative of f1 wrt z."""
        for x in xx:
            for y in yy:
                for z in zz:
                    for t in tt:
                        xyzt = (x, y, z, t)
                        d2f1_dz2_ref = -a*pi**2*t*sin(pi*y)*sin(pi*z)
                        self.assertAlmostEqual(d2f1_dz2(xyzt), d2f1_dz2_ref)

    def test_d2f1_dt2(self):
        """Test the 2nd derivative of f1 wrt t."""
        d2f1_dt2_ref = 0  # For all inputs.
        for x in xx:
            for y in yy:
                for z in zz:
                    for t in tt:
                        xyzt = (x, y, z, t)
                        self.assertAlmostEqual(d2f1_dt2(xyzt), d2f1_dt2_ref)

    def test_d2g0_dx2(self):
        """Test the 2nd derivative of g0 wrt x."""
        d2g0_dx2_ref = 0  # For all inputs.
        for x in xx:
            for y in yy:
                for z in zz:
                    for t in tt:
                        xyzt = (x, y, z, t)
                        self.assertAlmostEqual(d2g0_dx2(xyzt), d2g0_dx2_ref)

    def test_d2g0_dy2(self):
        """Test the 2nd derivative of g0 wrt y."""
        d2g0_dy2_ref = 0  # For all inputs.
        for x in xx:
            for y in yy:
                for z in zz:
                    for t in tt:
                        xyzt = (x, y, z, t)
                        self.assertAlmostEqual(d2g0_dy2(xyzt), d2g0_dy2_ref)

    def test_d2g0_dz2(self):
        """Test the 2nd derivative of g0 wrt z."""
        d2g0_dz2_ref = 0  # For all inputs.
        for x in xx:
            for y in yy:
                for z in zz:
                    for t in tt:
                        xyzt = (x, y, z, t)
                        self.assertAlmostEqual(d2g0_dz2(xyzt), d2g0_dz2_ref)

    def test_d2g0_dt2(self):
        """Test the 2nd derivative of g0 wrt t."""
        d2g0_dt2_ref = 0  # For all inputs.
        for x in xx:
            for y in yy:
                for z in zz:
                    for t in tt:
                        xyzt = (x, y, z, t)
                        self.assertAlmostEqual(d2g0_dt2(xyzt), d2g0_dt2_ref)

    def test_d2g1_dx2(self):
        """Test the 2nd derivative of g1 wrt x."""
        d2g1_dx2_ref = 0  # For all inputs.
        for x in xx:
            for y in yy:
                for z in zz:
                    for t in tt:
                        xyzt = (x, y, z, t)
                        self.assertAlmostEqual(d2g1_dx2(xyzt), d2g1_dx2_ref)

    def test_d2g1_dy2(self):
        """Test the 2nd derivative of g1 wrt y."""
        d2g1_dy2_ref = 0  # For all inputs.
        for x in xx:
            for y in yy:
                for z in zz:
                    for t in tt:
                        xyzt = (x, y, z, t)
                        self.assertAlmostEqual(d2g1_dy2(xyzt), d2g1_dy2_ref)

    def test_d2g1_dz2(self):
        """Test the 2nd derivative of g1 wrt z."""
        d2g1_dz2_ref = 0  # For all inputs.
        for x in xx:
            for y in yy:
                for z in zz:
                    for t in tt:
                        xyzt = (x, y, z, t)
                        self.assertAlmostEqual(d2g1_dz2(xyzt), d2g1_dz2_ref)

    def test_d2g1_dt2(self):
        """Test the 2nd derivative of g1 wrt t."""
        d2g1_dt2_ref = 0  # For all inputs.
        for x in xx:
            for y in yy:
                for z in zz:
                    for t in tt:
                        xyzt = (x, y, z, t)
                        self.assertAlmostEqual(d2g1_dt2(xyzt), d2g1_dt2_ref)

    def test_d2h0_dx2(self):
        """Test the 2nd derivative of h0 wrt x."""
        d2h0_dx2_ref = 0  # For all inputs.
        for x in xx:
            for y in yy:
                for z in zz:
                    for t in tt:
                        xyzt = (x, y, z, t)
                        self.assertAlmostEqual(d2h0_dx2(xyzt), d2h0_dx2_ref)

    def test_d2h0_dy2(self):
        """Test the 2nd derivative of h0 wrt y."""
        d2h0_dy2_ref = 0  # For all inputs.
        for x in xx:
            for y in yy:
                for z in zz:
                    for t in tt:
                        xyzt = (x, y, z, t)
                        self.assertAlmostEqual(d2h0_dy2(xyzt), d2h0_dy2_ref)

    def test_d2h0_dz2(self):
        """Test the 2nd derivative of h0 wrt z."""
        d2h0_dz2_ref = 0  # For all inputs.
        for x in xx:
            for y in yy:
                for z in zz:
                    for t in tt:
                        xyzt = (x, y, z, t)
                        self.assertAlmostEqual(d2h0_dz2(xyzt), d2h0_dz2_ref)

    def test_d2h0_dt2(self):
        """Test the 2nd derivative of h0 wrt t."""
        d2h0_dt2_ref = 0  # For all inputs.
        for x in xx:
            for y in yy:
                for z in zz:
                    for t in tt:
                        xyzt = (x, y, z, t)
                        self.assertAlmostEqual(d2h0_dt2(xyzt), d2h0_dt2_ref)

    def test_d2h1_dx2(self):
        """Test the 2nd derivative of h1 wrt x."""
        d2h1_dx2_ref = 0  # For all inputs.
        for x in xx:
            for y in yy:
                for z in zz:
                    for t in tt:
                        xyzt = (x, y, z, t)
                        self.assertAlmostEqual(d2h1_dx2(xyzt), d2h1_dx2_ref)

    def test_d2h1_dy2(self):
        """Test the 2nd derivative of h1 wrt y."""
        d2h1_dy2_ref = 0  # For all inputs.
        for x in xx:
            for y in yy:
                for z in zz:
                    for t in tt:
                        xyzt = (x, y, z, t)
                        self.assertAlmostEqual(d2h1_dy2(xyzt), d2h1_dy2_ref)

    def test_d2h1_dz2(self):
        """Test the 2nd derivative of h1 wrt z."""
        d2h1_dz2_ref = 0  # For all inputs.
        for x in xx:
            for y in yy:
                for z in zz:
                    for t in tt:
                        xyzt = (x, y, z, t)
                        self.assertAlmostEqual(d2h1_dz2(xyzt), d2h1_dz2_ref)

    def test_d2h1_dt2(self):
        """Test the 2nd derivative of h1 wrt t."""
        d2h1_dt2_ref = 0  # For all inputs.
        for x in xx:
            for y in yy:
                for z in zz:
                    for t in tt:
                        xyzt = (x, y, z, t)
                        self.assertAlmostEqual(d2h1_dt2(xyzt), d2h1_dt2_ref)

    def test_d2Y0_dx2(self):
        """Test the 2nd derivative of Y0 wrt x."""
        for x in xx:
            for y in yy:
                for z in zz:
                    for t in tt:
                        xyzt = (x, y, z, t)
                        d2Y0_dx2_ref = -pi**2*sin(pi*x)*sin(pi*y)*sin(pi*z)
                        self.assertAlmostEqual(d2Y0_dx2(xyzt), d2Y0_dx2_ref)

    def test_d2Y0_dy2(self):
        """Test the 2nd derivative of Y0 wrt y."""
        for x in xx:
            for y in yy:
                for z in zz:
                    for t in tt:
                        xyzt = (x, y, z, t)
                        d2Y0_dy2_ref = -pi**2*sin(pi*x)*sin(pi*y)*sin(pi*z)
                        self.assertAlmostEqual(d2Y0_dy2(xyzt), d2Y0_dy2_ref)

    def test_d2Y0_dz2(self):
        """Test the 2nd derivative of Y0 wrt z."""
        for x in xx:
            for y in yy:
                for z in zz:
                    for t in tt:
                        xyzt = (x, y, z, t)
                        d2Y0_dz2_ref = -pi**2*sin(pi*x)*sin(pi*y)*sin(pi*z)
                        self.assertAlmostEqual(d2Y0_dz2(xyzt), d2Y0_dz2_ref)

    def test_d2Y0_dt2(self):
        """Test the 2nd derivative of Y0 wrt t."""
        d2Y0_dt2_ref = 0  # For all inputs.
        for x in xx:
            for y in yy:
                for z in zz:
                    for t in tt:
                        xyzt = (x, y, z, t)
                        self.assertAlmostEqual(d2Y0_dt2(xyzt), d2Y0_dt2_ref)

    def test_del2bc(self):
        """Test the boundary condition Laplacian as an array."""
        self.assertIs(del2bc[0][0][0], d2f0_dx2)
        self.assertIs(del2bc[0][0][1], d2f0_dy2)
        self.assertIs(del2bc[0][0][2], d2f0_dz2)
        self.assertIs(del2bc[0][0][3], d2f0_dt2)
        self.assertIs(del2bc[0][1][0], d2f1_dx2)
        self.assertIs(del2bc[0][1][1], d2f1_dy2)
        self.assertIs(del2bc[0][1][2], d2f1_dz2)
        self.assertIs(del2bc[0][1][3], d2f1_dt2)
        self.assertIs(del2bc[1][0][0], d2g0_dx2)
        self.assertIs(del2bc[1][0][1], d2g0_dy2)
        self.assertIs(del2bc[1][0][2], d2g0_dz2)
        self.assertIs(del2bc[1][0][3], d2g0_dt2)
        self.assertIs(del2bc[1][1][0], d2g1_dx2)
        self.assertIs(del2bc[1][1][1], d2g1_dy2)
        self.assertIs(del2bc[1][1][2], d2g1_dz2)
        self.assertIs(del2bc[1][1][3], d2g1_dt2)
        self.assertIs(del2bc[2][0][0], d2h0_dx2)
        self.assertIs(del2bc[2][0][1], d2h0_dy2)
        self.assertIs(del2bc[2][0][2], d2h0_dz2)
        self.assertIs(del2bc[2][0][3], d2h0_dt2)
        self.assertIs(del2bc[2][1][0], d2h1_dx2)
        self.assertIs(del2bc[2][1][1], d2h1_dy2)
        self.assertIs(del2bc[2][1][2], d2h1_dz2)
        self.assertIs(del2bc[2][1][3], d2h1_dt2)
        self.assertIs(del2bc[3][0][0], d2Y0_dx2)
        self.assertIs(del2bc[3][0][1], d2Y0_dy2)
        self.assertIs(del2bc[3][0][2], d2Y0_dz2)
        self.assertIs(del2bc[3][0][3], d2Y0_dt2)
        self.assertIsNone(del2bc[3][1][0])
        self.assertIsNone(del2bc[3][1][1])
        self.assertIsNone(del2bc[3][1][2])
        self.assertIsNone(del2bc[3][1][3])

    def test_dG_ddY_dx(self):
        """Test the derivative of G wrt dY/dx."""
        dG_ddY_dx_ref = 0  # For all inputs.
        Y = 0.2
        delY = [0.3, 0.4, 0.5, 0.6]
        del2Y = [0.5, 0.6, 0.7, 0.8]
        for x in xx:
            for y in yy:
                for z in zz:
                    for t in tt:
                        xyzt = (x, y, z, t)
                        self.assertAlmostEqual(
                            dG_ddY_dx(xyzt, Y, delY, del2Y),
                            dG_ddY_dx_ref)

    def test_dG_ddY_dy(self):
        """Test the derivative of G wrt dY/dy."""
        dG_ddY_dy_ref = 0  # For all inputs.
        Y = 0.2
        delY = [0.3, 0.4, 0.5, 0.6]
        del2Y = [0.5, 0.6, 0.7, 0.8]
        for x in xx:
            for y in yy:
                for z in zz:
                    for t in tt:
                        xyzt = (x, y, z, t)
                        self.assertAlmostEqual(
                            dG_ddY_dy(xyzt, Y, delY, del2Y),
                            dG_ddY_dy_ref)

    def test_dG_ddY_dz(self):
        """Test the derivative of G wrt dY/dz."""
        dG_ddY_dz_ref = 0  # For all inputs.
        Y = 0.2
        delY = [0.3, 0.4, 0.5, 0.6]
        del2Y = [0.5, 0.6, 0.7, 0.8]
        for x in xx:
            for y in yy:
                for z in zz:
                    for t in tt:
                        xyzt = (x, y, z, t)
                        self.assertAlmostEqual(
                            dG_ddY_dz(xyzt, Y, delY, del2Y),
                            dG_ddY_dz_ref)

    def test_dG_ddY_dt(self):
        """Test the derivative of G wrt dY/dt."""
        dG_ddY_dt_ref = 1  # For all inputs.
        Y = 0.2
        delY = [0.3, 0.4, 0.5, 0.6]
        del2Y = [0.5, 0.6, 0.7, 0.8]
        for x in xx:
            for y in yy:
                for z in zz:
                    for t in tt:
                        xyzt = (x, y, z, t)
                        self.assertAlmostEqual(
                            dG_ddY_dt(xyzt, Y, delY, del2Y),
                            dG_ddY_dt_ref)

    def test_dG_ddelY(self):
        """Test the derivatives of G wrt gradients."""
        dG_ddelY_ref = [0, 0, 0, 1]  # For all inputs.
        Y = 0.2
        delY = [0.3, 0.4, 0.5, 0.6]
        del2Y = [0.5, 0.6, 0.7, 0.8]
        for x in xx:
            for y in yy:
                for z in zz:
                    for t in tt:
                        xyzt = (x, y, z, t)
                        for i in range(4):
                            self.assertAlmostEqual(
                                dG_ddelY[i](xyzt, Y, delY, del2Y),
                                dG_ddelY_ref[i])

    def test_dG_dd2Y_dx2(self):
        """Test the derivative of G wrt d2Y/dx2."""
        dG_dd2Y_dx2_ref = -D  # For all inputs.
        Y = 0.2
        delY = [0.3, 0.4, 0.5, 0.6]
        del2Y = [0.5, 0.6, 0.7, 0.8]
        for x in xx:
            for y in yy:
                for z in zz:
                    for t in tt:
                        xyzt = (x, y, z, t)
                        self.assertAlmostEqual(
                            dG_dd2Y_dx2(xyzt, Y, delY, del2Y),
                            dG_dd2Y_dx2_ref)

    def test_dG_dd2Y_dy2(self):
        """Test the derivative of G wrt d2Y/dy2."""
        dG_dd2Y_dy2_ref = -D  # For all inputs.
        Y = 0.2
        delY = [0.3, 0.4, 0.5, 0.6]
        del2Y = [0.5, 0.6, 0.7, 0.8]
        for x in xx:
            for y in yy:
                for z in zz:
                    for t in tt:
                        xyzt = (x, y, z, t)
                        self.assertAlmostEqual(
                            dG_dd2Y_dy2(xyzt, Y, delY, del2Y),
                            dG_dd2Y_dy2_ref)

    def test_dG_dd2Y_dz2(self):
        """Test the derivative of G wrt d2Y/dz2."""
        dG_dd2Y_dz2_ref = -D  # For all inputs.
        Y = 0.2
        delY = [0.3, 0.4, 0.5, 0.6]
        del2Y = [0.5, 0.6, 0.7, 0.8]
        for x in xx:
            for y in yy:
                for z in zz:
                    for t in tt:
                        xyzt = (x, y, z, t)
                        self.assertAlmostEqual(
                            dG_dd2Y_dz2(xyzt, Y, delY, del2Y),
                            dG_dd2Y_dz2_ref)

    def test_dG_dd2Y_dt2(self):
        """Test the derivative of G wrt d2Y/dt2."""
        dG_dd2Y_dt2_ref = 0  # For all inputs.
        Y = 0.2
        delY = [0.3, 0.4, 0.5, 0.6]
        del2Y = [0.5, 0.6, 0.7, 0.8]
        for x in xx:
            for y in yy:
                for z in zz:
                    for t in tt:
                        xyzt = (x, y, z, t)
                        self.assertAlmostEqual(
                            dG_dd2Y_dt2(xyzt, Y, delY, del2Y),
                            dG_dd2Y_dt2_ref)

    def test_dG_ddel2Y(self):
        """Test the derivatives of G wrt Laplacians."""
        dG_ddel2Y_ref = [-D, -D, -D, 0]
        Y = 0.2
        delY = [0.3, 0.4, 0.5, 0.6]
        del2Y = [0.5, 0.6, 0.7, 0.8]
        for x in xx:
            for y in yy:
                for z in zz:
                    for t in tt:
                        xyzt = (x, y, z, t)
                        for i in range(4):
                            self.assertAlmostEqual(
                                dG_ddel2Y[i](xyzt, Y, delY, del2Y),
                                dG_ddel2Y_ref[i])

    def test_A(self):
        """Test the optimized boundary condition function."""
        for x in xx:
            for y in yy:
                for z in zz:
                    for t in tt:
                        xyzt = (x, y, z, t)
                        A_ref = (a*t*x + (1 - t)*sin(pi*x))*sin(pi*y)*sin(pi*z)
                        self.assertAlmostEqual(A(xyzt), A_ref)

    def test_delA(self):
        """Test the optimized boundary condition function gradient."""
        delA_ref = [None, None, None, None]
        for x in xx:
            for y in yy:
                for z in zz:
                    for t in tt:
                        xyzt = (x, y, z, t)
                        delA_ref[0] = (
                            (a*t + pi*(1 - t)*cos(pi*x))*sin(pi*y)*sin(pi*z))
                        delA_ref[1] = (
                            pi*cos(pi*y)*(a*t*x + (1 - t)*sin(pi*x))*sin(pi*z))
                        delA_ref[2] = (
                            pi*cos(pi*z)*(a*t*x + (1 - t)*sin(pi*x))*sin(pi*y))
                        delA_ref[3] = (a*x - sin(pi*x))*sin(pi*y)*sin(pi*z)
                        for i in range(4):
                            self.assertAlmostEqual(delA(xyzt)[i], delA_ref[i])

    def test_del2A(self):
        """Test the optimized boundary condition function Laplacians."""
        del2A_ref = [None, None, None, None]
        for x in xx:
            for y in yy:
                for z in zz:
                    for t in tt:
                        xyzt = (x, y, z, t)
                        del2A_ref[0] = (
                            pi**2*(t - 1)*sin(pi*x)*sin(pi*y)*sin(pi*z))
                        del2A_ref[1] = (
                            -pi**2*(a*t*x + (1 - t)*sin(pi*x))*sin(pi*y) *
                            sin(pi*z))
                        del2A_ref[2] = (
                            -pi**2*(a*t*x + (1 - t)*sin(pi*x))*sin(pi*y) *
                            sin(pi*z))
                        del2A_ref[3] = 0
                        for i in range(4):
                            self.assertAlmostEqual(del2A(xyzt)[i],
                                                   del2A_ref[i])


if __name__ == '__main__':
    unittest.main()
