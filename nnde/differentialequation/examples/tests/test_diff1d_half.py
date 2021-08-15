"""Tests for the diff1d_half module."""


import unittest

import numpy as np

from nnde.differentialequation.examples.diff1d_half import (
    C, D,
    G,
    f0, f1, Y0, bc,
    df0_dx, df0_dt, df1_dx, df1_dt, dY0_dx, dY0_dt, delbc,
    d2f0_dx2, d2f0_dt2, d2f1_dx2, d2f1_dt2, d2Y0_dx2, d2Y0_dt2, del2bc,
    dG_dY, dG_dY_dx, dG_dY_dt, dG_ddelY,
    dG_d2Y_dx2, dG_d2Y_dt2, dG_ddel2Y,
    A, delA, del2A,
    Ya, dYa_dx, dYa_dt, delYa, d2Ya_dx2, d2Ya_dt2, del2Ya
)


# Grid points for testing.
xx = np.linspace(0, 1, 11)
tt = np.linspace(0, 1, 11)


class TestBuilder(unittest.TestCase):
    """Tests for the diff1d_half module."""

    def test_G(self):
        """Test the differential equation."""
        xt = (0, 0)
        Yt = 0
        delYt = (0, 0)
        del2Yt = (0, 0)
        G_ref = 0
        self.assertAlmostEqual(G(xt, Yt, delYt, del2Yt), G_ref)
        xt = (1, 1)
        Yt = 2
        delYt = (3, 4)
        del2Yt = (5, 6)
        G_ref = delYt[1] - D*del2Yt[0]
        self.assertAlmostEqual(G(xt, Yt, delYt, del2Yt), G_ref)

    def test_f0(self):
        """Test boundary condition at (x, t) = (0, t)."""
        f0_ref = C
        for x in xx:
            for t in tt:
                xt = (x, t)  # x is ignored
                self.assertAlmostEqual(f0(xt), f0_ref)

    def test_f1(self):
        """Test boundary condition at (x, t) = (1, t)."""
        f1_ref = C
        for x in xx:
            for t in tt:
                xt = (x, t)  # x is ignored
                self.assertAlmostEqual(f1(xt), f1_ref)

    def test_Y0(self):
        """Test boundary condition at (x, t) = (x, 0)."""
        Y0_ref = C
        for x in xx:
            for t in tt:
                xt = (x, t)  # t is ignored
                self.assertAlmostEqual(Y0(xt), Y0_ref)

    def test_bc(self):
        """Test that the boundary conditions are continuous at corners."""
        xt = (0, 0)
        self.assertAlmostEqual(bc[0][0](xt), bc[1][0](xt))
        xt = (1, 0)
        self.assertAlmostEqual(bc[0][1](xt), bc[1][0](xt))

    def test_df0_dx(self):
        """Test the first derivative of f0 wrt x."""
        df0_dx_ref = 0
        for x in xx:
            for t in tt:
                xt = (x, t)  # x is ignored
                self.assertAlmostEqual(df0_dx(xt), df0_dx_ref)

    def test_df0_dt(self):
        """Test the first derivative of f0 wrt t."""
        df0_dt_ref = 0
        for x in xx:
            for t in tt:
                xt = (x, t)  # x is ignored
                self.assertAlmostEqual(df0_dt(xt), df0_dt_ref)

    def test_df1_dx(self):
        """Test the first derivative of f1 wrt x."""
        df1_dx_ref = 0
        for x in xx:
            for t in tt:
                xt = (x, t)  # x is ignored
                self.assertAlmostEqual(df1_dx(xt), df1_dx_ref)

    def test_df1_dt(self):
        """Test the first derivative of f1 wrt t."""
        df1_dt_ref = 0
        for x in xx:
            for t in tt:
                xt = (x, t)  # x is ignored
                self.assertAlmostEqual(df1_dt(xt), df1_dt_ref)

    def test_dY0_dx(self):
        """Test the first derivative of Y0 wrt x."""
        dY0_dx_ref = 0
        for x in xx:
            for t in tt:
                xt = (x, t)  # t is ignored
                self.assertAlmostEqual(dY0_dx(xt), dY0_dx_ref)

    def test_dY0_dt(self):
        """Test the first derivative of Y0 wrt t."""
        dY0_dt_ref = 0
        for x in xx:
            for t in tt:
                xt = (x, t)  # t is ignored
                self.assertAlmostEqual(dY0_dt(xt), dY0_dt_ref)

    def test_delbc(self):
        """Test the boundary condition gradients are continuous at corners."""
        xt = (0, 0)
        for k in range(2):
            self.assertAlmostEqual(delbc[0][0][k](xt), delbc[1][0][k](xt))
        xt = (1, 0)
        for k in range(2):
            self.assertAlmostEqual(delbc[0][1][k](xt), delbc[1][0][k](xt))

    def test_d2f0_dx2(self):
        """Test the second derivative of f0 wrt x."""
        d2f0_dx2_ref = 0
        for x in xx:
            for t in tt:
                xt = (x, t)  # x is ignored
                self.assertAlmostEqual(d2f0_dx2(xt), d2f0_dx2_ref)

    def test_d2f0_dt2(self):
        """Test the second derivative of f0 wrt t."""
        d2f0_dt2_ref = 0
        for x in xx:
            for t in tt:
                xt = (x, t)  # x is ignored
                self.assertAlmostEqual(d2f0_dt2(xt), d2f0_dt2_ref)

    def test_d2f1_dx2(self):
        """Test the second derivative of f1 wrt x."""
        d2f1_dx2_ref = 0
        for x in xx:
            for t in tt:
                xt = (x, t)  # x is ignored
                self.assertAlmostEqual(d2f1_dx2(xt), d2f1_dx2_ref)

    def test_d2f1_dt2(self):
        """Test the second derivative of f1 wrt t."""
        d2f1_dt2_ref = 0
        for x in xx:
            for t in tt:
                xt = (x, t)  # x is ignored
                self.assertAlmostEqual(d2f1_dt2(xt), d2f1_dt2_ref)

    def test_d2Y0_dx2(self):
        """Test the second derivative of Y0 wrt x."""
        d2Y0_dx2_ref = 0
        for x in xx:
            for t in tt:
                xt = (x, t)  # t is ignored
                self.assertAlmostEqual(d2Y0_dx2(xt), d2Y0_dx2_ref)

    def test_d2Y0_dt2(self):
        """Test the second derivative of Y0 wrt t."""
        d2Y0_dt2_ref = 0
        for x in xx:
            for t in tt:
                xt = (x, t)  # t is ignored
                self.assertAlmostEqual(d2Y0_dt2(xt), d2Y0_dt2_ref)

    def test_del2bc(self):
        """Test the boundary condition Laplacian are continuous at corners."""
        xt = (0, 0)
        for k in range(2):
            self.assertAlmostEqual(del2bc[0][0][k](xt), del2bc[1][0][k](xt))
        xt = (1, 0)
        for k in range(2):
            self.assertAlmostEqual(del2bc[0][1][k](xt), del2bc[1][0][k](xt))

    def test_dG_dY(self):
        """Test the derivative of G wrt Y."""
        dG_dY_ref = 0  # For all inputs.
        Y = 0.2
        delY = [0.3, 0.4]
        del2Y = [0.5, 0.6]
        for x in xx:
            for t in tt:
                xt = (x, t)
                self.assertAlmostEqual(dG_dY(xt, Y, delY, del2Y), dG_dY_ref)

    def test_dG_dY_dx(self):
        """Test the derivative of G wrt dY/dx."""
        dG_dY_dx_ref = 0  # For all inputs.
        Y = 0.2
        delY = [0.3, 0.4]
        del2Y = [0.5, 0.6]
        for x in xx:
            for t in tt:
                xt = (x, t)
                self.assertAlmostEqual(dG_dY_dx(xt, Y, delY, del2Y),
                                       dG_dY_dx_ref)

    def test_dG_dY_dt(self):
        """Test the derivative of G wrt dY/dt."""
        dG_dY_dt_ref = 1  # For all inputs.
        Y = 0.2
        delY = [0.3, 0.4]
        del2Y = [0.5, 0.6]
        for x in xx:
            for t in tt:
                xt = (x, t)
                self.assertAlmostEqual(dG_dY_dt(xt, Y, delY, del2Y),
                                       dG_dY_dt_ref)

    def test_dG_ddelY(self):
        """Test the derivatives of G wrt gradients."""
        dG_dY_dx_ref = 0  # For all inputs.
        dG_dY_dt_ref = 1  # For all inputs.
        dG_ddelY_ref = [dG_dY_dx_ref, dG_dY_dt_ref]
        Y = 0.2
        delY = [0.3, 0.4]
        del2Y = [0.5, 0.6]
        for x in xx:
            for t in tt:
                xt = (x, t)
                for i in range(2):
                    self.assertAlmostEqual(dG_ddelY[i](xt, Y, delY, del2Y),
                                           dG_ddelY_ref[i])

    def test_dG_d2Y_dx2(self):
        """Test the derivative of G wrt d2Y/dx2."""
        dG_d2Y_dx2_ref = -D  # For all inputs.
        Y = 0.2
        delY = [0.3, 0.4]
        del2Y = [0.5, 0.6]
        for x in xx:
            for t in tt:
                xt = (x, t)
                self.assertAlmostEqual(dG_d2Y_dx2(xt, Y, delY, del2Y),
                                       dG_d2Y_dx2_ref)

    def test_dG_d2Y_dt2(self):
        """Test the derivative of G wrt d2Y/dt2."""
        dG_d2Y_dt2_ref = 0  # For all inputs.
        Y = 0.2
        delY = [0.3, 0.4]
        del2Y = [0.5, 0.6]
        for x in xx:
            for t in tt:
                xt = (x, t)
                self.assertAlmostEqual(dG_d2Y_dt2(xt, Y, delY, del2Y),
                                       dG_d2Y_dt2_ref)

    def test_dG_ddel2Y(self):
        """Test the derivatives of G wrt Laplacian components."""
        dG_d2Y_dx2_ref = -D  # For all inputs.
        dG_d2Y_dt2_ref = 0  # For all inputs.
        dG_ddel2Y_ref = [dG_d2Y_dx2_ref, dG_d2Y_dt2_ref]
        Y = 0.2
        delY = [0.3, 0.4]
        del2Y = [0.5, 0.6]
        for x in xx:
            for t in tt:
                xt = (x, t)
                for i in range(2):
                    self.assertAlmostEqual(dG_ddel2Y[i](xt, Y, delY, del2Y),
                                           dG_ddel2Y_ref[i])

    def test_A(self):
        """Test the optimized boundary condition function."""
        A_ref = C  # For all inputs.
        for x in xx:
            for t in tt:
                xt = (x, t)
                self.assertAlmostEqual(A(xt), A_ref)

    def test_delA(self):
        """Test the optimized boundary condition function gradient."""
        delA_ref = [0, 0]  # For all inputs.
        for x in xx:
            for t in tt:
                xt = (x, t)
                for i in range(2):
                    self.assertAlmostEqual(delA(xt)[i], delA_ref[i])

    def test_del2A(self):
        """Test the optimized boundary condition function Laplacian."""
        del2A_ref = [0, 0]  # For all inputs.
        for x in xx:
            for t in tt:
                xt = (x, t)
                for i in range(2):
                    self.assertAlmostEqual(del2A(xt)[i], del2A_ref[i])

    def test_Ya(self):
        """Test the analytical solution."""
        Ya_ref = C  # For all inputs.
        for x in xx:
            for t in tt:
                xt = (x, t)
                self.assertAlmostEqual(Ya(xt), Ya_ref)

    def test_dYa_dx(self):
        """Test the first derivative of the analytical solution wrt x."""
        dYa_dx_ref = 0  # For all inputs.
        for x in xx:
            for t in tt:
                xt = (x, t)
                self.assertAlmostEqual(dYa_dx(xt), dYa_dx_ref)

    def test_dYa_dt(self):
        """Test the first derivative of the analytical solution wrt t."""
        dYa_dt_ref = 0  # For all inputs.
        for x in xx:
            for t in tt:
                xt = (x, t)
                self.assertAlmostEqual(dYa_dt(xt), dYa_dt_ref)

    def test_delYa(self):
        """Test the analytical solution gradient."""
        delYa_ref = [0, 0]  # For all inputs.
        for x in xx:
            for t in tt:
                xt = (x, t)
                for i in range(2):
                    self.assertAlmostEqual(delYa[i](xt), delYa_ref[i])

    def test_d2Ya_dx2(self):
        """Test the second derivative of the analytical solution wrt x."""
        d2Ya_dx2_ref = 0  # For all inputs.
        for x in xx:
            for t in tt:
                xt = (x, t)
                self.assertAlmostEqual(d2Ya_dx2(xt), d2Ya_dx2_ref)

    def test_d2Ya_dt2(self):
        """Test the second derivative of the analytical solution wrt t."""
        d2Ya_dt2_ref = 0  # For all inputs.
        for x in xx:
            for t in tt:
                xt = (x, t)
                self.assertAlmostEqual(d2Ya_dt2(xt), d2Ya_dt2_ref)

    def test_del2Ya(self):
        """Test the analytical solution Laplacian."""
        del2Ya_ref = [0, 0]  # For all inputs.
        for x in xx:
            for t in tt:
                xt = (x, t)
                for i in range(2):
                    self.assertAlmostEqual(del2Ya[i](xt), del2Ya_ref[i])


if __name__ == '__main__':
    unittest.main()
