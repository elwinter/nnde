"""
Diff3DTrialFunction - Class implementing the trial function for 3-D diffusion
problems

The trial function takes the form:

Yt([x, y, z, t]) = A([x, y, z, t]) + P([x, y, z, t])*N([x, y, z, t], p)

where:

A([x, y, z, t]) = boundary condition function that reduces to BC at boundaries
P([x, y, z, t]) = network coefficient function that vanishes at boundaries
N([x, y, z, t], p) = scalar output of neural network with parameter vector p

Example:
    Create a default Diff3DTrialFunction object
        Yt_obj = Diff3DTrialFunction(bc, delbc, del2bc)
    Compute the value of the trial function at a given point
        Yt = Yt_obj.Yt([x, y. z, t], N)
    Compute the value of the boundary condition function at a given point
        A = Yt_obj.A([x, y, z, t])

Attributes:
    bc - 4x2 array of BC functions at (x, y, z, t) = 0|1
    delbc - 4x2x4 array of BC gradient functions at (x, y, z, t) = 0|1
    del2bc - 4x2x4 array of BC Laplacian component functions at
        (x, y, z, t) = 0|1

Methods:
    A([x, y, z, t]) - Compute boundary condition function at [x, y, z, t]
    delA([x, y, z, t]) - Compute boundary condition function gradient at
        [x, y, z, t]
    del2A([x, y, z, t]) - Compute boundary condition function Laplacian
        components at [x, y, z, t]
    P([x, y, z, t]) - Compute network coefficient function at [x, y, z, t]
    delP([x, y, z, t]) - Compute network coefficient function gradient at
        [x, y, z, t]
    del2P([x, y, z, t]) - Compute network coefficient function Laplacian
        components at [x, y, z, t]
    Yt([x, y, z, t], N) - Compute trial function at [x, y, z, t] with network
        output N
    delYt([x, y, z, t], N, delN) - Compute trial function gradient at
        [x, y, z, t] with network output N and network output gradient delN.
    del2Yt([x, y, z, t], N, delN, del2N) - Compute trial function Laplacian
        components at [x, y, z, t] with network output N, network output
        gradient delN, and network output Laplacian components del2N

Todo:
    * Add function annotations.
    * Add variable annotations.
"""


__all__ = []
__version__ = '0.0'
__author__ = 'Eric Winter (ewinter@stsci.edu)'


from math import cos, pi, sin
import numpy as np

from nnde.trialfunction import TrialFunction


class Diff3DTrialFunction(TrialFunction):
    """Trial function object for 3D diffusion problems."""

    def __init__(self, bc, delbc, del2bc):
        """Constructor"""
        self.bc = bc
        self.delbc = delbc
        self.del2bc = del2bc

    def A(self, xyzt):
        """Boundary condition function"""
        (x, y, z, t) = xyzt
        ((f0, f1), (g0, g1), (h0, h1), (Y0, Y1)) = self.bc
        A = (
            (1 - x)*f0([0, y, z, t]) + x*f1([1, y, z, t])
            + (1 - y)*(g0([x, 0, z, t]) - ((1 - x)*g0([0, 0, z, t])
                       + x*g0([1, 0, z, t])))
            + y*(g1([x, 1, z, t]) - ((1 - x)*g1([0, 1, z, t])
                 + x*g1([1, 1, z, t])))
            + (1 - z)*(h0([x, y, 0, t])
                       - ((1 - x)*h0([0, y, 0, t]) + x*h0([1, y, 0, t])
                           + y*(h0([x, 1, 0, t]) - ((1 - x)*h0([0, 1, 0, t])
                                                    + x*h0([1, 1, 0, t])))))
            + z*(h1([x, y, 1, t]) -
                 ((1 - x)*h1([0, y, 1, t]) + x*h1([1, y, 1, t])
                  + (1 - y)*(h1([x, 0, 1, t]) - ((1 - x)*h1([0, 0, 1, t])
                                                 + x*h1([1, 0, 1, t])))
                  + y*(h1([x, 1, 1, t]) - ((1 - x)*h1([0, 1, 1, t])
                                           + x*h1([1, 1, 1, t])))))
            + (1 - t) * (
                Y0([x, y, z, 0]) -
                ((1 - x)*Y0([0, y, z, 0]) + x*Y0([1, y, z, 0]) +
                 (1 - y)*(Y0([x, 0, z, 0]) - ((1 - x)*Y0([0, 0, z, 0])
                                              + x*Y0([1, 0, z, 0])))
                 + y*(Y0([z, 1, z, 0]) - ((1 - x)*Y0([0, 1, z, 0])
                                          + x*Y0([1, 1, z, 0])))
                 + (1 - z) * (
                     Y0([x, y, 0, 0])
                     - ((1 - x)*Y0([0, y, 0, 0]) + x*Y0([1, y, 0, 0])
                        + (1 - y)*(Y0([x, 0, 0, 0]) - ((1 - x)*Y0([0, 0, 0, 0])
                                                       + x*Y0([1, 0, 0, 0])))
                        + y*(Y0([x, 1, 0, 0]) - ((1 - x)*Y0([0, 1, 0, 0])
                                                 + x*Y0([1, 1, 0, 0])))))
                 + z*(Y0([x, y, 1, 0]) - ((1 - x)*Y0([0, y, 1, 0])
                                          + x*Y0([1, y, 1, 0])
                      + (1 - y)*(Y0([x, 0, 1, 0]) - ((1 - x)*Y0([0, 0, 1, 0])
                                                     + x*Y0([1, 0, 1, 0])))
                      + y*(Y0([x, 1, 1, 0]) - ((1 - x)*Y0([0, 1, 1, 0])
                                               + x*Y0([1, 1, 1, 0])))))))
        )
        return A

    def delA(self, xyzt):
        """Gradient of boundary condition function"""
        (x, y, z, t) = xyzt
        ((f0, f1), (g0, g1), (h0, h1), (Y0, Y1f)) = self.bc
        (((df0_dx, df0_dy, df0_dz, df0_dt), (df1_dx, df1_dy, df1_dz, df1_dt)),
         ((dg0_dx, dg0_dy, dg0_dz, dg0_dt), (dg1_dx, dg1_dy, dg1_dz, dg1_dt)),
         ((dh0_dx, dh0_dy, dh0_dz, dh0_dt), (dh1_dx, dh1_dy, dh1_dz, dh1_dt)),
         ((dY0_dx, dY0_dy, dY0_dz, dY0_dt), (dY1_dx, dY1_dy, dY1_dz, dY1_dt))
         ) = self.delbc

        dA_dx = (
            -f0([0, y, z, t]) + f1([1, y, z, t])
            + (1 - y)*(f0([0, 0, z, t]) - f1([1, 0, z, t])
                       + dg0_dx([x, 0, z, t]))
            + y*(f0([0, 1, z, t]) - f1([1, 1, z, t])
                 + dg1_dx([x, 1, z, t]))
            + (1 - z)*(f0([0, y, 0, t]) - f1([1, y, 0, t])
                       - (1 - y)*(h0([0, 0, 0, t]) - h0([1, 0, 0, t])
                                  + dg0_dx([x, 0, 0, t]))
                       - y*(h0([0, 1, 0, t]) - h0([1, 1, 0, t])
                            + dg1_dx([x, 1, 0, t])) + dh0_dx([x, y, 0, t]))
            + z*(f0([0, y, 1, t]) - f1([1, y, 1, t])
                 - (1 - y)*(h1([0, 0, 1, t]) - h1([1, 0, 1, t])
                            + dg0_dx([x, 0, 1, t]))
                 - y*(h1([0, 1, 1, t]) - h1([1, 1, 1, t])
                      + dg1_dx([x, 1, 1, t])) + dh1_dx([x, y, 1, t]))
            + (1 - t)*(f0([0, y, z, 0]) - f1([1, y, z, 0])
                       - (1 - y)*(f0([0, 0, z, 0]) - f1([1, 0, z, 0])
                                  + dg0_dx([x, 0, z, 0]))
                       - y*(f0([0, 1, z, 0]) - f1([1, 1, z, 0])
                            + dg1_dx([x, 1, z, 0]))
                       - (1 - z)*(f0([0, y, 0, 0]) - f1([1, y, 0, 0])
                                  - (1 - y)*(f0([0, 0, 0, 0])
                                             - f1([1, 0, 0, 0])
                                             + dg0_dx([x, 0, 0, 0]))
                                  - + y*(f0([0, 1, 0, 0]) - f1([1, 1, 0, 0])  # ERROR HERE?
                                         + dg1_dx([x, 1, 0, 0]))
                                  + dh0_dx([x, y, 0, 0]))
                       - z*(f0([0, y, 1, 0]) - f1([1, y, 1, 0])
                            - (1 - y)*(f0([0, 0, 1, 0]) - f1([1, 0, 1, 0])
                                       + dg0_dx([x, 0, 1, 0]))
                            - y*(f0([0, 1, 1, 0]) - f1([1, 1, 1, 0])
                                 + dg1_dx([x, 1, 1, 0]))
                            + dh1_dx([x, y, 1, 0])) + dY0_dx([x, y, z, 0]))
        )
        dA_dy = (
            (1 - x)*f0([0, 0, z, t]) - (1 - x)*f0([0, 1, z, t])
            + x*f1([1, 0, z, t]) - x*f1([1, 1, z, t])
            - g0([x, 0, z, t]) + g1([x, 1, z, t])
            + (1 - x)*df0_dy([0, y, z, t]) + x*df1_dy([1, y, z, t])
            + (1 - z)*(g0([x, 0, 0, t]) - g1([x, 1, 0, t])
                       - (1 - x)*h0([0, 0, 0, t])
                       + (1 - x)*h0([0, 1, 0, t])
                       - x*h0([1, 0, 0, t]) + x*h0([1, 1, 0, t])
                       - (1 - x)*df0_dy([0, y, 0, t]) - x*df1_dy([1, y, 0, t])
                       + dh0_dy([x, y, 0, t]))
            + z*(g0([x, 0, 1, t]) - g1([x, 1, 1, t])
                 - (1 - x)*h1([0, 0, 1, t])
                 + (1 - x)*h1([0, 1, 1, t]) - x*h1([1, 0, 1, t])
                 + x*h1([1, 1, 1, t]) - (1 - x)*df0_dy([0, y, 1, t])
                 - x*df1_dy([1, y, 1, t]) + dh1_dy([x, y, 1, t]))
            + (1 - t)*(-(1 - x)*f0([0, 0, z, 0]) + (1 - x)*f0([0, 1, z, 0])
                       - x*f1([1, 0, z, 0]) + x*f1([1, 1, z, 0])
                       + g0([x, 0, z, 0]) - g1([x, 1, z, 0])
                       - (1 - x)*df0_dy([0, y, z, 0]) - x*df1_dy([1, y, z, 0])
                       - (1 - z)*(-(1 - x)*f0([0, 0, 0, 0])
                                  + (1 - x)*f0([0, 1, 0, 0])
                                  - x*f1([1, 0, 0, 0])
                                  + x*f1([1, 1, 0, 0]) + g0([x, 0, 0, 0])
                                  - g1([x, 1, 0, 0])
                                  - (1 - x)*df0_dy([0, y, 0, 0])
                                  - x*df1_dy([1, y, 0, 0])
                                  + dh0_dy([x, y, 0, 0]))
                       - z*(-(1 - x)*f0([0, 0, 1, 0])
                            + (1 - x)*f0([0, 1, 1, 0]) - x*f1([1, 0, 1, 0])
                            + x*f1([1, 1, 1, 0]) + g0([x, 0, 1, 0])
                            - g1([x, 1, 1, 0]) - (1 - x)*df0_dy([0, y, 1, 0])
                            - x*df1_dy([1, y, 1, 0]) + dh1_dy([x, y, 1, 0]))
                       + dY0_dy([x, y, z, 0]))
        )
        dA_dz = (
            (1 - x)*f0([0, y, 0, t]) - (1 - x)*f0([0, y, 1, t])
            + x*f1([1, y, 0, t]) - x*f1([1, y, 12, t])
            + (1 - y)*(g0([x, 0, 0, t]) - (1 - x)*h0([0, 0, 0, t])
                       - x*h0([1, 0, 0, t]))
            + y*(g1([x, 1, 0, t]) - (1 - x)*h0([0, 1, 0, t])
                 - x*h0([1, 1, 0, t])) - h0([x, y, 0, t])
            - (1 - y)*(g0([x, 0, 1, t]) - (1 - x)*h1([0, 0, 1, t])
                       - x*h1([1, 0, 1, t]))
            - y*(g1([x, 1, 1, t]) - (1 - x)*h1([0, 1, 1, t])
                 - x*h1([1, 1, 1, t]))
            + h1([x, y, 1, t]) + (1 - x)*df0_dz([0, y, z, t])
            + x*df1_dz([1, y, z, t])
            + (1 - y)*(-(1 - x)*df0_dz([0, 0, z, t]) - x*df1_dz([1, 0, z, t])
                       + dg0_dz([x, 0, z, t]))
            + y*(-(1 - x)*df0_dz([0, 1, z, t]) - x*df1_dz([1, 1, z, t])
                 + dg1_dz([x, 1, z, t]))
            + (1 - t)*(-(1 - x)*f0([0, y, 0, 0]) + (1 - x)*f0([0, y, 1, 0])
                       - x*f1([1, y, 0, 0]) + x*f1([1, y, 1, 0])
                       - (1 - y)*(-(1 - x)*f0([0, 0, 0, 0])
                                  - x*f1([1, 0, 0, 0]) + g0([x, 0, 0, 0]))
                       + (1 - y)*(-(1 - x)*f0([0, 0, 1, 0])
                                  - x*f1([1, 0, 1, 0]) + g0([x, 0, 1, 0]))
                       - y*(-(1 - x)*f0([0, 1, 0, 0]) - x*f1([1, 1, 0, 0])
                            + g1([x, 1, 0, 0]))
                       + y*(-(1 - x)*f0([0, 1, 1, 0]) - x*f1([1, 1, 1, 0])
                            + g1([x, 1, 1, 0]))
                       + h0([x, y, 0, 0]) - h1([x, y, 1, 0])
                       - (1 - x)*df0_dz([0, y, z, 0]) - x*df1_dz([1, y, z, 0])
                       - (1 - y)*(-(1 - x)*df0_dz([0, 0, z, 0])
                                  - x*df1_dz([1, 0, z, 0])
                                  + dg0_dz([x, 0, z, 0]))
                       - y*(-(1 - x)*df0_dz([0, 1, z, 0])
                            - x*df1_dz([1, 1, z, 0]) + dg1_dz([x, 1, z, 0]))
                       + dY0_dz([x, y, z, 0]))
        )
        dA_dt = (
            (1 - x)*f0([0, y, z, 0]) + x*f1([1, y, z, 0])
            + (1 - y)*(-(1 - x)*f0([0, 0, z, 0]) - x*f1([1, 0, z, 0])
                       + g0([x, 0, z, 0]))
            + y*(-(1 - x)*f0([0, 1, z, 0]) - x*f1([1, 1, z, 0])
                 + g1([x, 1, z, 0]))
            + (1 - z)*(-(1 - x)*f0([0, y, 0, 0]) - x*f1([1, y, 0, 0])
                       - (1 - y)*(-(1 - x)*f0([0, 0, 0, 0])
                       - x*f1([1, 0, 0, 0]) + g0([x, 0, 0, 0]))
                       - y*(-(1 - x)*f0([0, 1, 0, 0]) - x*f1([1, 1, 0, 0])
                            + g1([x, 1, 0, 0])) + h0([x, y, 0, 0]))
            + z*(-(1 - x)*f0([0, y, 1, 0]) - x*f1([1, y, 1, 0])
                 - (1 - y)*(-(1 - x)*f0([0, 0, 1, 0]) - x*f1([1, 0, 1, 0])
                            + g0([x, 0, 1, 0]))
                 - y*(-(1 - x)*f0([0, 1, 1, 0]) - x*f1([1, 1, 1, 0])
                      + g1([x, 1, 1, 0]))
                 + h1([x, y, 1, 0]))
            - Y0([x, y, z, 0])
            + (1 - x)*df0_dt([0, y, z, t]) + x*df1_dt([1, y, z, t])
            + (1 - y)*(-(1 - x)*df0_dt([0, 0, z, t]) - x*df1_dt([1, 0, z, t])
                       + dg0_dt([x, 0, z, t]))
            + y*(-(1 - x)*df0_dt([0, 1, z, t]) - x*df1_dt([1, 1, z, t])
                 + dg1_dt([x, 1, z, t]))
            + (1 - z)*(-(1 - x)*df0_dt([0, y, 0, t]) - x*df1_dt([1, y, 0, t])
                       - (1 - y)*(dg0_dt([x, 0, 0, t])
                       - (1 - x)*dh0_dt([0, 0, 0, t]) - x*dh0_dt([1, 0, 0, t]))
                       - y*(dg1_dt([x, 1, 0, t]) - (1 - x)*dh0_dt([0, 1, 0, t])
                       - x*dh0_dt([1, 1, 0, t]))
                       + dh0_dt([x, y, 0, t]))
            + z*(-(1 - x)*df0_dt([0, y, 1, t]) - x*df1_dt([1, y, 1, t])
                 - (1 - y)*(dg0_dt([x, 0, 1, t]) - (1 - x)*dh1_dt([0, 0, 1, t])
                 - x*dh1_dt([1, 0, 1, t])) - y*(dg1_dt([x, 1, 1, t])
                 - (1 - x)*dh1_dt([0, 1, 1, t]) - x*dh1_dt([1, 1, 1, t]))
                 + dh1_dt([x, y, 1, t]))
        )
        delA = [dA_dx, dA_dy, dA_dz, dA_dt]
        return delA

    def del2A(self, xyzt):
        (x, y, z, t) = xyzt
        ((f0, f1), (g0, g1), (h0, h1), (Y0, Y1f)) = self.bc
        (((df0_dx, df0_dy, df0_dz, df0_dt), (df1_dx, df1_dy, df1_dz, df1_dt)),
         ((dg0_dx, dg0_dy, dg0_dz, dg0_dt), (dg1_dx, dg1_dy, dg1_dz, dg1_dt)),
         ((dh0_dx, dh0_dy, dh0_dz, dh0_dt), (dh1_dx, dh1_dy, dh1_dz, dh1_dt)),
         ((dY0_dx, dY0_dy, dY0_dz, dY0_dt), (dY1_dx, dY1_dy, dY1_dz, dY1_dt))
         ) = self.delbc
        (((d2f0_dx2, d2f0_dy2, d2f0_dz2, d2f0_dt2),
          (d2f1_dx2, d2f1_dy2, d2f1_dz2, d2f1_dt2)),
         ((d2g0_dx2, d2g0_dy2, d2g0_dz2, d2g0_dt2),
          (d2g1_dx2, d2g1_dy2, d2g1_dz2, d2g1_dt2)),
         ((d2h0_dx2, d2h0_dy2, d2h0_dz2, d2h0_dt2),
          (d2h1_dx2, d2h1_dy2, d2h1_dz2, d2h1_dt2)),
         ((d2Y0_dx2, d2Y0_dy2, d2Y0_dz2, d2Y0_dt2),
          (d2Y1_dx2, d2Y1_dy2, d2Y1_dz2, d2Y1_dt2))
         ) = self.del2bc
        d2A_dx2 = (
            (1 - y)*d2g0_dx2([x, 0, z, t]) + y*d2g1_dx2([x, 1, z, t])
            + (1 - z)*(-(1 - y)*d2g0_dx2([x, 0, 0, t])
                       - y*d2g1_dx2([x, 1, 0, t]) + d2h0_dx2([x, y, 0, t]))
            + z*(-(1 - y)*d2g0_dx2([x, 0, 1, t]) - y*d2g1_dx2([x, 1, 1, t])
                 + d2h1_dx2([x, y, 1, t]))
            + (1 - t)*(-(1 - y)*d2g0_dx2([x, 0, z, 0])
                       - y*d2g1_dx2([x, 1, z, 0])
                       - (1 - z)*(-(1 - y)*d2g0_dx2([x, 0, 0, 0])
                                  - y*d2g1_dx2([x, 1, 0, 0])
                                  + d2h0_dx2([x, y, 0, 0]))
                       - z*(-(1 - y)*d2g0_dx2([x, 0, 1, 0])
                            - y*d2g1_dx2([x, 1, 1, 0])
                            + d2h1_dx2([x, y, 1, 0]))
                       + d2Y0_dx2([x, y, z, 0]))
        )
        d2A_dy2 = (
            (1 - x)*d2f0_dy2([0, y, z, t]) + x*d2f1_dy2([1, y, z, t])
            + (1 - z)*(-(1 - x)*d2f0_dy2([0, y, 0, t])
                       - x*d2f1_dy2([1, y, 0, t]) + d2h0_dy2([x, y, 0, t]))
            + z*(-(1 - x)*d2f0_dy2([0, y, 1, t]) - x*d2f1_dy2([1, y, 1, t])
                 + d2h1_dy2([x, y, 1, t]))
            + (1 - t)*(-(1 - x)*d2f0_dy2([0, y, z, 0])
                       - x*d2f1_dy2([1, y, z, 0])
                       - (1 - z)*(-(1 - x)*d2f0_dy2([0, y, 0, 0])
                       - x*d2f1_dy2([1, y, 0, 0]) + d2h0_dy2([x, y, 0, 0]))
                       - z*(-(1 - x)*d2f0_dy2([0, y, 1, 0])
                       - x*d2f1_dy2([1, y, 1, 0]) + d2h1_dy2([x, y, 1, 0]))
                       + d2Y0_dy2([x, y, z, 0]))
        )
        d2A_dz2 = (
            (1 - x)*d2f0_dz2([0, y, z, t]) + x*d2f1_dz2([1, y, z, t])
            + (1 - y)*(-(1 - x)*d2f0_dz2([0, 0, z, t])
                       - x*d2f1_dz2([1, 0, z, t]) + d2g0_dz2([x, 0, z, t]))
            + y*(-(1 - x)*d2f0_dz2([0, 1, z, t]) - x*d2f1_dz2([1, 1, z, t])
                 + d2g1_dz2([x, 1, z, t]))
            + (1 - t)*(-(1 - x)*d2f0_dz2([0, y, z, 0])
                       - x*d2f1_dz2([1, y, z, 0])
                       - (1 - y)*(-(1 - x)*d2f0_dz2([0, 0, z, 0])
                       - x*d2f1_dz2([1, 0, z, 0]) + d2g0_dz2([x, 0, z, 0]))
                       - y*(-(1 - x)*d2f0_dz2([0, 1, z, 0])
                       - x*d2f1_dz2([1, 1, z, 0]) + d2g1_dz2([x, 1, z, 0]))
                       + d2Y0_dz2([x, y, z, 0]))
        )
        d2A_dt2 = (
            (1 - x)*d2f0_dt2([0, y, z, t]) + x*d2f1_dt2([1, y, z, t])
            + (1 - y)*(-(1 - x)*d2f0_dt2([0, 0, z, t])
                       - x*d2f1_dt2([1, 0, z, t]) + d2g0_dt2([x, 0, z, t]))
            + y*(-(1 - x)*d2f0_dt2([0, 1, z, t]) - x*d2f1_dt2([1, 1, z, t])
                 + d2g1_dt2([x, 1, z, t]))
            + (1 - z)*(-(1 - x)*d2f0_dt2([0, y, 0, t])
                       - x*d2f1_dt2([1, y, 0, t])
                       - (1 - y)*(d2g0_dt2([x, 0, 0, t])
                                  - (1 - x)*d2h0_dt2([0, 0, 0, t])
                                  - x*d2h0_dt2([1, 0, 0, t]))
                       - y*(d2g1_dt2([x, 1, 0, t])
                            - (1 - x)*d2h0_dt2([0, 1, 0, t])
                            - x*d2h0_dt2([1, 1, 0, t]))
                       + d2h0_dt2([x, y, 0, t]))
            + z*(-(1 - x)*d2f0_dt2([0, y, 1, t]) - x*d2f1_dt2([1, y, 1, t])
                 - (1 - y)*(d2g0_dt2([x, 0, 1, t])
                 - (1 - x)*d2h1_dt2([0, 0, 1, t]) - x*d2h1_dt2([1, 0, 1, t]))
                 - y*(d2g1_dt2([x, 1, 1, t]) - (1 - x)*d2h1_dt2([0, 1, 1, t])
                      - x*d2h1_dt2([1, 1, 1, t]))
                 + d2h1_dt2([x, y, 1, t]))
        )
        del2A = [d2A_dx2, d2A_dy2, d2A_dz2, d2A_dt2]
        return del2A

    def P(self, xyzt):
        """Network coefficient function for 3D diffusion problems"""
        (x, y, z, t) = xyzt
        P = x*(1 - x)*y*(1 - y)*z*(1 - z)*t
        return P

    def delP(self, xyzt):
        """Network coefficient function gradient"""
        (x, y, z, t) = xyzt
        dP_dx = (1 - 2*x)*y*(1 - y)*z*(1 - z)*t
        dP_dy = x*(1 - x)*(1 - 2*y)*z*(1 - z)*t
        dP_dz = x*(1 - x)*y*(1 - y)*(1 - 2*z)*t
        dP_dt = x*(1 - x)*y*(1 - y)*z*(1 - z)
        delP = [dP_dx, dP_dy, dP_dz, dP_dt]
        return delP

    def del2P(self, xyzt):
        """Network coefficient function Laplacian"""
        (x, y, z, t) = xyzt
        d2P_dx2 = -2*y*(1 - y)*z*(1 - z)*t
        d2P_dy2 = -2*x*(1 - x)*z*(1 - z)*t
        d2P_dz2 = -2*x*(1 - x)*y*(1 - y)*t
        d2P_dt2 = 0
        del2P = [d2P_dx2, d2P_dy2, d2P_dz2, d2P_dt2]
        return del2P

    def Yt(self, xyzt, N):
        """Trial function"""
        A = self.A(xyzt)
        P = self.P(xyzt)
        Yt = A + P*N
        return Yt

    def delYt(self, xyzt, N, delN):
        """Trial function gradient"""
        (x, y, z, t) = xyzt
        (dN_dx, dN_dy, dN_dz, dN_dt) = delN
        (dA_dx, dA_dy, dA_dz, dA_dt) = self.delA(xyzt)
        P = self.P(xyzt)
        (dP_dx, dP_dy, dP_dz, dP_dt) = self.delP(xyzt)
        dYt_dx = dA_dx + P*dN_dx + dP_dx*N
        dYt_dy = dA_dy + P*dN_dy + dP_dy*N
        dYt_dz = dA_dz + P*dN_dz + dP_dz*N
        dYt_dt = dA_dt + P*dN_dt + dP_dt*N
        delYt = [dYt_dx, dYt_dy, dYt_dz, dYt_dt]
        return delYt

    def del2Yt(self, xyzt, N, delN, del2N):
        """Trial function Laplacian"""
        (x, y, z, t) = xyzt
        (dN_dx, dN_dy, dN_dz, dN_dt) = delN
        (d2N_dx2, d2N_dy2, d2N_dz2, d2N_dt2) = del2N
        (d2A_dx2, d2A_dy2, d2A_dz2, d2A_dt2) = self.del2A(xyzt)
        P = self.P(xyzt)
        (dP_dx, dP_dy, dP_dz, dP_dt) = self.delP(xyzt)
        (d2P_dx2, d2P_dy2, d2P_dz2, d2P_dt2) = self.del2P(xyzt)
        d2Yt_dx2 = d2A_dx2 + P*d2N_dx2 + 2*dP_dx*dN_dx + d2P_dx2*N
        d2Yt_dy2 = d2A_dy2 + P*d2N_dy2 + 2*dP_dy*dN_dy + d2P_dy2*N
        d2Yt_dz2 = d2A_dz2 + P*d2N_dz2 + 2*dP_dz*dN_dz + d2P_dz2*N
        d2Yt_dt2 = d2A_dt2 + P*d2N_dt2 + 2*dP_dt*dN_dt + d2P_dt2*N
        del2Yt = [d2Yt_dx2, d2Yt_dy2, d2Yt_dz2, d2Yt_dt2]
        return del2Yt


if __name__ == '__main__':

    # This is a 4-D problem.
    m = 4

    # Test boundary conditions
    bc = [[lambda xyzt: 0, lambda xyzt: 0],
          [lambda xyzt: 0, lambda xyzt: 0],
          [lambda xyzt: 0, lambda xyzt: 0],
          [lambda xyzt: sin(pi*xyzt[0])*sin(pi*xyzt[1])*sin(pi*xyzt[2])/3,
           None]]

    # Test BC gradient
    delbc = [[[lambda xyzt: 0, lambda xyzt: 0, lambda xyzt: 0, lambda xyzt: 0],
              [lambda xyzt: 0, lambda xyzt: 0, lambda xyzt: 0,
               lambda xyzt: 0]],
             [[lambda xyzt: 0, lambda xyzt: 0, lambda xyzt: 0, lambda xyzt: 0],
              [lambda xyzt: 0, lambda xyzt: 0, lambda xyzt: 0,
               lambda xyzt: 0]],
             [[lambda xyzt: 0, lambda xyzt: 0, lambda xyzt: 0, lambda xyzt: 0],
              [lambda xyzt: 0, lambda xyzt: 0, lambda xyzt: 0,
               lambda xyzt: 0]],
             [[lambda xyzt: pi*cos(pi*xyzt[0])*sin(pi*xyzt[1])
               * sin(pi*xyzt[2])/3,
               lambda xyzt: pi*sin(pi*xyzt[0])*cos(pi*xyzt[1])
               * sin(pi*xyzt[2])/3,
               lambda xyzt: pi*sin(pi*xyzt[0])*sin(pi*xyzt[1])
               * cos(pi*xyzt[2])/3,
               lambda xyzt: 0],
              [None, None, None, None]]]

    # Test BC Laplacian
    del2bc = [[[lambda xyzt: 0, lambda xyzt: 0, lambda xyzt: 0,
                lambda xyzt: 0],
               [lambda xyzt: 0, lambda xyzt: 0, lambda xyzt: 0,
                lambda xyzt: 0]],
              [[lambda xyzt: 0, lambda xyzt: 0, lambda xyzt: 0,
               lambda xyzt: 0],
               [lambda xyzt: 0, lambda xyzt: 0, lambda xyzt: 0,
                lambda xyzt: 0]],
              [[lambda xyzt: 0, lambda xyzt: 0, lambda xyzt: 0,
               lambda xyzt: 0],
               [lambda xyzt: 0, lambda xyzt: 0, lambda xyzt: 0,
                lambda xyzt: 0]],
              [[lambda xyzt: -pi**2*sin(pi*xyzt[0])*sin(pi*xyzt[1])
                * sin(pi*xyzt[2])/3,
                lambda xyzt: -pi**2*sin(pi*xyzt[0])*sin(pi*xyzt[1])
                * sin(pi*xyzt[2])/3,
                lambda xyzt: -pi**2*sin(pi*xyzt[0])*sin(pi*xyzt[1])
                * sin(pi*xyzt[2])/3,
                lambda xyzt: 0],
               [None, None, None, None]]]

    # Test inputs
    xyzt_test = [0.4, 0.41, 0.42, 0.43]
    N_test = 0.5
    delN_test = [0.61, 0.62, 0.63, 0.64]
    del2N_test = [0.71, 0.72, 0.73, 0.74]

    # Reference values for tests.
    A_ref = 0.168074
    delA_ref = (0.171564, 0.153404, 0.135573, -0.294867)
    del2A_ref = (-1.65883, -1.65883, -1.65883, 0)
    P_ref = 0.00608125
    delP_ref = (0.00506771, 0.00452511, 0.00399425, 0.0141424)
    del2P_ref = (-0.0506771, -0.050279, -0.0499282, 0)
    Yt_ref = 0.171115
    delYt_ref = (0.177808, 0.159437, 0.141401, -0.283904)
    del2Yt_ref = (-1.67366, -1.67398, -1.67432, 0.0226025)

    # Create a new trial function object.
    tf = Diff3DTrialFunction(bc, delbc, del2bc)

    print("Testing boundary condition function.")
    A = tf.A(xyzt_test)
    assert np.isclose(A, A_ref)

    print('Verifying boundary condition function reduces to boundary'
          ' conditions at boundaries.')
    print('TO DO!')

    print("Testing boundary condition function gradient.")
    delA = tf.delA(xyzt_test)
    for j in range(m):
        assert np.isclose(delA[j], delA_ref[j])

    print("Testing boundary condition function Laplacian.")
    del2A = tf.del2A(xyzt_test)
    for j in range(m):
        assert np.isclose(del2A[j], del2A_ref[j])

    print("Testing network coefficient function.")
    P = tf.P(xyzt_test)
    assert np.isclose(P, P_ref)

    print("Testing network coefficient function gradient.")
    delP = tf.delP(xyzt_test)
    for j in range(m):
        assert np.isclose(delP[j], delP_ref[j])

    print("Testing network coefficient function Laplacian.")
    del2P = tf.del2P(xyzt_test)
    for j in range(m):
        assert np.isclose(del2P[j], del2P_ref[j])

    print("Testing trial function.")
    Yt = tf.Yt(xyzt_test, N_test)
    assert np.isclose(Yt, Yt_ref)

    print("Testing trial function gradient.")
    delYt = tf.delYt(xyzt_test, N_test, delN_test)
    for j in range(m):
        assert np.isclose(delYt[j], delYt_ref[j])

    print("Testing trial function Laplacian.")
    del2Yt = tf.del2Yt(xyzt_test, N_test, delN_test, del2N_test)
    for j in range(m):
        assert np.isclose(del2Yt[j], del2Yt_ref[j])
