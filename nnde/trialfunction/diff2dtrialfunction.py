"""
Diff2DTrialFunction - Class implementing the trial function for 2-D diffusion
problems

The trial function takes the form:

Yt([x, y, t]) = A([x, y, t]) + P([x, y, t])*N([x, y, t], p)

where:

A([x, y, t]) = boundary condition function that reduces to BC at boundaries
P([x, y, t]) = network coefficient function that vanishes at boundaries
N([x, y, t], p) = scalar output of neural network with parameter vector p

Example:
    Create a default Diff2DTrialFunction object from boundary conditions
        Yt_obj = Diff2DTrialFunction(bc, delbc, del2bc)
    Compute the value of the trial function at a given point
        Yt = Yt_obj.Yt([x, y, t], N)
    Compute the value of the boundary condition function at a given point
        A = Yt_obj.A([x, y, t])

Attributes:
    bc - 3x2 array of BC functions at (x, y, t) = 0|1
    delbc - 3x2x3 array of BC gradient functions at (x, y, t) = 0|1
    del2bc - 3x2x3 array of BC Laplacian component functions at (x, y, t) = 0|1

Methods:
    A([x, y, t]) - Compute boundary condition function at [x, y, t]
    delA([x, y, t]) - Compute boundary condition function gradient at [x, y, t]
    del2A([x, y, t]) - Compute boundary condition function Laplacian components
        at [x, y, t]
    P([x, y, t]) - Compute network coefficient function at [x, y, t]
    delP([x, y, t]) - Compute network coefficient function gradient at
        [x, y, t]
    del2P([x, y, t]) - Compute network coefficient function Laplacian
        components at [x, y, t]
    Yt([x, y, t], N) - Compute trial function at [x, y, t] with network output
        N
    delYt([x, y, t], N, delN) - Compute trial function gradient at [x, y, t]
        with network output N and network output gradient delN.
    del2Yt([x, y, t], N, delN, del2N) - Compute trial function Laplacian
        components at [x, y, t] with network output N, network output gradient
        delN, and network output Laplacian components del2N

Todo:
    * Add function annotations.
    * Add variable annotations.
"""


from math import cos, pi, sin
import numpy as np

from nnde.trialfunction.trialfunction import TrialFunction


class Diff2DTrialFunction(TrialFunction):
    """Trial function object for 2-D diffusion problems."""

    def __init__(self, bc, delbc, del2bc):
        """Constructor"""
        self.bc = bc
        self.delbc = delbc
        self.del2bc = del2bc

    def A(self, xyt):
        """Boundary condition function"""
        (x, y, t) = xyt
        ((f0, f1), (g0, g1), (Y0, Y1)) = self.bc
        A = (
             (1 - x)*f0([0, y, t]) + x*f1([1, y, t])
             + (1 - y)*(g0([x, 0, t]) - ((1 - x)*g0([0, 0, t])
                        + x*g0([1, 0, t])))
             + y*(g1([x, 1, t]) - ((1 - x)*g1([0, 1, t]) + x*g1([1, 1, t])))
             + (1 - t)*(Y0([x, y, 0])
                        - ((1 - x)*Y0([0, y, 0]) + x*Y0([1, y, 0])
                           + (1 - y)*(Y0([x, 0, 0]) - ((1 - x)*Y0([0, 0, 0])
                                                       + x*Y0([1, 0, 0])))
                           + y*(Y0([x, 1, 0]) - ((1 - x)*Y0([0, 1, 0])
                                                 + x*Y0([1, 1, 0])))))
        )
        return A

    def delA(self, xyt):
        """Boundary condition function gradient"""
        (x, y, t) = xyt
        ((f0, f1), (g0, g1), (Y0, Y1)) = self.bc
        (((df0_dx, df0_dy, df0_dt), (df1_dx, df1_dy, df1_dt)),
         ((dg0_dx, dg0_dy, dg0_dt), (dg1_dx, dg1_dy, dg1_dt)),
         ((dY0_dx, dY0_dy, dY0_dt), (dY1_dx, dY1_dy, dY1_dt))) = self.delbc
        dA_dx = (
            f1([1, y, t]) - f0([0, y, t])
            + (1 - y)*(f0([0, 0, t]) - f1([1, 0, t]) + dg0_dx([x, 0, t]))
            + y*(f0([0, 1, t]) - f1([1, 1, t]) + dg1_dx([x, 1, t]))
            + (1 - t)*(f0([0, y, 0]) - f1([1, y, 0])
                       - (1 - y)*(f0([0, 0, 0]) - f1([1, 0, 0])
                                  + dg0_dx([x, 0, 0]))
                       - y*(f0([0, 1, 0]) - f1([1, 1, 0]) + dg1_dx([x, 1, 0]))
                       + dY0_dx([x, y, 0]))
        )
        dA_dy = (
            (1 - x)*f0([0, 0, t]) - (1 - x)*f0([0, 1, t])
            + x*f1([1, 0, t]) - x*f1([1, 1, t])
            - g0([x, 0, t]) + g1([x, 1, t])
            + (1 - x)*df0_dy([0, y, t]) + x*df1_dy([1, y, t])
            + (1 - t)*(
                (x - 1)*f0([0, 0, 0]) + f0([0, 1, 0])
                - x*(f0([0, 1, 0]) + f1([1, 0, 0]) - f1([1, 1, 0]))
                + g0([x, 0, 0]) - g1([x, 1, 0])
                - (1 - x)*df0_dy([0, y, 0]) - x*df1_dy([1, y, 0])
                + dY0_dy([x, y, 0])
            )
        )
        dA_dt = (
            (1 - x)*f0([0, y, 0]) + x*f1([1, y, 0])
            + (1 - y)*((x - 1)*f0([0, 0, 0]) - x*f1([1, 0, 0]) + g0([x, 0, 0]))
            + y*((x - 1)*f0([0, 1, 0]) - x*f1([1, 1, 0]) + g1([x, 1, 0]))
            - Y0([x, y, 0]) + (1 - x)*df0_dt([0, y, t]) + x*df1_dt([1, y, t])
            + (1 - y)*((x - 1)*df0_dt([0, 0, t]) - x*df1_dt([1, 0, t])
                       + dg0_dt([x, 0, t]))
            + y*((x - 1)*df0_dt([0, 1, t]) - x*df1_dt([1, 1, t])
                 + dg1_dt([x, 1, t]))
        )
        delA = [dA_dx, dA_dy, dA_dt]
        return delA

    def del2A(self, xyt):
        """Laplacian of boundary condition function"""
        (x, y, t) = xyt
        ((f0, f1), (g0, g1), (Y0, Y1)) = self.bc
        (((df0_dx, df0_dy, df0_dt), (df1_dx, df1_dy, df1_dt)),
         ((dg0_dx, dg0_dy, dg0_dt), (dg1_dx, dg1_dy, dg1_dt)),
         ((dY0_dx, dY0_dy, dY0_dt), (dY1_dx, dY1_dy, dY1_dt))) = self.delbc
        (
         ((d2f0_dx2, d2f0_dy2, d2f0_dt2), (d2f1_dx2, d2f1_dy2, d2f1_dt2)),
         ((d2g0_dx2, d2g0_dy2, d2g0_dt2), (d2g1_dx2, d2g1_dy2, d2g1_dt2)),
         ((d2Y0_dx2, d2Y0_dy2, d2Y0_dt2), (d2Y1_dx2, d2Y1_dy2, d2Y1_dt2))
         ) = self.del2bc
        d2A_dx2 = (
            (1 - y)*d2g0_dx2([x, 0, t]) + y*d2g1_dx2([x, 1, t])
            + (1 - t)*((y - 1)*d2Y0_dx2([x, 0, 0]) - y*d2Y0_dx2([x, 1, 0])
                       + d2Y0_dx2([x, y, 0]))
        )
        d2A_dy2 = (
            (1 - x)*d2f0_dy2([0, y, t]) + x*d2f1_dy2([1, y, t])
            + (1 - t)*((x - 1)*d2Y0_dy2([0, y, 0]) - x*d2Y0_dy2([1, y, 0])
                       + d2Y0_dy2([x, y, 0]))
        )
        d2A_dt2 = (
            (1 - x)*d2f0_dt2([0, y, t]) + x*d2f1_dt2([1, y, t])
            + (1 - y)*((x - 1)*d2g0_dt2([0, 0, t]) - x*d2g0_dt2([1, 0, t])
                       + d2g0_dt2([x, 0, t]))
            + y*((x - 1)*d2g1_dt2([0, 1, t]) - x*d2g1_dt2([1, 1, t])
                 + d2g1_dt2([x, 1, t]))
        )
        del2A = [d2A_dx2, d2A_dy2, d2A_dt2]
        return del2A

    def P(self, xyt):
        """Network coefficient function"""
        (x, y, t) = xyt
        P = x*(1 - x)*y*(1 - y)*t
        return P

    def delP(self, xyt):
        """Network coefficient function gradient"""
        (x, y, t) = xyt
        dP_dx = (1 - 2*x)*y*(1 - y)*t
        dP_dy = x*(1 - x)*(1 - 2*y)*t
        dP_dt = x*(1 - x)*y*(1 - y)
        delP = [dP_dx, dP_dy, dP_dt]
        return delP

    def del2P(self, xyt):
        """Network coefficient function Laplacian"""
        (x, y, t) = xyt
        d2P_dx2 = -2*y*(1 - y)*t
        d2P_dy2 = -2*x*(1 - x)*t
        d2P_dt2 = 0
        del2P = [d2P_dx2, d2P_dy2, d2P_dt2]
        return del2P

    def Yt(self, xyt, N):
        """Trial function"""
        A = self.A(xyt)
        P = self.P(xyt)
        Yt = A + P*N
        return Yt

    def delYt(self, xyt, N, delN):
        """Trial function gradient"""
        (x, y, t) = xyt
        (dN_dx, dN_dy, dN_dt) = delN
        (dA_dx, dA_dy, dA_dt) = self.delA(xyt)
        P = self.P(xyt)
        (dP_dx, dP_dy, dP_dt) = self.delP(xyt)
        dYt_dx = dA_dx + P*dN_dx + dP_dx*N
        dYt_dy = dA_dy + P*dN_dy + dP_dy*N
        dYt_dt = dA_dt + P*dN_dt + dP_dt*N
        delYt = [dYt_dx, dYt_dy, dYt_dt]
        return delYt

    def del2Yt(self, xyt, N, delN, del2N):
        """Trial function Laplacian"""
        (x, y, t) = xyt
        (dN_dx, dN_dy, dN_dt) = delN
        (d2N_dx2, d2N_dy2, d2N_dt2) = del2N
        (d2A_dx2, d2A_dy2, d2A_dt2) = self.del2A(xyt)
        P = self.P(xyt)
        (dP_dx, dP_dy, dP_dt) = self.delP(xyt)
        (d2P_dx2, d2P_dy2, d2P_dt2) = self.del2P(xyt)
        d2Yt_dx2 = d2A_dx2 + P*d2N_dx2 + 2*dP_dx*dN_dx + d2P_dx2*N
        d2Yt_dy2 = d2A_dy2 + P*d2N_dy2 + 2*dP_dy*dN_dy + d2P_dy2*N
        d2Yt_dt2 = d2A_dt2 + P*d2N_dt2 + 2*dP_dt*dN_dt + d2P_dt2*N
        del2Yt = [d2Yt_dx2, d2Yt_dy2, d2Yt_dt2]
        return del2Yt
