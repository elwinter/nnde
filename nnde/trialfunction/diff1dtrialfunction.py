"""
Diff1DTrialFunction - Class implementing the trial function for 1-D diffusion
problems

The trial function takes the form:

Yt([x, t]) = A([x, t]) + P([x, t])*N([x, t], p)

where:

A([x, t]) = boundary condition function that reduces to BC at boundaries
P([x, t]) = network coefficient function that vanishes at boundaries
N([x, t], p) = scalar output of neural network with parameter vector p

Example:
    Create a default Diff1DTrialFunction object from boundary conditions
        Yt_obj = Diff1DTrialFunction(bcf, delbcf, del2bcf)
    Compute the value of the trial function at a given point
        Yt = Yt_obj.Yt([x, t], N)
    Compute the value of the boundary condition function at a given point
        A = Yt_obj.A([x, t])

Attributes:
    bc - 2x2 array of BC functions at (x,t)=0|1
    delbc - 2x2x2 array of BC gradient functions at (x,t)=0|1
    del2bc - 2x2x2 array of BC Laplacian component functions at (x,t)=0|1

Methods:
    A([x, t]) - Compute boundary condition function at [x, t]
    delA([x, t]) - Compute boundary condition function gradient at [x, t]
    del2A([x, t]) - Compute boundary condition function Laplacian components
        at [x, t]
    P([x, t]) - Compute network coefficient function at [x, t]
    delP([x, t]) - Compute network coefficient function gradient at [x, t]
    del2P([x, t]) - Compute network coefficient function Laplacian components
        at [x, t]
    Yt([x, t], N) - Compute trial function at [x, t] with network output N
    delYt([x, t], N, delN) - Compute trial function gradient at [x, t] with
        network output N and network output gradient delN.
    del2Yt([x, t], N, delN, del2N) - Compute trial function Laplacian
        components at [x, t] with network output N, network output gradient
        delN, and network output Laplacian components del2N

Todo:
    * Add function annotations.
    * Add variable annotations.
"""


from math import cos, pi, sin
import numpy as np

from nnde.trialfunction.trialfunction import TrialFunction


class Diff1DTrialFunction(TrialFunction):
    """Trial function object for 1D diffusion problems."""

    def __init__(self, bc, delbc, del2bc):
        """Constructor"""
        self.bc = bc
        self.delbc = delbc
        self.del2bc = del2bc

    def A(self, xt):
        """Boundary condition function"""
        (x, t) = xt
        ((f0, f1), (Y0, Y1)) = self.bc
        A = (
             (1 - x)*f0([0, t]) + x*f1([1, t])
             + (1 - t)*(Y0([x, 0]) - ((1 - x)*Y0([0, 0]) + x*Y0([1, 0])))
        )
        return A

    def delA(self, xt):
        """Boundary condition function gradient"""
        (x, t) = xt
        ((f0, f1), (Y0, Y1)) = self.bc
        (((df0_dx, df0_dt), (df1_dx, df1_dt)),
         ((dY0_dx, dY0_dt), (dY1_dx, dY1_dt))) = self.delbc
        dA_dx = (
            f1([1, t]) - f0([0, t])
            + (1 - t)*(dY0_dx([x, 0]) + Y0([0, 0]) - Y0([1, 0]))
        )
        dA_dt = (
            (1 - x)*df0_dt([0, t]) + x*df1_dt([1, t])
            - Y0([x, 0]) + (1 - x)*Y0([0, 0]) + x*Y0([1, 0])
        )
        delA = [dA_dx, dA_dt]
        return delA

    def del2A(self, xt):
        """Laplacian of boundary condition function"""
        (x, t) = xt
        ((f0, f1), (Y0, Y1)) = self.bc
        (((df0_dx, df0_dt), (df1_dx, df1_dt)),
         ((dY0_dx, dY0_dt), (dY1_dx, dY1_dt))) = self.delbc
        (((d2f0_dx2, d2f0_dt2), (d2f1_dx2, d2f1_dt2)),
         ((d2Y0_dx2, d2Y0_dt2), (d2Y1_dx2, d2Y1_dt2))) = self.del2bc
        d2A_dx2 = (1 - t)*d2Y0_dx2([x, 0])
        d2A_dt2 = (1 - x)*d2f0_dt2([0, t]) + x*d2f1_dt2([1, t])
        del2A = [d2A_dx2, d2A_dt2]
        return del2A

    def P(self, xt):
        """Network coefficient function"""
        (x, t) = xt
        P = x*(1 - x)*t
        return P

    def delP(self, xt):
        """Network coefficient function gradient"""
        (x, t) = xt
        dP_dx = (1 - 2*x)*t
        dP_dt = x*(1 - x)
        delP = [dP_dx, dP_dt]
        return delP

    def del2P(self, xt):
        """Network coefficient function Laplacian"""
        (x, t) = xt
        d2P_dx2 = -2*t
        d2P_dt2 = 0
        del2P = [d2P_dx2, d2P_dt2]
        return del2P

    def Yt(self, xt, N):
        """Trial function"""
        A = self.A(xt)
        P = self.P(xt)
        Yt = A + P*N
        return Yt

    def delYt(self, xt, N, delN):
        """Trial function gradient"""
        (x, t) = xt
        (dN_dx, dN_dt) = delN
        (dA_dx, dA_dt) = self.delA(xt)
        P = self.P(xt)
        (dP_dx, dP_dt) = self.delP(xt)
        dYt_dx = dA_dx + P*dN_dx + dP_dx*N
        dYt_dt = dA_dt + P*dN_dt + dP_dt*N
        delYt = [dYt_dx, dYt_dt]
        return delYt

    def del2Yt(self, xt, N, delN, del2N):
        """Trial function Laplacian"""
        (x, t) = xt
        (dN_dx, dN_dt) = delN
        (d2N_dx2, d2N_dt2) = del2N
        (d2A_dx2, d2A_dt2) = self.del2A(xt)
        P = self.P(xt)
        (dP_dx, dP_dt) = self.delP(xt)
        (d2P_dx2, d2P_dt2) = self.del2P(xt)
        d2Yt_dx2 = d2A_dx2 + P*d2N_dx2 + 2*dP_dx*dN_dx + d2P_dx2*N
        d2Yt_dt2 = d2A_dt2 + P*d2N_dt2 + 2*dP_dt*dN_dt + d2P_dt2*N
        del2Yt = [d2Yt_dx2, d2Yt_dt2]
        return del2Yt
