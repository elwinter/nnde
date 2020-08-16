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


__all__ = []
__version__ = '0.0'
__author__ = 'Eric Winter (ewinter@stsci.edu)'


from math import cos, pi, sin
import numpy as np

from nnde.trialfunction import TrialFunction


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


if __name__ == '__main__':

    # This is a 2-D problem.
    m = 2

    # Test boundary conditions
    bc = [[lambda xt: 0, lambda xt: 0],
          [lambda xt: sin(pi*xt[0]), None]]

    # Test BC gradient
    delbc = [[[lambda xt: 0, lambda xt: 0], [lambda xt: 0, lambda xt: 0]],
             [[lambda xt: pi*cos(pi*xt[0]), lambda xt: 0], [None, None]]]

    # Test BC Laplacian
    del2bc = [[[lambda xt: 0, lambda xt: 0], [lambda xt: 0, lambda xt: 0]],
              [[lambda xt: -pi**2*sin(pi*xt[0]), lambda xt: 0], [None, None]]]

    # Test inputs.
    xt_test = [0.4, 0.41]
    N_test = 0.5
    delN_test = [0.61, 0.62]
    del2N_test = [0.71, 0.72]

    # Reference values for tests.
    A_ref = 0.5611233446141407
    delA_ref = (0.572775, -0.951057)
    del2A_ref = (-5.53807, 0)
    P_ref = 0.0984
    delP_ref = (0.082, 0.24)
    del2P_ref = [-0.82, 0]
    Yt_ref = 0.610323
    delYt_ref = [0.673799, -0.770049]
    del2Yt_ref = [-5.77816, 0.368448]

    # Create a new trial function object.
    tf = Diff1DTrialFunction(bc, delbc, del2bc)

    print("Testing boundary condition function.")
    A = tf.A(xt_test)
    assert np.isclose(A, A_ref)

    print("Testing boundary condition function gradient.")
    delA = tf.delA(xt_test)
    for j in range(m):
        assert np.isclose(delA[j], delA_ref[j])

    print("Testing boundary condition function Laplacian.")
    del2A = tf.del2A(xt_test)
    for j in range(m):
        assert np.isclose(del2A[j], del2A_ref[j])

    print("Testing network coefficient function.")
    P = tf.P(xt_test)
    assert np.isclose(P, P_ref)

    print("Testing network coefficient function gradient.")
    delP = tf.delP(xt_test)
    for j in range(m):
        assert np.isclose(delP[j], delP_ref[j])

    print("Testing network coefficient function Laplacian.")
    del2P = tf.del2P(xt_test)
    for j in range(m):
        assert np.isclose(del2P[j], del2P_ref[j])

    print("Testing trial function.")
    Yt = tf.Yt(xt_test, N_test)
    assert np.isclose(Yt, Yt_ref)

    print("Testing trial function gradient.")
    delYt = tf.delYt(xt_test, N_test, delN_test)
    for j in range(m):
        assert np.isclose(delYt[j], delYt_ref[j])

    print("Testing trial function Laplacian.")
    del2Yt = tf.del2Yt(xt_test, N_test, delN_test, del2N_test)
    for j in range(m):
        assert np.isclose(del2Yt[j], del2Yt_ref[j])
