import numpy as np

from nnde.trialfunction.trialfunction import TrialFunction


class PDE2BVPTrialFunction(TrialFunction):

    def __init__(self, bc, delbc, del2bc):
        """Constructor"""
        self.bc = bc
        self.delbc = delbc
        self.del2bc = del2bc

    def A(self, xy):
        """Boundary condition function"""
        (x, y) = xy
        ((f0, f1), (g0, g1)) = self.bc
        _A = (
             (1 - x)*f0([0, y]) + x*f1([1, y])
             + (1 - y)*(g0([x, 0]) - ((1 - x)*g0([0, 0]) + x*g0([1, 0])))
             + y*(g1([x, 1]) - ((1 - x)*g1([0, 1]) + x*g1([1, 1])))
        )
        return _A

    def delA(self, xy):
        """Boundary condition function gradient"""
        (x, y) = xy
        ((f0, f1), (g0, g1)) = self.bc
        (((df0_dx, df0_dy), (df1_dx, df1_dy)),
         ((dg0_dx, dg0_dy), (dg1_dx, dg1_dy))) = self.delbc
        dA_dx = (
            -f0([0, y]) + f1([1, y])
            - (-1 + y)*(g0([0, 0]) - g0([1, 0]) + dg0_dx([x, 0]))
            + y*(g1([0, 1]) - g1([1, 1]) + dg1_dx([x, 1]))
        )
        dA_dy = (
            (1 - x)*g0([0, 0]) + x*g0([1, 0]) - g0([x, 0]) + (-1 + x)*g1([0, 1])
            - x*g1([1, 1]) + g1([x, 1]) - (-1 + x)*df0_dy([0, y]) + x*df1_dy([1, y])
        )
        _delA = [dA_dx, dA_dy]
        return _delA

    def del2A(self, xy):
        """Laplacian of boundary condition function"""
        (x, y) = xy
        ((f0, f1), (g0, g1)) = self.bc
        (((df0_dx, df0_dy), (df1_dx, df1_dy)),
         ((dg0_dx, dg0_dy), (dg1_dx, dg1_dy))) = self.delbc
        (((d2f0_dx2, d2f0_dy2), (d2f1_dx2, d2f1_dy2)),
         ((d2g0_dx2, d2g0_dy2), (d2g1_dx2, d2g1_dy2))) = self.del2bc
        d2A_dx2 = (1 - y)*d2g0_dx2([x, 0]) + y*d2g1_dx2([x, 1])
        d2A_dy2 = (1 - x)*d2f0_dy2([0, y]) + x*d2f1_dy2([1, y])
        _del2A = [d2A_dx2, d2A_dy2]
        return _del2A

    def P(self, xy):
        """Network coefficient function"""
        (x, y) = xy
        _P = x*(1 - x)*y*(1 - y)
        return _P

    def delP(self, xy):
        """Network coefficient function gradient"""
        (x, y) = xy
        dP_dx = (1 - 2*x)*y*(1 - y)
        dP_dy = x*(1 - x)*(1 - 2*y)
        _delP = [dP_dx, dP_dy]
        return _delP

    def del2P(self, xy):
        """Network coefficient function Laplacian"""
        (x, y) = xy
        d2P_dx2 = -2*y*(1 - y)
        d2P_dy2 = x*(1 - x)*-2
        _del2P = [d2P_dx2, d2P_dy2]
        return _del2P

    def Yt(self, xy, N):
        """Trial function"""
        A = self.A(xy)
        P = self.P(xy)
        _Yt = A + P*N
        # print("xy = %s" % xy)
        # print("A = %s" % A)
        # print("P = %s" % P)
        # print("N = %s" % N)
        # print("_Yt = %s" % _Yt)
        return _Yt

    def delYt(self, xy, N, delN):
        """Trial function gradient"""
        (x, y) = xy
        (dN_dx, dN_dy) = delN
        (dA_dx, dA_dy) = self.delA(xy)
        P = self.P(xy)
        (dP_dx, dP_dy) = self.delP(xy)
        dYt_dx = dA_dx + P*dN_dx + dP_dx*N
        dYt_dy = dA_dy + P*dN_dy + dP_dy*N
        _delYt = [dYt_dx, dYt_dy]
        return _delYt

    def del2Yt(self, xy, N, delN, del2N):
        """Trial function Laplacian"""
        (x, y) = xy
        (dN_dx, dN_dy) = delN
        (d2N_dx2, d2N_dy2) = del2N
        (d2A_dx2, d2A_dy2) = self.del2A(xy)
        P = self.P(xy)
        (dP_dx, dP_dy) = self.delP(xy)
        (d2P_dx2, d2P_dy2) = self.del2P(xy)
        d2Yt_dx2 = d2A_dx2 + P*d2N_dx2 + 2*dP_dx*dN_dx + d2P_dx2*N
        d2Yt_dy2 = d2A_dy2 + P*d2N_dy2 + 2*dP_dy*dN_dy + d2P_dy2*N
        _del2Yt = [d2Yt_dx2, d2Yt_dy2]
        return _del2Yt
