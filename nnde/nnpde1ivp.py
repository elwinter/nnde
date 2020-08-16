"""
NNPDE1IVP - Class to solve 1st-order partial differential equation initial
value problems using a neural network

This module provides the functionality to solve 1st-order partial differential
equation initial value problems using a neural network.

Example:
    Create an empty NNPDE1IVP object.
        net = NNPDE1IVP()
    Create an NNPDE1IVP object for a PDE1IVP object.
        net = NNPDE1IVP(pde1ivp_obj)
    Create an NNPDE1IVP object for a PDE1IVP object, with 20 hidden nodes.
        net = NNPDE1IVP(pde1ivp_obj, nhid=20)

Attributes:
    None

Methods:
    train
    run
    run_gradient

Todo:
    * Add function annotations.
    * Add variable annotations.
    * Combine error and gradient code into a single function for speed.
"""


__all__ = []
__version__ = '0.0'
__author__ = 'Eric Winter (ewinter@stsci.edu)'


from math import sqrt
import numpy as np
from scipy.optimize import minimize

from nnde.kdelta import kdelta
from nnde.pde1ivp import PDE1IVP
import nnde.sigma as sigma
from nnde.slffnn import SLFFNN
from nnde.trainingdata import create_training_grid


# Default values for method parameters
DEFAULT_DEBUG = False
DEFAULT_ETA = 0.01
DEFAULT_MAXEPOCHS = 1000
DEFAULT_NHID = 10
DEFAULT_TRAINALG = 'delta'
DEFAULT_UMAX = 1
DEFAULT_UMIN = -1
DEFAULT_VERBOSE = False
DEFAULT_VMAX = 1
DEFAULT_VMIN = -1
DEFAULT_WMAX = 1
DEFAULT_WMIN = -1
DEFAULT_OPTS = {
    'debug':     DEFAULT_DEBUG,
    'eta':       DEFAULT_ETA,
    'maxepochs': DEFAULT_MAXEPOCHS,
    'nhid':      DEFAULT_NHID,
    'umax':      DEFAULT_UMAX,
    'umin':      DEFAULT_UMIN,
    'verbose':   DEFAULT_VERBOSE,
    'vmax':      DEFAULT_VMAX,
    'vmin':      DEFAULT_VMIN,
    'wmax':      DEFAULT_WMAX,
    'wmin':      DEFAULT_WMIN
    }


# Vectorize sigma functions for speed.
# s_v = np.vectorize(sigma.s)
# s1_v = np.vectorize(sigma.s1)
# s2_v = np.vectorize(sigma.s2)


class NNPDE1IVP(SLFFNN):
    """Solve a 1st-order PDE IVP with a single-layer feedforward neural
    network."""

    # Public methods

    def __init__(self, eq, nhid=DEFAULT_NHID):
        super().__init__()

        # Save the differential equation object.
        self.eq = eq

        # Initialize all network parameters to 0.
        self.w = np.zeros((m, nhid))
        self.u = np.zeros(nhid)
        self.v = np.zeros(nhid)

        # Clear the result structure for minimize() calls.
        self.res = None

        # Initialize iteration counter.
        self.nit = 0

        # Pre-vectorize (_v suffix) functions for efficiency.
        self.G_v = np.vectorize(self.eq.G)
        self.dG_dY_v = np.vectorize(self.eq.dG_dY)
        self.dG_ddelY_v = [np.vectorize(f) for f in self.eq.dG_ddelY]
        self.Yt_v = np.vectorize(self._Yt)
        self.delYt_v = [np.vectorize(f) for f in self._delYt]

    def __str__(self):
        s = ''
        s += "%s\n" % self.eq.name
        s += "w = %s\n" % self.w
        s += "u = %s\n" % self.u
        s += "v = %s\n" % self.v
        return s.rstrip()

    def train(self, x, trainalg=DEFAULT_TRAINALG, opts=DEFAULT_OPTS):
        """Train the network to solve a 1st-order PDE IVP. """
        my_opts = dict(DEFAULT_OPTS)
        my_opts.update(opts)

        if trainalg == 'delta':
            self._train_delta(x, my_opts)
        elif trainalg in ('Nelder-Mead', 'Powell', 'CG', 'BFGS',
                          'Newton-CG'):
            self._train_minimize(x, trainalg, my_opts)
        else:
            print('ERROR: Invalid training algorithm (%s)!' % trainalg)
            exit(1)

    # def run(self, x):
    #     """Compute the trained solution."""
    #     w = self.w
    #     u = self.u
    #     v = self.v

    #     z = np.outer(x, w) + u
    #     s = s_v(z)
    #     N = s.dot(v)
    #     Yt = self.Yt_v(x, N)

    #     return Yt

    def run_debug(self, x):
        """Compute the trained solution (debug version)."""
        n = len(x)
        H = len(self.v)
        w = self.w
        u = self.u
        v = self.v

        z = np.zeros((n, H))
        for i in range(n):
            for k in range(H):
                z[i, k] = u[k]
                for j in range(m):
                    z[i, k] += w[j, k]*x[i, j]

        s = np.zeros((n, H))
        for i in range(n):
            for k in range(H):
                s[i, k] = sigma.s(z[i, k])

        N = np.zeros(n)
        for i in range(n):
            for k in range(H):
                N[i] += v[k]*s[i, k]

        Yt = np.zeros(n)
        for i in range(n):
            Yt[i] = self._Yt(x[i], N[i])

        return Yt

#     def run_gradient(self, x):
#         """Compute the trained derivative."""
#         w = self.w
#         u = self.u
#         v = self.v

#         z = np.outer(x, w) + u
#         s = s_v(z)
#         s1 = s1_v(s)
#         N = s.dot(v)
#         dN_dx = s1.dot(v*w)
#         dYt_dx = self.dYt_dx_v(x, N, dN_dx)

#         return dYt_dx

    def run_gradient_debug(self, x):
        """Compute the trained derivative (debug version)."""
        n = len(x)
        H = len(self.v)
        w = self.w
        u = self.u
        v = self.v

        z = np.zeros((n, H))
        for i in range(n):
            for k in range(H):
                z[i, k] = u[k]
                for j in range(m):
                    z[i, k] += w[j, k]*x[i, j]

        s = np.zeros((n, H))
        for i in range(n):
            for k in range(H):
                s[i, k] = sigma.s(z[i, k])

        s1 = np.zeros((n, H))
        for i in range(n):
            for k in range(H):
                s1[i, k] = sigma.s1(s[i, k])

        N = np.zeros(n)
        for i in range(n):
            for k in range(H):
                N[i] += s[i, k]*v[k]

        delN = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                for k in range(H):
                    delN[i, j] += v[k]*s1[i, k]*w[j, k]

        delYt = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                delYt[i, j] = self._delYt[j](self, x[i], N[i], delN[i])

        return delYt

    # Internal methods below this point

    def _A(self, x_):
        """Boundary condition function"""
        (x, y) = x_
        f0 = self.eq.bc[0][0]
        g0 = self.eq.bc[1][0]
        A = (1 - x)*f0([0, y]) + (1 - y)*(g0([x, 0]) - (1 - x)*g0([0, 0]))
        return A

    def _dA_dx(self, x_):
        """Partial of A() wrt x"""
        (x, y) = x_
        ((f0, f1), (g0, g1)) = self.eq.bc
        (((df0_dx, df0_dy), (df1_dx, df1_dy)),
         ((dg0_dx, dg0_dy), (dg1_dx, dg1_dy))) = self.eq.delbc
        dA_dx = (
            (1 - x)*df0_dx([0, y]) - f0([0, y])
            + (1 - y)*(dg0_dx([x, 0]) - (1 - x)*dg0_dx([0, 0]) + g0([0, 0]))
        )
        return dA_dx

    def _dA_dy(self, x_):
        """Partial of A() wrt y"""
        (x, y) = x_
        ((f0, f1), (g0, g1)) = self.eq.bc
        (((df0_dx, df0_dy), (df1_dx, df1_dy)),
         ((dg0_dx, dg0_dy), (dg1_dx, dg1_dy))) = self.eq.delbc
        dA_dy = (
            (1 - x)*df0_dy([0, y])
            + (1 - y)*(dg0_dy([x, 0]) - (1 - x)*dg0_dy([0, 0]))
            - (g0([x, 0]) - (1 - x)*g0([0, 0]))
        )
        return dA_dy

    def _P(self, x_):
        (x, y) = x_
        P = x*y
        return P

    def _dP_dx(self, x_):
        """Partial of P wrt x"""
        (x, y) = x_
        return y

    def _dP_dy(self, x_):
        """Partial of P wrt y"""
        (x, y) = x_
        return x

    _delP = [_dP_dx, _dP_dy]

    def _Yt(self, x, N):
        """Trial function"""
        Yt = self._A(x) + self._P(x)*N
        return Yt

    def _dYt_dx(self, x_, N, dN_dx_):
        """First derivative of trial function wrt x"""
        (dN_dx, dN_dy) = dN_dx_
        dYt_dx = self._dA_dx(x_) + self._P(x_)*dN_dx + self._dP_dx(x_)*N
        return dYt_dx

    def _dYt_dy(self, x_, N, dN_dx_):
        """First derivative of trial function wrt y"""
        (dN_dx, dN_dy) = dN_dx_
        dYt_dy = self._dA_dy(x_) + self._P(x_)*dN_dy + self._dP_dy(x_)*N
        return dYt_dy

    _delYt = [_dYt_dx, _dYt_dy]

    def _train_delta(self, x, opts=DEFAULT_OPTS):
        """Train the network using the delta method. """

        my_opts = dict(DEFAULT_OPTS)
        my_opts.update(opts)

        # Sanity-check arguments.
        assert len(x) > 0
        assert opts['maxepochs'] > 0
        assert opts['eta'] > 0
        assert opts['vmin'] < opts['vmax']
        assert opts['wmin'] < opts['wmax']
        assert opts['umin'] < opts['umax']

        # Determine the number of training points, independent variables, and
        # hidden nodes.
        n = len(x)  # Number of training points
        m = len(self.eq.bc)
        H = len(self.v)   # Number of hidden nodes

        # Change notation for convenience.
        debug = my_opts['debug']
        verbose = my_opts['verbose']
        eta = my_opts['eta']  # Learning rate
        maxepochs = my_opts['maxepochs']  # Number of training epochs
        wmin = my_opts['wmin']  # Network parameter limits
        wmax = my_opts['wmax']
        umin = my_opts['umin']
        umax = my_opts['umax']
        vmin = my_opts['vmin']
        vmax = my_opts['vmax']

        # Create the hidden node weights, biases, and output node weights.
        w = np.random.uniform(wmin, wmax, (m, H))
        u = np.random.uniform(umin, umax, H)
        v = np.random.uniform(vmin, vmax, H)

        # Initial parameter deltas are 0.
        dE_dw = np.zeros((m, H))
        dE_du = np.zeros(H)
        dE_dv = np.zeros(H)

        # Train the network.
        for epoch in range(maxepochs):
            if debug:
                print('Starting epoch %d.' % epoch)

            # Compute the new values of the network parameters.
            w -= eta*dE_dw
            u -= eta*dE_du
            v -= eta*dE_dv

            # Compute the input, the sigmoid function, and its derivatives,
            # for each hidden node and training point.
            # x is nx1, w, u are 1xH
            # z, s, s1, s2 are nxH
            z = np.dot(x, w) + u
            s = s_v(z)
            s1 = s1_v(s)
            s2 = s2_v(s)

            # Compute the network output and its derivatives, for each
            # training point.
            # s, v are Hx1
            # N is scalar
            N = s.dot(v)
            P = np.zeros(n)
            for i in range(n):
                P[i] = self._P(x[i])
            delN = np.dot(s1, (v*w).T)
            dN_dw = (v[np.newaxis, np.newaxis, :]
                     * s1[:, np.newaxis, :]*x[:, :, np.newaxis])
            dN_du = v*s1
            dN_dv = np.copy(s)
            d2N_dwdx = np.zeros((n, m, m, H))
            for i in range(n):
                for j in range(m):
                    for jj in range(m):
                        d2N_dwdx[i, j, jj] = v*(s1[i]*kdelta(j, jj) +
                                                s2[i]*w[jj]*x[i, j])

            d2N_dudx = (v[np.newaxis, np.newaxis, :]
                        * s2[:, np.newaxis, :]*w[np.newaxis, :, :])
            d2N_dvdx = s1[:, np.newaxis, :]*w[np.newaxis, :, :]

            # Compute the value of the trial solution, its coefficients,
            # and derivatives, for each training point.
            Yt = self.Yt_v(x, N)
            RESUME HERE
            delYt = self.dYt_dx_v(x, N, dN_dx)
#             # Temporary broadcast version of x.
#             x_b = np.broadcast_to(x, (H, n)).T
#             dYt_dw = x_b*dN_dw
#             dYt_du = x_b*dN_du
#             dYt_dv = x_b*dN_dv
#             d2Yt_dwdx = x_b*d2N_dwdx + dN_dw
#             d2Yt_dudx = x_b*d2N_dudx + dN_du
#             d2Yt_dvdx = x_b*d2N_dvdx + dN_dv

#             # Compute the value of the original differential equation for
#             # each training point, and its derivatives.
#             G = self.G_v(x, Yt, dYt_dx)
#             dG_dYt = self.dG_dY_v(x, Yt, dYt_dx)
#             dG_dYtdx = self.dG_ddYdx_v(x, Yt, dYt_dx)
#             # Temporary broadcast versions of dG_dyt and dG_dytdx.
#             dG_dYt_b = np.broadcast_to(dG_dYt, (H, n)).T
#             dG_dYtdx_b = np.broadcast_to(dG_dYtdx, (H, n)).T
#             dG_dw = dG_dYt_b*dYt_dw + dG_dYtdx_b*d2Yt_dwdx
#             dG_du = dG_dYt_b*dYt_du + dG_dYtdx_b*d2Yt_dudx
#             dG_dv = dG_dYt_b*dYt_dv + dG_dYtdx_b*d2Yt_dvdx

#             # Compute the error function for this epoch.
#             E = np.sum(G**2)

#             # Compute the partial derivatives of the error with respect to
#             # the network parameters.
#             # Temporary boradcast version of G.
#             G_b = np.broadcast_to(G, (H, n)).T
#             dE_dw = 2*np.sum(G_b*dG_dw, axis=0)
#             dE_du = 2*np.sum(G_b*dG_du, axis=0)
#             dE_dv = 2*np.sum(G_b*dG_dv, axis=0)

#             # Compute RMS error for this epoch.
#             rmse = sqrt(E/n)
#             if verbose:
#                 print(epoch, rmse)

#         # Save the optimized parameters.
#         self.w = w
#         self.u = u
#         self.v = v

    def _train_delta_debug(self, x, opts=DEFAULT_OPTS):
        """Train using the delta method (debug version). """

        my_opts = dict(DEFAULT_OPTS)
        my_opts.update(opts)

        # Sanity-check arguments.
        assert len(x) > 0
        assert opts['maxepochs'] > 0
        assert opts['eta'] > 0
        assert opts['vmin'] < opts['vmax']
        assert opts['wmin'] < opts['wmax']
        assert opts['umin'] < opts['umax']

        # Determine the number of training points, independent variables, and
        # hidden nodes.
        n = len(x)  # Number of training points
        m = len(self.eq.bc)
        H = len(self.v)   # Number of hidden nodes

        # Change notation for convenience.
        debug = my_opts['debug']
        verbose = my_opts['verbose']
        eta = my_opts['eta']  # Learning rate
        maxepochs = my_opts['maxepochs']  # Number of training epochs
        wmin = my_opts['wmin']  # Network parameter limits
        wmax = my_opts['wmax']
        umin = my_opts['umin']
        umax = my_opts['umax']
        vmin = my_opts['vmin']
        vmax = my_opts['vmax']

        # Create the hidden node weights, biases, and output node weights.
        w = np.random.uniform(wmin, wmax, (m, H))
        u = np.random.uniform(umin, umax, H)
        v = np.random.uniform(vmin, vmax, H)

        # Initial parameter deltas are 0.
        dE_dw = np.zeros((m, H))
        dE_du = np.zeros(H)
        dE_dv = np.zeros(H)

        # Train the network.
        for epoch in range(maxepochs):
            if debug:
                print('Starting epoch %d.' % epoch)

            # Compute the new values of the network parameters.
            for j in range(m):
                for k in range(H):
                    w[j, k] -= eta*dE_dw[j, k]

            for k in range(H):
                u[k] -= eta*dE_du[k]

            for k in range(H):
                v[k] -= eta*dE_dv[k]

            # Compute the input, the sigmoid function, and its derivatives,
            # for each hidden node and each training point.
            z = np.zeros((n, H))
            for i in range(n):
                for k in range(H):
                    z[i, k] = u[k]
                    for j in range(m):
                        z[i, k] += w[j, k]*x[i, j]

            s = np.zeros((n, H))
            for i in range(n):
                for k in range(H):
                    s[i, k] = sigma.s(z[i, k])

            s1 = np.zeros((n, H))
            for i in range(n):
                for k in range(H):
                    s1[i, k] = sigma.s1(s[i, k])

            s2 = np.zeros((n, H))
            for i in range(n):
                for k in range(H):
                    s2[i, k] = sigma.s2(s[i, k])

            # Compute the network output and its derivatives, for each
            # training point.
            N = np.zeros(n)
            for i in range(n):
                for k in range(H):
                    N[i] += v[k]*s[i, k]

            P = np.zeros(n)
            for i in range(n):
                P[i] = self._P(x[i])

            delP = np.zeros((n, m))
            for i in range(n):
                for j in range(m):
                    delP[i, j] = self._delP[j](self, x[i])

            delN = np.zeros((n, m))
            for i in range(n):
                for j in range(m):
                    for k in range(H):
                        delN[i, j] += v[k]*s1[i, k]*w[j, k]

            dN_dw = np.zeros((n, m, H))
            for i in range(n):
                for j in range(m):
                    for k in range(H):
                        dN_dw[i, j, k] = v[k]*s1[i, k]*x[i, j]

            dN_du = np.zeros((n, H))
            for i in range(n):
                for k in range(H):
                    dN_du[i, k] = v[k]*s1[i, k]

            dN_dv = np.zeros((n, H))
            for i in range(n):
                for k in range(H):
                    dN_dv[i, k] = s[i, k]

            d2N_dwdx = np.zeros((n, m, m, H))
            for i in range(n):
                for j in range(m):
                    for jj in range(m):
                        for k in range(H):
                            d2N_dwdx[i, j, jj, k] = (
                                v[k]*(s1[i, k]*kdelta(j, jj)
                                      + s2[i, k]*w[jj, k]*x[i, j])
                            )

            d2N_dudx = np.zeros((n, m, H))
            for i in range(n):
                for j in range(m):
                    for k in range(H):
                        d2N_dudx[i, j, k] = v[k]*s2[i, k]*w[j, k]

            d2N_dvdx = np.zeros((n, m, H))
            for i in range(n):
                for j in range(m):
                    for k in range(H):
                        d2N_dvdx[i, j, k] = s1[i, k]*w[j, k]

            # Compute the value of the trial solution and its derivatives,
            # for each training point.
            Yt = np.zeros(n)
            for i in range(n):
                Yt[i] = self._Yt(x[i], N[i])

            delYt = np.zeros((n, m))
            for i in range(n):
                for j in range(m):
                    delYt[i, j] = self._delYt[j](self, x[i], N[i], delN[i])

            dYt_dw = np.zeros((n, m, H))
            for i in range(n):
                for j in range(m):
                    for k in range(H):
                        dYt_dw[i, j, k] = P[i]*dN_dw[i, j, k]

            dYt_du = np.zeros((n, H))
            for i in range(n):
                for k in range(H):
                    dYt_du[i, k] = P[i]*dN_du[i, k]

            dYt_dv = np.zeros((n, H))
            for i in range(n):
                for k in range(H):
                    dYt_dv[i, k] = P[i]*dN_dv[i, k]

            d2Yt_dwdx = np.zeros((n, m, m, H))
            for i in range(n):
                for j in range(m):
                    for jj in range(m):
                        for k in range(H):
                            d2Yt_dwdx[i, j, jj, k] = (
                                P[i]*d2N_dwdx[i, j, jj, k]
                                + delP[i, jj]*dN_dw[i, j, k]
                            )

            d2Yt_dudx = np.zeros((n, m, H))
            for i in range(n):
                for j in range(m):
                    for k in range(H):
                        d2Yt_dudx[i, j, k] = (
                            P[i]*d2N_dudx[i, j, k]
                            + delP[i, j]*dN_du[i, k]
                        )

            d2Yt_dvdx = np.zeros((n, m, H))
            for i in range(n):
                for j in range(m):
                    for k in range(H):
                        d2Yt_dvdx[i, j, k] = (
                            P[i]*d2N_dvdx[i, j, k]
                            + delP[i, j]*dN_dv[i, k]
                        )

            # Compute the value of the original differential equation for
            # each training point, and its derivatives.
            G = np.zeros(n)
            for i in range(n):
                G[i] = self.eq.G(x[i], Yt[i], delYt[i])

            dG_dYt = np.zeros(n)
            for i in range(n):
                dG_dYt[i] = self.eq.dG_dY(x[i], Yt[i], delYt[i])

            dG_ddelYt = np.zeros((n, m))
            for i in range(n):
                for j in range(m):
                    dG_ddelYt[i, j] = (
                        self.eq.dG_ddelY[j](x[i], Yt[i], delYt[i])
                    )

            dG_dw = np.zeros((n, m, H))
            for i in range(n):
                for j in range(m):
                    for k in range(H):
                        dG_dw[i, j, k] = dG_dYt[i]*dYt_dw[i, j, k]
                        for jj in range(m):
                            dG_dw[i, j, k] += (
                                dG_ddelYt[i, jj]*d2Yt_dwdx[i, j, jj, k]
                            )

            dG_du = np.zeros((n, H))
            for i in range(n):
                for k in range(H):
                    dG_du[i, k] = dG_dYt[i]*dYt_du[i, k]
                    for j in range(m):
                        dG_du[i, k] += dG_ddelYt[i, j]*d2Yt_dudx[i, j, k]

            dG_dv = np.zeros((n, H))
            for i in range(n):
                for k in range(H):
                    dG_dv[i, k] = dG_dYt[i]*dYt_dv[i, k]
                    for j in range(m):
                        dG_dv[i, k] += dG_ddelYt[i, j]*d2Yt_dvdx[i, j, k]

            # Compute the error function for this epoch.
            E = 0
            for i in range(n):
                E += G[i]**2

            # Compute the partial derivatives of the error with respect to
            # the network parameters.
            dE_dw = np.zeros((m, H))
            for j in range(m):
                for k in range(H):
                    for i in range(n):
                        dE_dw[j, k] += 2*G[i]*dG_dw[i, j, k]

            dE_du = np.zeros(H)
            for k in range(H):
                for i in range(n):
                    dE_du[k] += 2*G[i]*dG_du[i, k]

            dE_dv = np.zeros(H)
            for k in range(H):
                for i in range(n):
                    dE_dv[k] += 2*G[i]*dG_dv[i, k]

            # Compute the RMS error for this epoch.
            rmse = sqrt(E/n)
            if verbose:
                print(epoch, rmse)

        # Save the optimized parameters.
        self.w = w
        self.u = u
        self.v = v

    def _train_minimize(self, x, trainalg, opts=DEFAULT_OPTS):
        """Train the network using the SciPy minimize() function. """

        my_opts = dict(DEFAULT_OPTS)
        my_opts.update(opts)

        # Sanity-check arguments.
        assert len(x) > 0
        assert opts['vmin'] < opts['vmax']
        assert opts['wmin'] < opts['wmax']
        assert opts['umin'] < opts['umax']

        # Create the hidden node weights, biases, and output node weights.
        m = len(self.eq.bc)
        H = len(self.v)

        # Create the hidden node weights, biases, and output node weights.
        w = np.random.uniform(my_opts['wmin'], my_opts['wmax'], (m, H))
        u = np.random.uniform(my_opts['umin'], my_opts['umax'], H)
        v = np.random.uniform(my_opts['vmin'], my_opts['vmax'], H)

        # Assemble the network parameters into a single 1-D vector for
        # use by the minimize() method.
        p = np.hstack((self.w.flatten(), self.u, self.v))

        # Add the status callback if requested.
        callback = None
        if my_opts['verbose']:
            callback = self._print_progress

        # Minimize the error function to get the new parameter values.
        if trainalg in ('Nelder-Mead', 'Powell', 'CG', 'BFGS'):
            jac = None
        elif trainalg in ('Newton-CG',):
            jac = self.__compute_error_gradient
        res = minimize(self._compute_error_debug, p, method=trainalg,
                       jac=jac, args=(x), callback=callback)
        self.res = res

        # Unpack the optimized network parameters.
        for j in range(m):
            self.w[j] = res.x[j*H:(j + 1)*H]
        self.u = res.x[m*H:(m + 1)*H]
        self.v = res.x[(m + 1)*H:(m + 2)*H]

#     def _compute_error(self, p, x):
#         """Compute the error function using the current parameter values."""

#         # Unpack the network parameters (hsplit() returns views, so no
#         # copies made).
#         (w, u, v) = np.hsplit(p, 3)

#         # Compute the forward pass through the network.
#         z = np.outer(x, w) + u
#         s = s_v(z)
#         s1 = s1_v(s)
#         N = s.dot(v)
#         dN_dx = s1.dot(v*w)
#         Yt = self.Yt_v(x, N)
#         dYt_dx = self.dYt_dx_v(x, N, dN_dx)
#         G = self.G_v(x, Yt, dYt_dx)
#         E = np.sum(G**2)

#         return E

    def _compute_error_debug(self, p, x):
        """Compute the error function using the current parameter values
           (debug version)."""

        # Determine the number of training points, independent variables, and
        # hidden nodes.
        n = len(x)  # Number of training points
        m = len(self.eq.bc)
        H = len(self.v)

        # Unpack the network parameters.
        w = np.zeros((m, H))
        for j in range(m):
            w[j] = p[j*H:(j + 1)*H]
        u = p[m*H:(m + 1)*H]
        v = p[(m + 1)*H:(m + 2)*H]

        # Compute the forward pass through the network.
        z = np.zeros((n, H))
        for i in range(n):
            for k in range(H):
                z[i, k] = u[k]
                for j in range(m):
                    z[i, k] += w[j, k]*x[i, j]

        s = np.zeros((n, H))
        for i in range(n):
            for k in range(H):
                s[i, k] = sigma.s(z[i, k])

        s1 = np.zeros((n, H))
        for i in range(n):
            for k in range(H):
                s1[i, k] = sigma.s1(s[i, k])

        N = np.zeros(n)
        for i in range(n):
            for k in range(H):
                N[i] += v[k]*s[i, k]

        dN_dx = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                for k in range(H):
                    dN_dx[i, j] += v[k]*s1[i, k]*w[j, k]

        Yt = np.zeros(n)
        for i in range(n):
            Yt[i] = self._Yt(x[i], N[i])

        delYt = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                delYt[i, j] = self._delYt[j](self, x[i], N[i], dN_dx[i])

        G = np.zeros(n)
        for i in range(n):
            G[i] = self.eq.G(x[i], Yt[i], delYt[i])

        E = 0
        for i in range(n):
            E += G[i]**2

        return E

#     def _compute_error_gradient(self, p, x):
#         """Compute the gradient of the error function wrt network
#         parameters."""

#         # Fetch the number of training points.
#         n = len(x)

#         # Unpack the network parameters (hsplit() returns views, so no
#         # copies made).
#         H = len(self.v)
#         (w, u, v) = np.hsplit(p, 3)

#         # Compute the forward pass through the network.
#         z = np.outer(x, w) + u
#         s = s_v(z)
#         s1 = s1_v(s)
#         s2 = s2_v(s)

#         # WARNING: Numpy and loop code below can give different results with
#         # Newton-CG after a few iterations. The differences are very slight,
#         # but they result in significantly different values for the weights
#         # and biases. To avoid this for now, loop code has been retained for
#         # some computations below.

#         # N = s.dot(v)
#         N = np.zeros(n)
#         for i in range(n):
#             for k in range(H):
#                 N[i] += s[i, k]*v[k]

#         # dN_dx = s1.dot(v*w)
#         dN_dx = np.zeros(n)
#         for i in range(n):
#             for k in range(H):
#                 dN_dx[i] += s1[i, k]*v[k]*w[k]

#         # dN_dw = s1*np.outer(x, v)
#         dN_dw = np.zeros((n, H))
#         for i in range(n):
#             for k in range(H):
#                 dN_dw[i, k] = s1[i, k]*x[i]*v[k]

#         dN_du = s1*v
#         dN_dv = s

#         # d2N_dwdx = v*(s1 + s2*np.outer(x, w))
#         d2N_dwdx = np.zeros((n, H))
#         for i in range(n):
#             for k in range(H):
#                 d2N_dwdx[i, k] = v[k]*(s1[i, k] + s2[i, k]*x[i]*w[k])

#         d2N_dudx = v*s2*w
#         d2N_dvdx = s1*w
#         Yt = self._Yt(x, N)
#         dYt_dx = self._dYt_dx(x, N, dN_dx)
#         dYt_dw = np.broadcast_to(x, (H, n)).T*dN_dw
#         dYt_du = np.broadcast_to(x, (H, n)).T*dN_du
#         dYt_dv = np.broadcast_to(x, (H, n)).T*dN_dv
#         d2Yt_dwdx = np.broadcast_to(x, (H, n)).T*d2N_dwdx + dN_dw
#         d2Yt_dudx = np.broadcast_to(x, (H, n)).T*d2N_dudx + dN_du
#         d2Yt_dvdx = np.broadcast_to(x, (H, n)).T*d2N_dvdx + dN_dv

#         G = self.G_v(x, Yt, dYt_dx)
#         dG_dYt = self.dG_dY_v(x, Yt, dYt_dx)
#         dG_dYtdx = self.dG_ddYdx_v(x, Yt, dYt_dx)
#         dG_dw = (np.broadcast_to(dG_dYt, (H, n)).T*dYt_dw
#                  + np.broadcast_to(dG_dYtdx, (H, n)).T*d2Yt_dwdx)
#         dG_du = (np.broadcast_to(dG_dYt, (H, n)).T*dYt_du
#                  + np.broadcast_to(dG_dYtdx, (H, n)).T*d2Yt_dudx)
#         dG_dv = (np.broadcast_to(dG_dYt, (H, n)).T*dYt_dv
#                  + np.broadcast_to(dG_dYtdx, (H, n)).T*d2Yt_dvdx)

#         dE_dw = 2*np.sum(np.broadcast_to(G, (H, n)).T*dG_dw, axis=0)
#         dE_du = 2*np.sum(np.broadcast_to(G, (H, n)).T*dG_du, axis=0)
#         dE_dv = 2*np.sum(np.broadcast_to(G, (H, n)).T*dG_dv, axis=0)

#         jac = np.hstack((dE_dw, dE_du, dE_dv))

#         return jac

#     def _compute_error_gradient_debug(self, p, x):
#         """Compute the gradient of the error function wrt network
#         parameters (debug version)."""

#         # Fetch the number of training points.
#         n = len(x)

#         # Unpack the network parameters (hsplit() returns views, so no
#         # copies made).
#         H = len(self.v)
#         (w, u, v) = np.hsplit(p, 3)

#         # Compute the forward pass through the network.
#         z = np.zeros((n, H))
#         for i in range(n):
#             for k in range(H):
#                 z[i, k] = x[i]*w[k] + u[k]

#         s = np.zeros((n, H))
#         for i in range(n):
#             for k in range(H):
#                 s[i, k] = sigma.s(z[i, k])

#         s1 = np.zeros((n, H))
#         for i in range(n):
#             for k in range(H):
#                 s1[i, k] = sigma.s1(s[i, k])

#         s2 = np.zeros((n, H))
#         for i in range(n):
#             for k in range(H):
#                 s2[i, k] = sigma.s2(s[i, k])

#         N = np.zeros(n)
#         for i in range(n):
#             for k in range(H):
#                 N[i] += v[k]*s[i, k]

#         dN_dx = np.zeros(n)
#         for i in range(n):
#             for k in range(H):
#                 dN_dx[i] += s1[i, k]*v[k]*w[k]

#         dN_dw = np.zeros((n, H))
#         for i in range(n):
#             for k in range(H):
#                 dN_dw[i, k] = s1[i, k]*x[i]*v[k]

#         dN_du = np.zeros((n, H))
#         for i in range(n):
#             for k in range(H):
#                 dN_du[i, k] = s1[i, k]*v[k]

#         dN_dv = np.zeros((n, H))
#         for i in range(n):
#             for k in range(H):
#                 dN_dv[i, k] = s[i, k]

#         d2N_dwdx = np.zeros((n, H))
#         for i in range(n):
#             for k in range(H):
#                 d2N_dwdx[i, k] = v[k]*(s1[i, k] + s2[i, k]*x[i]*w[k])

#         d2N_dudx = np.zeros((n, H))
#         for i in range(n):
#             for k in range(H):
#                 d2N_dudx[i, k] = v[k]*s2[i, k]*w[k]

#         d2N_dvdx = np.zeros((n, H))
#         for i in range(n):
#             for k in range(H):
#                 d2N_dvdx[i, k] = s1[i, k]*w[k]

#         Yt = np.zeros(n)
#         for i in range(n):
#             Yt[i] = self._Yt(x[i], N[i])

#         dYt_dx = np.zeros(n)
#         for i in range(n):
#             dYt_dx[i] = self._dYt_dx(x[i], N[i], dN_dx[i])

#         dYt_dw = np.zeros((n, H))
#         for i in range(n):
#             for k in range(H):
#                 dYt_dw[i, k] = x[i]*dN_dw[i, k]

#         dYt_du = np.zeros((n, H))
#         for i in range(n):
#             for k in range(H):
#                 dYt_du[i, k] = x[i]*dN_du[i, k]

#         dYt_dv = np.zeros((n, H))
#         for i in range(n):
#             for k in range(H):
#                 dYt_dv[i, k] = x[i]*dN_dv[i, k]

#         d2Yt_dwdx = np.zeros((n, H))
#         for i in range(n):
#             for k in range(H):
#                 d2Yt_dwdx[i, k] = x[i]*d2N_dwdx[i, k] + dN_dw[i, k]

#         d2Yt_dudx = np.zeros((n, H))
#         for i in range(n):
#             for k in range(H):
#                 d2Yt_dudx[i, k] = x[i]*d2N_dudx[i, k] + dN_du[i, k]

#         d2Yt_dvdx = np.zeros((n, H))
#         for i in range(n):
#             for k in range(H):
#                 d2Yt_dvdx[i, k] = x[i]*d2N_dvdx[i, k] + dN_dv[i, k]

#         G = np.zeros(n)
#         for i in range(n):
#             G[i] = self.eq.G(x[i], Yt[i], dYt_dx[i])

#         dG_dYt = np.zeros(n)
#         for i in range(n):
#             dG_dYt[i] = self.eq.dG_dY(x[i], Yt[i], dYt_dx[i])

#         dG_ddYtdx = np.zeros(n)
#         for i in range(n):
#             dG_ddYtdx[i] = self.eq.dG_ddYdx(x[i], Yt[i], dYt_dx[i])

#         dG_dw = np.zeros((n, H))
#         for i in range(n):
#             for k in range(H):
#                 dG_dw[i, k] = (dG_dYt[i]*dYt_dw[i, k]
#                                + dG_ddYtdx[i]*d2Yt_dwdx[i, k])

#         dG_du = np.zeros((n, H))
#         for i in range(n):
#             for k in range(H):
#                 dG_du[i, k] = (dG_dYt[i]*dYt_du[i, k]
#                                + dG_ddYtdx[i]*d2Yt_dudx[i, k])

#         dG_dv = np.zeros((n, H))
#         for i in range(n):
#             for k in range(H):
#                 dG_dv[i, k] = (dG_dYt[i]*dYt_dv[i, k]
#                                + dG_ddYtdx[i]*d2Yt_dvdx[i, k])

#         dE_dw = np.zeros(H)
#         for k in range(H):
#             for i in range(n):
#                 dE_dw[k] += 2*G[i]*dG_dw[i, k]

#         dE_du = np.zeros(H)
#         for k in range(H):
#             for i in range(n):
#                 dE_du[k] += 2*G[i]*dG_du[i, k]

#         dE_dv = np.zeros(H)
#         for k in range(H):
#             for i in range(n):
#                 dE_dv[k] += 2*G[i]*dG_dv[i, k]

#         jac = np.zeros(3*H)
#         for j in range(H):
#             jac[j] = dE_dw[j]
#         for j in range(H):
#             jac[H + j] = dE_du[j]
#         for j in range(H):
#             jac[2*H + j] = dE_dv[j]

#         return jac

    def _print_progress(self, xk):
        """Callback for minimize() to print progress message from
        optimizer"""
        print('nit =', self.nit)
        self.nit += 1
        # print('xk =', xk)


if __name__ == '__main__':
    pass

    # Create training data.
    nx = 5
    ny = 5
    xy_train = np.array(create_training_grid([nx, ny]))
    print('The training points are:\n', xy_train)
    N = nx*ny
    assert len(xy_train) == N
    print('A total of %d training points were created.' % N)

    # Options for training
    training_opts = {}
    training_opts['debug'] = True
    training_opts['verbose'] = True
    training_opts['eta'] = 0.01
    training_opts['maxepochs'] = 1000
    H = 5

    # Test each training algorithm on each equation.
    for eq in ('eq.example_pde1ivp_01',):
        print('Examining %s.' % eq)
        pde1ivp = PDE1IVP(eq)
        print(pde1ivp)

        # Determine the number of dimensions in the problem.
        m = len(pde1ivp.bc)
        print('Differential equation %s has %d dimensions.' % (eq, m))

        # (Optional) analytical solution and derivative
        if pde1ivp.Ya:
            Ya = np.zeros(N)
            for i in range(N):
                Ya[i] = pde1ivp.Ya(xy_train[i])
            print('The analytical solution at the training points is:')
            print(Ya)
        if pde1ivp.delYa:
            delYa = np.zeros((N, m))
            for i in range(N):
                for j in range(m):
                    delYa[i, j] = pde1ivp.delYa[j](xy_train[i])
            print('The analytical gradient at the training points is:')
            print(delYa)

        # Create and train the networks.
        # for trainalg in ('delta', 'Nelder-Mead', 'Powell', 'CG', 'BFGS',
        #                  'Newton-CG'):
        for trainalg in ('BFGS',):
            print('Training using %s algorithm.' % trainalg)
            net = NNPDE1IVP(pde1ivp, nhid=H)
            print(net)
            np.random.seed(0)  # Use same seed for reproducibility.
            try:
                net.train(xy_train, trainalg=trainalg, opts=training_opts)
            except (OverflowError, ValueError) as e:
                print('Error using %s algorithm on %s!' % (trainalg, eq))
                print(e)
                continue
            # print(net.res)
            print('The trained network is:')
            print(net)
            Yt = net.run_debug(xy_train)
            print('The trained solution is:')
            print('Yt =', Yt)
            delYt = net.run_gradient_debug(xy_train)
            print('The trained gradient is:')
            print('delYt =', delYt)

            # (Optional) Error in solution and derivative
            if pde1ivp.Ya:
                print('The error in the trained solution is:')
                print(Yt - Ya)
            if pde1ivp.delYa:
                print('The error in the trained gradient is:')
                print(delYt - delYa)
