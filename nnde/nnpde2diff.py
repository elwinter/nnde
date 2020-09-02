"""
NNPDE2DIFF - Class to solve 1-, 2-, and 3D diffusion problems using a
neural network

This module provides the functionality to solve 1-, 2-, and 3-D diffusion
problems using a neural network.

Example:
    Create an empty NNPDE2DIFF object.
        net = NNPDE2DIFF()
    Create an NNPDE2DIFF object for a PDE2DIFF1D object.
        net = NNPDE2DIFF(pde2diff1d_obj)
    Create an NNPDE2DIFF object for a PDE2DIFF2D object.
        net = NNPDE2DIFF(pde2diff2d_obj)
    Create an NNPDE2DIFF object for a PDE2DIFF3D object.
        net = NNPDE2DIFF(pde2diff3d_obj)
    Create an NNPDE2DIFF object for a PDE2DIFF2D object, with 20 hidden
              nodes.
        net = NNPDE2DIFF(pde2diff2d_obj, nhid=20)

Attributes:
    None

Methods:
    train()
    run()
    run_gradient()
    run_laplacian()

Todo:
    * Add function annotations.
    * Add variable annotations.
    * Combine error and gradient code into a single function for speed.
"""


__all__ = []
__version__ = "0.0"
__author__ = "Eric Winter (ewinter@stsci.edu)"


from math import sqrt
import sys

import numpy as np
from scipy.optimize import minimize

from nnde.diff1dtrialfunction import Diff1DTrialFunction
from nnde.diff2dtrialfunction import Diff2DTrialFunction
from nnde.diff3dtrialfunction import Diff3DTrialFunction
from nnde.kdelta import kdelta
from nnde.pde2diff import PDE2DIFF
import nnde.sigma as sigma
from nnde.slffnn import SLFFNN
from nnde.trainingdata import create_training_grid

# Default values for method parameters
DEFAULT_DEBUG = False
DEFAULT_ETA = 0.1
DEFAULT_MAXEPOCHS = 1000
DEFAULT_NHID = 10
DEFAULT_TRAINALG = "delta"
DEFAULT_UMAX = 1
DEFAULT_UMIN = -1
DEFAULT_USE_JACOBIAN = False
DEFAULT_VERBOSE = False
DEFAULT_VMAX = 1
DEFAULT_VMIN = -1
DEFAULT_WMAX = 1
DEFAULT_WMIN = -1
DEFAULT_OPTS = {
    "debug":        DEFAULT_DEBUG,
    "eta":          DEFAULT_ETA,
    "maxepochs":    DEFAULT_MAXEPOCHS,
    "nhid":         DEFAULT_NHID,
    "umax":         DEFAULT_UMAX,
    "umin":         DEFAULT_UMIN,
    "use_jacobian": DEFAULT_USE_JACOBIAN,
    "verbose":      DEFAULT_VERBOSE,
    "vmax":         DEFAULT_VMAX,
    "vmin":         DEFAULT_VMIN,
    "wmax":         DEFAULT_WMAX,
    "wmin":         DEFAULT_WMIN
    }


# Vectorize sigma functions for speed.
s_v = np.vectorize(sigma.s)
s1_v = np.vectorize(sigma.s1)
s2_v = np.vectorize(sigma.s2)
s3_v = np.vectorize(sigma.s3)


class NNPDE2DIFF(SLFFNN):
    """Solve a diffusion problem with a neural network"""

    def __init__(self, eq, nhid=DEFAULT_NHID):
        super().__init__()

        # Save the differential equation object.
        self.eq = eq

        # Determine the number of dimensions.
        m = len(eq.bc)

        # Initialize all network parameters to 0.
        self.w = np.zeros((m, nhid))
        self.u = np.zeros(nhid)
        self.v = np.zeros(nhid)

        # Select the appropriate trial function.
        if m == 2:
            self.tf = Diff1DTrialFunction(eq.bc, eq.delbc, eq.del2bc)
        elif m == 3:
            self.tf = Diff2DTrialFunction(eq.bc, eq.delbc, eq.del2bc)
        elif m == 4:
            self.tf = Diff3DTrialFunction(eq.bc, eq.delbc, eq.del2bc)
        else:
            print("Unexpected problem dimensionality: %s!", m)
            exit(1)

        # Clear the result structure for minimize() calls.
        self.res = None

        # Initialize iteration counter.
        self.nit = 0

        # If the supplied equation object has optimized versions of the
        # boundary condition function and derivatives, use them.
        if hasattr(self.eq, "A"):
            self.tf.A = self.eq.A
        if hasattr(self.eq, "delA"):
            self.tf.delA = self.eq.delA
        if hasattr(self.eq, "del2A"):
            self.tf.del2A = self.eq.del2A

        # Create the parameter history array.
        self.phist = np.hstack((self.w.flatten(), self.u, self.v))

    def train(self, x, trainalg=DEFAULT_TRAINALG, opts=DEFAULT_OPTS):
        """Train the network to solve a 2-D diffusion problem"""
        my_opts = dict(DEFAULT_OPTS)
        my_opts.update(opts)

        if trainalg == "delta":
            self._train_delta(x, opts=my_opts)
        elif trainalg in ("Nelder-Mead", "Powell", "CG", "BFGS"):
            self._train_minimize(x, trainalg, opts=my_opts)
        else:
            print("ERROR: Invalid training algorithm (%s)!" % trainalg)
            exit(1)

    def run(self, x):
        """Compute the trained solution."""

        # Get references to the network parameters for convenience.
        w = self.w
        u = self.u
        v = self.v

        # Compute the activation for each input point and hidden node.
        z = np.dot(x, w) + u

        # Compute the sigma function for each input point and hidden node.
        s = s_v(z)

        # Compute the network output for each input point.
        N = np.dot(s, v)

        # Compute the value of the trial function for each input point.
        n = len(x)
        Yt = np.zeros(n)
        for i in range(n):
            Yt[i] = self.tf.Yt(x[i], N[i])

        # Return the trial function values for each input point.
        return Yt

    def run_debug(self, x):
        """Compute the trained solution (debug version)."""
        n = len(x)
        m = len(self.eq.bc)
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
            Yt[i] = self.tf.Yt(x[i], N[i])

        return Yt

    def run_gradient(self, x):
        """Compute the trained derivative."""
        n = len(x)
        m = len(self.eq.bc)
        w = self.w
        u = self.u
        v = self.v

        z = x.dot(w) + u
        s = s_v(z)
        s1 = s1_v(s)
        N = s.dot(v)
        delN = s1.dot((w*v).T)
        delYt = np.zeros((n, m))
        for i in range(n):
            delYt[i] = self.tf.delYt(x[i], N[i], delN[i])

        return delYt

    def run_gradient_debug(self, x):
        """Compute the trained derivative (debug version)."""
        n = len(x)
        m = len(self.eq.bc)
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
            delYt[i] = self.tf.delYt(x[i], N[i], delN[i])

        return delYt

    def run_laplacian(self, x):
        """Compute the trained Laplacian."""

        # Fetch the number n of input points at which to calculate the
        # output, and the number m of components of each point.
        n = len(x)
        m = len(x[0])

        # Get references to the network parameters for convenience.
        w = self.w
        u = self.u
        v = self.v

        # Compute the net input, the sigmoid function and its
        # derivatives, for each hidden node and each training point.
        z = x.dot(w) + u
        s = s_v(z)
        s1 = s1_v(s)
        s2 = s2_v(s)

        # Compute the network output and its derivatives, for each
        # training point.
        N = s.dot(v)
        delN = s1.dot((w*v).T)
        del2N = s2.dot((w**2*v).T)

        # Compute the Laplacian components for the trial function.
        del2Yt = np.zeros((n, m))
        for i in range(n):
            del2Yt[i] = self.tf.del2Yt(x[i], N[i], delN[i], del2N[i])

        return del2Yt

    def run_laplacian_debug(self, x):
        """Compute the trained derivative (debug version)."""
        n = len(x)
        m = len(self.eq.bc)
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

        s2 = np.zeros((n, H))
        for i in range(n):
            for k in range(H):
                s2[i, k] = sigma.s2(s[i, k])

        N = np.zeros(n)
        for i in range(n):
            for k in range(H):
                N[i] += s[i, k]*v[k]

        delN = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                for k in range(H):
                    delN[i, j] += v[k]*s1[i, k]*w[j, k]

        del2N = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                for k in range(H):
                    del2N[i, j] += v[k]*s2[i, k]*w[j, k]**2

        del2Yt = np.zeros((n, m))
        for i in range(n):
            del2Yt[i] = self.tf.del2Yt(x[i], N[i], delN[i], del2N[i])

        return del2Yt

    # Internal methods below this point

    def __str__(self):
        """Print a human-readable version of the network."""
        s = ""
        s += "%s\n" % self.eq.name
        s += "w = %s\n" % self.w
        s += "u = %s\n" % self.u
        s += "v = %s\n" % self.v
        return s

    def _train_delta(self, x, opts=DEFAULT_OPTS):
        """Train using the delta method."""

        my_opts = dict(DEFAULT_OPTS)
        my_opts.update(opts)

        # Determine the number of training points, independent variables, and
        # hidden nodes.
        n = len(x)
        m = len(self.eq.bc)
        H = len(self.v)

        # Change notation for convenience.
        debug = my_opts["debug"]
        verbose = my_opts["verbose"]
        eta = my_opts["eta"]  # Learning rate
        maxepochs = my_opts["maxepochs"]  # Number of training epochs
        wmin = my_opts["wmin"]  # Network parameter limits
        wmax = my_opts["wmax"]
        umin = my_opts["umin"]
        umax = my_opts["umax"]
        vmin = my_opts["vmin"]
        vmax = my_opts["vmax"]

        # Create the hidden node weights, biases, and output node weights.
        w = np.random.uniform(wmin, wmax, (m, H))
        u = np.random.uniform(umin, umax, H)
        v = np.random.uniform(vmin, vmax, H)

        # Initial parameter deltas are 0.
        dE_dw = np.zeros((m, H))
        dE_du = np.zeros(H)
        dE_dv = np.zeros(H)

        # This small identity matrix is used during the computation of
        # some of the derivatives below.
        kd = np.identity(m)[np.newaxis, :, :, np.newaxis]

        # Train the network for the specified number of epochs.
        for epoch in range(maxepochs):
            if debug:
                print("Starting epoch %d." % epoch)

            # Compute the new values of the network parameters.
            w -= eta*dE_dw
            u -= eta*dE_du
            v -= eta*dE_dv

            # Log the current parameter values.
            self.phist = (
                np.vstack((self.phist, np.hstack((w.flatten(), u, v))))
            )

            # Compute the input, the sigmoid function, and its derivatives,
            # for each hidden node and each training point.
            z = x.dot(w) + u
            s = s_v(z)
            s1 = s1_v(s)
            s2 = s2_v(s)
            s3 = s3_v(s)

            # Compute the network output and its derivatives, for each
            # training point.
            N = s.dot(v)
            delN = s1.dot((w*v).T)
            del2N = s2.dot((w**2*v).T)
            dN_dw = v*s1[:, np.newaxis, :]*x[:, :, np.newaxis]
            dN_du = v*s1
            dN_dv = s
            d2N_dwdx = (v[np.newaxis, np.newaxis, np.newaxis, :]
                        * (s1[:, np.newaxis, np.newaxis, :]*kd
                           + s2[:, np.newaxis, np.newaxis, :]
                           * w[np.newaxis, np.newaxis, :, :]
                           * x[:, :, np.newaxis, np.newaxis]))
            d2N_dudx = v*s2[:, np.newaxis, :]*w[np.newaxis, :, :]
            d2N_dvdx = s1[:, np.newaxis, :]*w[np.newaxis, :, :]
            d3N_dwdx2 = (v[np.newaxis, np.newaxis, np.newaxis, :]
                         * (2*s2[:, np.newaxis, np.newaxis, :]
                            * w[np.newaxis, np.newaxis, :, :]*kd
                            + s3[:, np.newaxis, np.newaxis, :]
                            * w[np.newaxis, :, np.newaxis, :]**2
                            * x[:, :, np.newaxis, np.newaxis]))
            d3N_dudx2 = v*s3[:, np.newaxis, :]*w[np.newaxis, :, :]**2
            d3N_dvdx2 = s2[:, np.newaxis, :]*w[np.newaxis, :, :]**2

            # Compute the value of the trial solution, its coefficients,
            # and derivatives, for each training point.
            P = np.zeros(n)
            delP = np.zeros((n, m))
            del2P = np.zeros((n, m))
            Yt = np.zeros(n)
            delYt = np.zeros((n, m))
            del2Yt = np.zeros((n, m))
            for i in range(n):
                P[i] = self.tf.P(x[i])
                delP[i] = self.tf.delP(x[i])
                del2P[i] = self.tf.del2P(x[i])
                Yt[i] = self.tf.Yt(x[i], N[i])
                delYt[i] = self.tf.delYt(x[i], N[i], delN[i])
                del2Yt[i] = self.tf.del2Yt(x[i], N[i], delN[i], del2N[i])
            dYt_dw = P[:, np.newaxis, np.newaxis]*dN_dw
            dYt_du = P[:, np.newaxis]*dN_du
            dYt_dv = P[:, np.newaxis]*dN_dv
            d2Yt_dwdx = (P[:, np.newaxis, np.newaxis, np.newaxis]*d2N_dwdx
                         + delP[:, np.newaxis, :, np.newaxis]
                         * dN_dw[:, :, np.newaxis, :])
            d2Yt_dudx = (P[:, np.newaxis, np.newaxis]*d2N_dudx
                         + delP[:, :, np.newaxis]*dN_du[:, np.newaxis, :])
            d2Yt_dvdx = (P[:, np.newaxis, np.newaxis]*d2N_dvdx
                         + delP[:, :, np.newaxis]*dN_dv[:, np.newaxis, :])
            d3Yt_dwdx2 = (P[:, np.newaxis, np.newaxis, np.newaxis]*d3N_dwdx2
                          + 2*delP[:, np.newaxis, :, np.newaxis]*d2N_dwdx
                          + del2P[:, np.newaxis, :, np.newaxis]
                          * dN_dw[:, :, np.newaxis, :])
            d3Yt_dudx2 = (P[:, np.newaxis, np.newaxis]*d3N_dudx2
                          + 2*delP[:, :, np.newaxis]*d2N_dudx
                          + del2P[:, :, np.newaxis]*dN_du[:, np.newaxis, :])
            d3Yt_dvdx2 = (P[:, np.newaxis, np.newaxis]*d3N_dvdx2
                          + 2*delP[:, :, np.newaxis]*d2N_dvdx
                          + del2P[:, :, np.newaxis]*dN_dv[:, np.newaxis, :])

            # Compute the value of the original differential equation
            # for each training point, and its derivatives.
            G = np.zeros(n)
            dG_dYt = np.zeros(n)
            dG_ddelYt = np.zeros((n, m))
            dG_ddel2Yt = np.zeros((n, m))
            for i in range(n):
                G[i] = self.eq.G(x[i], Yt[i], delYt[i], del2Yt[i])
                dG_dYt[i] = self.eq.dG_dY(x[i], Yt[i], delYt[i], del2Yt[i])
                for j in range(m):
                    dG_ddelYt[i, j] = (
                        self.eq.dG_ddelY[j](x[i], Yt[i], delYt[i], del2Yt[i])
                    )
                    dG_ddel2Yt[i, j] = (
                        self.eq.dG_ddel2Y[j](x[i], Yt[i], delYt[i], del2Yt[i])
                    )

            dG_dw = dG_dYt[:, np.newaxis, np.newaxis]*dYt_dw
            for i in range(n):
                for j in range(m):
                    for k in range(H):
                        for jj in range(m):
                            dG_dw[i, j, k] += \
                                dG_ddelYt[i, jj]*d2Yt_dwdx[i, j, jj, k] + \
                                dG_ddel2Yt[i, jj]*d3Yt_dwdx2[i, j, jj, k]

            dG_du = dG_dYt[:, np.newaxis]*dYt_du
            for i in range(n):
                for k in range(H):
                    for j in range(m):
                        dG_du[i, k] += \
                            dG_ddelYt[i, j]*d2Yt_dudx[i, j, k] + \
                            dG_ddel2Yt[i, j]*d3Yt_dudx2[i, j, k]

            dG_dv = dG_dYt[:, np.newaxis]*dYt_dv
            for i in range(n):
                for k in range(H):
                    for j in range(m):
                        dG_dv[i, k] += \
                            dG_ddelYt[i, j]*d2Yt_dvdx[i, j, k] + \
                            dG_ddel2Yt[i, j]*d3Yt_dvdx2[i, j, k]

            # Compute the error function for this epoch.
            E2 = np.sum(G**2)
            if verbose:
                rmse = sqrt(E2/n)
                print(epoch, rmse)

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

        # Save the optimized parameters.
        self.w = w
        self.u = u
        self.v = v

    def _train_delta_debug(self, x, opts=DEFAULT_OPTS):
        """Train using the delta method (debug version)."""

        my_opts = dict(DEFAULT_OPTS)
        my_opts.update(opts)

        # Determine the number of training points, independent variables, and
        # hidden nodes.
        n = len(x)
        m = len(self.eq.bc)
        H = len(self.v)

        # Change notation for convenience.
        debug = my_opts["debug"]
        verbose = my_opts["verbose"]
        eta = my_opts["eta"]  # Learning rate
        maxepochs = my_opts["maxepochs"]  # Number of training epochs
        wmin = my_opts["wmin"]  # Network parameter limits
        wmax = my_opts["wmax"]
        umin = my_opts["umin"]
        umax = my_opts["umax"]
        vmin = my_opts["vmin"]
        vmax = my_opts["vmax"]

        # Create the hidden node weights, biases, and output node weights.
        w = np.random.uniform(wmin, wmax, (m, H))
        u = np.random.uniform(umin, umax, H)
        v = np.random.uniform(vmin, vmax, H)
        # w = np.zeros((m, H))
        # u = np.zeros(H)
        # v = np.zeros(H)

        # Initial parameter deltas are 0.
        dE_dw = np.zeros((m, H))
        dE_du = np.zeros(H)
        dE_dv = np.zeros(H)

        # Train the network.
        for epoch in range(maxepochs):
            if debug:
                print("Starting epoch %d." % epoch)

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

            s3 = np.zeros((n, H))
            for i in range(n):
                for k in range(H):
                    s3[i, k] = sigma.s3(s[i, k])

            # Compute the network output and its derivatives, for each
            # training point.
            N = np.zeros(n)
            for i in range(n):
                for k in range(H):
                    N[i] += v[k]*s[i, k]

            delN = np.zeros((n, m))
            for i in range(n):
                for j in range(m):
                    for k in range(H):
                        delN[i, j] += v[k]*s1[i, k]*w[j, k]

            del2N = np.zeros((n, m))
            for i in range(n):
                for j in range(m):
                    for k in range(H):
                        del2N[i, j] += v[k]*s2[i, k]*w[j, k]**2

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

            d3N_dwdx2 = np.zeros((n, m, m, H))
            for i in range(n):
                for j in range(m):
                    for jj in range(m):
                        for k in range(H):
                            d3N_dwdx2[i, j, jj, k] = (
                                v[k]*(2*s2[i, k]*w[jj, k]*kdelta(j, jj)
                                      + s3[i, k]*w[j, k]**2*x[i, j])
                            )

            d3N_dudx2 = np.zeros((n, m, H))
            for i in range(n):
                for j in range(m):
                    for k in range(H):
                        d3N_dudx2[i, j, k] = v[k]*s3[i, k]*w[j, k]**2

            d3N_dvdx2 = np.zeros((n, m, H))
            for i in range(n):
                for j in range(m):
                    for k in range(H):
                        d3N_dvdx2[i, j, k] = s2[i, k]*w[j, k]**2

            # Compute the value of the trial solution and its derivatives,
            # for each training point.
            P = np.zeros(n)
            for i in range(n):
                P[i] = self.tf.P(x[i])

            delP = np.zeros((n, m))
            for i in range(n):
                delP[i] = self.tf.delP(x[i])

            del2P = np.zeros((n, m))
            for i in range(n):
                del2P[i] = self.tf.del2P(x[i])

            Yt = np.zeros(n)
            for i in range(n):
                Yt[i] = self.tf.Yt(x[i], N[i])

            delYt = np.zeros((n, m))
            for i in range(n):
                delYt[i] = self.tf.delYt(x[i], N[i], delN[i])

            del2Yt = np.zeros((n, m))
            for i in range(n):
                del2Yt[i] = self.tf.del2Yt(x[i], N[i], delN[i], del2N[i])

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

            d3Yt_dwdx2 = np.zeros((n, m, m, H))
            for i in range(n):
                for j in range(m):
                    for jj in range(m):
                        for k in range(H):
                            d3Yt_dwdx2[i, j, jj, k] = (
                                P[i]*d3N_dwdx2[i, j, jj, k]
                                + 2*delP[i, jj]*d2N_dwdx[i, j, jj, k]
                                + del2P[i, jj]*dN_dw[i, j, k]
                            )

            d3Yt_dudx2 = np.zeros((n, m, H))
            for i in range(n):
                for j in range(m):
                    for k in range(H):
                        d3Yt_dudx2[i, j, k] = (
                            P[i]*d3N_dudx2[i, j, k]
                            + 2*delP[i, j]*d2N_dudx[i, j, k]
                            + del2P[i, j]*dN_du[i, k]
                        )

            d3Yt_dvdx2 = np.zeros((n, m, H))
            for i in range(n):
                for j in range(m):
                    for k in range(H):
                        d3Yt_dvdx2[i, j, k] = (
                            P[i]*d3N_dvdx2[i, j, k]
                            + 2*delP[i, j]*d2N_dvdx[i, j, k]
                            + del2P[i, j]*dN_dv[i, k]
                        )

            # Compute the value of the original differential equation
            # for each training point, and its derivatives.
            G = np.zeros(n)
            for i in range(n):
                G[i] = self.eq.G(x[i], Yt[i], delYt[i], del2Yt[i])

            dG_dYt = np.zeros(n)
            for i in range(n):
                dG_dYt[i] = self.eq.dG_dY(x[i], Yt[i], delYt[i], del2Yt[i])

            dG_ddelYt = np.zeros((n, m))
            for i in range(n):
                for j in range(m):
                    dG_ddelYt[i, j] = (
                        self.eq.dG_ddelY[j](x[i], Yt[i], delYt[i], del2Yt[i])
                    )

            dG_ddel2Yt = np.zeros((n, m))
            for i in range(n):
                for j in range(m):
                    dG_ddel2Yt[i, j] = (
                        self.eq.dG_ddel2Y[j](x[i], Yt[i], delYt[i], del2Yt[i])
                    )

            dG_dw = np.zeros((n, m, H))
            for i in range(n):
                for j in range(m):
                    for k in range(H):
                        dG_dw[i, j, k] = dG_dYt[i]*dYt_dw[i, j, k]
                        for jj in range(m):
                            dG_dw[i, j, k] += (
                                dG_ddelYt[i, jj]*d2Yt_dwdx[i, j, jj, k]
                                + dG_ddel2Yt[i, jj]*d3Yt_dwdx2[i, j, jj, k]
                            )

            dG_du = np.zeros((n, H))
            for i in range(n):
                for k in range(H):
                    dG_du[i, k] = dG_dYt[i]*dYt_du[i, k]
                    for j in range(m):
                        dG_du[i, k] += (
                            dG_ddelYt[i, j]*d2Yt_dudx[i, j, k]
                            + dG_ddel2Yt[i, j] * d3Yt_dudx2[i, j, k]
                        )

            dG_dv = np.zeros((n, H))
            for i in range(n):
                for k in range(H):
                    dG_dv[i, k] = dG_dYt[i]*dYt_dv[i, k]
                    for j in range(m):
                        dG_dv[i, k] += (
                            dG_ddelYt[i, j]*d2Yt_dvdx[i, j, k]
                            + dG_ddel2Yt[i, j] * d3Yt_dvdx2[i, j, k]
                        )

            # Compute the error function for this epoch.
            E2 = 0
            for i in range(n):
                E2 += G[i]**2
            if verbose:
                rmse = sqrt(E2/n)
                print(epoch, rmse)

            # Compute the partial derivatives of the error with respect to the
            # network parameters.
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
        assert opts["vmin"] < opts["vmax"]
        assert opts["wmin"] < opts["wmax"]
        assert opts["umin"] < opts["umax"]

        # Create the hidden node weights, biases, and output node weights.
        m = len(self.eq.bc)
        H = len(self.v)

        # Create the hidden node weights, biases, and output node weights.
        self.w = np.random.uniform(my_opts["wmin"], my_opts["wmax"], (m, H))
        self.u = np.random.uniform(my_opts["umin"], my_opts["umax"], H)
        self.v = np.random.uniform(my_opts["vmin"], my_opts["vmax"], H)
        # self.w = np.zeros((m, H))
        # self.u = np.zeros(H)
        # self.v = np.zeros(H)

        # Assemble the network parameters into a single 1-D vector for
        # use by the minimize() method.
        p = np.hstack((self.w.flatten(), self.u, self.v))

        # Set up the iteration callback.
        callback = self._iteration_callback
        self._verbose = my_opts["verbose"]

        # Use the error jacobian if requested.
        jac = None
        if my_opts["use_jacobian"]:
            jac = self._compute_error_jacobian_debug

        # Minimize the error function to get the new parameter values.
        res = minimize(self._compute_error_debug, p, method=trainalg,
                       jac=jac, args=(x), callback=callback)
        self.res = res

        # Unpack the optimized network parameters.
        for j in range(m):
            self.w[j] = res.x[j*H:(j + 1)*H]
        self.u = res.x[m*H:(m + 1)*H]
        self.v = res.x[(m + 1)*H:(m + 2)*H]

    def _compute_error(self, p, x):
        """Compute the current error in the trained solution."""

        # Unpack the network parameters.
        n = len(x)
        m = len(x[0])
        H = int(len(p)/(m + 2))
        w = np.zeros((m, H))
        for j in range(m):
            w[j] = p[j*H:(j + 1)*H]
        u = p[(m - 1)*H:m*H]
        v = p[m*H:(m + 1)*H]

        # Weighted inputs and transfer functions and derivatives.
        z = x.dot(w) + u
        s = s_v(z)
        s1 = s1_v(s)
        s2 = s2_v(s)

        # Network output and derivatives.
        N = s.dot(v)
        delN = s1.dot((w*v).T)
        del2N = s2.dot((w**2*v).T)

        # Trial function and derivatives
        Yt = np.zeros(n)
        delYt = np.zeros((n, m))
        del2Yt = np.zeros((n, m))
        for i in range(n):
            Yt[i] = self.tf.Yt(x[i], N[i])
            delYt[i] = self.tf.delYt(x[i], N[i], delN[i])
            del2Yt[i] = self.tf.del2Yt(x[i], N[i], delN[i], del2N[i])

        # Differential equation
        G = np.zeros(n)
        for i in range(n):
            G[i] = self.eq.G(x[i], Yt[i], delYt[i], del2Yt[i])

        E2 = np.sum(G**2)

        return E2

    def _compute_error_debug(self, p, x):
        """Compute the error function using the current parameter values
           (debug version)."""

        # Determine the number of training points, independent variables, and
        # hidden nodes.
        n = len(x)
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

        s2 = np.zeros((n, H))
        for i in range(n):
            for k in range(H):
                s2[i, k] = sigma.s2(s[i, k])

        N = np.zeros(n)
        for i in range(n):
            for k in range(H):
                N[i] += v[k]*s[i, k]

        delN = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                for k in range(H):
                    delN[i, j] += v[k]*s1[i, k]*w[j, k]

        del2N = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                for k in range(H):
                    del2N[i, j] += v[k]*s2[i, k]*w[j, k]**2

        Yt = np.zeros(n)
        for i in range(n):
            Yt[i] = self.tf.Yt(x[i], N[i])

        delYt = np.zeros((n, m))
        for i in range(n):
            delYt[i] = self.tf.delYt(x[i], N[i], delN[i])

        del2Yt = np.zeros((n, m))
        for i in range(n):
            del2Yt[i] = self.tf.del2Yt(x[i], N[i], delN[i], del2N[i])

        G = np.zeros(n)
        for i in range(n):
            G[i] = self.eq.G(x[i], Yt[i], delYt[i], del2Yt[i])

        E2 = 0
        for i in range(n):
            E2 += G[i]**2

        return E2

    # def _compute_error_jacobian(self, p, x):
    #     """Compute the Jacobian of the error function."""
    #     return None

    def _compute_error_jacobian_debug(self, p, x):
        """Compute the Jacobian of the error function (debug version)."""

        # Determine the number of training points, independent variables, and
        # hidden nodes.
        n = len(x)
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

        s2 = np.zeros((n, H))
        for i in range(n):
            for k in range(H):
                s2[i, k] = sigma.s2(s[i, k])

        s3 = np.zeros((n, H))
        for i in range(n):
            for k in range(H):
                s3[i, k] = sigma.s3(s[i, k])

        N = np.zeros(n)
        for i in range(n):
            for k in range(H):
                N[i] += v[k]*s[i, k]

        delN = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                for k in range(H):
                    delN[i, j] += v[k]*s1[i, k]*w[j, k]

        del2N = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                for k in range(H):
                    del2N[i, j] += v[k]*s2[i, k]*w[j, k]**2

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

        d3N_dwdx2 = np.zeros((n, m, m, H))
        for i in range(n):
            for j in range(m):
                for jj in range(m):
                    for k in range(H):
                        d3N_dwdx2[i, j, jj, k] = (
                            v[k]*(2*s2[i, k]*w[jj, k]*kdelta(j, jj)
                                  + s3[i, k]*w[j, k]**2*x[i, j])
                        )

        d3N_dudx2 = np.zeros((n, m, H))
        for i in range(n):
            for j in range(m):
                for k in range(H):
                    d3N_dudx2[i, j, k] = v[k]*s3[i, k]*w[j, k]**2

        d3N_dvdx2 = np.zeros((n, m, H))
        for i in range(n):
            for j in range(m):
                for k in range(H):
                    d3N_dvdx2[i, j, k] = s2[i, k]*w[j, k]**2

        P = np.zeros(n)
        for i in range(n):
            P[i] = self.tf.P(x[i])

        delP = np.zeros((n, m))
        for i in range(n):
            delP[i] = self.tf.delP(x[i])

        del2P = np.zeros((n, m))
        for i in range(n):
            del2P[i] = self.tf.del2P(x[i])

        Yt = np.zeros(n)
        for i in range(n):
            Yt[i] = self.tf.Yt(x[i], N[i])

        delYt = np.zeros((n, m))
        for i in range(n):
            delYt[i] = self.tf.delYt(x[i], N[i], delN[i])

        del2Yt = np.zeros((n, m))
        for i in range(n):
            del2Yt[i] = self.tf.del2Yt(x[i], N[i], delN[i], del2N[i])

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

        d3Yt_dwdx2 = np.zeros((n, m, m, H))
        for i in range(n):
            for j in range(m):
                for jj in range(m):
                    for k in range(H):
                        d3Yt_dwdx2[i, j, jj, k] = (
                            P[i]*d3N_dwdx2[i, j, jj, k]
                            + 2*delP[i, jj]*d2N_dwdx[i, j, jj, k]
                            + del2P[i, jj]*dN_dw[i, j, k]
                        )

        d3Yt_dudx2 = np.zeros((n, m, H))
        for i in range(n):
            for j in range(m):
                for k in range(H):
                    d3Yt_dudx2[i, j, k] = (
                        P[i]*d3N_dudx2[i, j, k]
                        + 2*delP[i, j]*d2N_dudx[i, j, k]
                        + del2P[i, j]*dN_du[i, k]
                    )

        d3Yt_dvdx2 = np.zeros((n, m, H))
        for i in range(n):
            for j in range(m):
                for k in range(H):
                    d3Yt_dvdx2[i, j, k] = (
                        P[i]*d3N_dvdx2[i, j, k]
                        + 2*delP[i, j]*d2N_dvdx[i, j, k]
                        + del2P[i, j]*dN_dv[i, k]
                    )

        G = np.zeros(n)
        for i in range(n):
            G[i] = self.eq.G(x[i], Yt[i], delYt[i], del2Yt[i])

        dG_dYt = np.zeros(n)
        for i in range(n):
            dG_dYt[i] = self.eq.dG_dY(x[i], Yt[i], delYt[i], del2Yt[i])

        dG_ddelYt = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                dG_ddelYt[i, j] = (
                    self.eq.dG_ddelY[j](x[i], Yt[i], delYt[i], del2Yt[i])
                )

        dG_ddel2Yt = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                dG_ddel2Yt[i, j] = (
                    self.eq.dG_ddel2Y[j](x[i], Yt[i], delYt[i], del2Yt[i])
                )

        dG_dw = np.zeros((n, m, H))
        for i in range(n):
            for j in range(m):
                for k in range(H):
                    dG_dw[i, j, k] = dG_dYt[i]*dYt_dw[i, j, k]
                    for jj in range(m):
                        dG_dw[i, j, k] += (
                            dG_ddelYt[i, jj]*d2Yt_dwdx[i, j, jj, k]
                            + dG_ddel2Yt[i, jj]*d3Yt_dwdx2[i, j, jj, k]
                        )

        dG_du = np.zeros((n, H))
        for i in range(n):
            for k in range(H):
                dG_du[i, k] = dG_dYt[i]*dYt_du[i, k]
                for j in range(m):
                    dG_du[i, k] += (
                        dG_ddelYt[i, j]*d2Yt_dudx[i, j, k]
                        + dG_ddel2Yt[i, j] * d3Yt_dudx2[i, j, k]
                    )

        dG_dv = np.zeros((n, H))
        for i in range(n):
            for k in range(H):
                dG_dv[i, k] = dG_dYt[i]*dYt_dv[i, k]
                for j in range(m):
                    dG_dv[i, k] += (
                        dG_ddelYt[i, j]*d2Yt_dvdx[i, j, k]
                        + dG_ddel2Yt[i, j] * d3Yt_dvdx2[i, j, k]
                    )

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

        jac = np.hstack((dE_dw.flatten(), dE_du, dE_dv))

        return jac

    def _iteration_callback(self, xk):
        """Callback after each optimizer iteration"""
        if self._verbose:
            print("nit =", self.nit)
        self.nit += 1

        # Log the current parameters.
        self.phist = np.vstack((self.phist, xk))


if __name__ == "__main__":

    # Create training data.
    nx = 5
    ny = 5
    nz = 5
    nt = 5
    xt_train = np.array(create_training_grid([nx, nt]))
    xyt_train = np.array(create_training_grid([nx, ny, nt]))
    xyzt_train = np.array(create_training_grid([nx, ny, nz, nt]))

    # Options for training
    training_opts = {}
    training_opts["debug"] = False
    training_opts["verbose"] = True
    training_opts["eta"] = 0.1
    training_opts["maxepochs"] = 1000
    training_opts["use_jacobian"] = False
    H = 5

    # Test each training algorithm on each equation.
    for pde in ("eq.diff1d_halfsine",):
        print("Examining %s." % pde)

        # Read the equation definition.
        eq = PDE2DIFF(pde)

        # Fetch the dimensionality of the problem.
        m = len(eq.bc)
        print("Differential equation %s has %d dimensions." % (eq, m))

        # Select the appropriate training set.
        if m == 2:
            x_train = xt_train
        elif m == 3:
            x_train = xyt_train
        elif m == 4:
            x_train = xyzt_train
        else:
            print("INVALID PROBLEM DIMENSION: %s" % m)
            sys.exit(1)
        n = len(x_train)

        # Analytical solution (if available)
        Ya = None
        if eq.Ya is not None:
            print("Computing analytical solution at training points.")
            Ya = np.zeros(n)
            for i in range(n):
                Ya[i] = eq.Ya(x_train[i])
            print("Ya =", Ya)

        # Analytical gradient (if available)
        delYa = None
        if eq.delYa is not None:
            print("Computing analytical gradient at training points.")
            delYa = np.zeros((n, m))
            for i in range(n):
                for j in range(m):
                    delYa[i][j] = eq.delYa[j](x_train[i])
            print("delYa =", delYa)

        # Analytical Laplacian (if available)
        del2Ya = None
        if eq.del2Ya is not None:
            print("Computing analytical Laplacian at training points.")
            del2Ya = np.zeros((n, m))
            for i in range(n):
                for j in range(m):
                    del2Ya[i][j] = eq.del2Ya[j](x_train[i])
            print("del2Ya =", del2Ya)

        # for trainalg in ("delta", "Nelder-Mead", "Powell", "CG", "BFGS",
        #                  "Newton-CG"):
        for trainalg in ("BFGS",):
            print("Training using %s algorithm." % trainalg)

            # Create and train the neural network.
            net = NNPDE2DIFF(eq, nhid=H)
            np.random.seed(0)
            try:
                net.train(x_train, trainalg=trainalg, opts=training_opts)
            except (OverflowError, ValueError) as e:
                print("Error using %s algorithm on %s!" % (trainalg, pde))
                print(e)
                print()
                continue

            if net.res:
                print(net.res)
            print("The trained network is:")
            print(net)

            # Run the network to get the trained solution.
            Yt = net.run(x_train)
            print("The trained solution is:")
            print("Yt =", Yt)

            Yt_rmserr = None
            if eq.Ya:
                Yt_err = Yt - Ya
                print("The error in the trained solution is:")
                print("Yt_err =", Yt_err)
                Yt_rmserr = sqrt(np.sum(Yt_err**2)/n)

            delYt = net.run_gradient(x_train)
            print("The trained gradient is:")
            print("delYt =", delYt)

            delYt_rmserr = None
            if eq.delYa:
                delYt_err = delYt - delYa
                print("The error in the trained gradient is:")
                print("delYt_err =", delYt_err)
                delYt_rmserr = sqrt(np.sum(delYt_err**2)/(n*m))

            del2Yt = net.run_laplacian(x_train)
            print("The trained Laplacian is:")
            print("del2Yt =", del2Yt)

            del2Yt_rmserr = None
            if eq.del2Ya:
                del2Yt_err = del2Yt - del2Ya
                print("The error in the trained Laplacian is:")
                print("del2Yt_err =", del2Yt_err)
                del2Yt_rmserr = sqrt(np.sum(del2Yt_err**2)/(n*m))

            if eq.Ya:
                print("RMS error for trained solution is %s." % Yt_rmserr)
            if eq.delYa:
                print("RMS error for trained gradient is %s." % delYt_rmserr)
            if eq.del2Ya:
                print("RMS error for trained Laplacian is %s." % del2Yt_rmserr)

            # Summary report
            if Yt_rmserr and delYt_rmserr and del2Yt_rmserr:
                print(nx, nt, H, Yt_rmserr, delYt_rmserr, del2Yt_rmserr)
