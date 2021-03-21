"""Class to solve 1st-order ordinary differential equation
initial value problems using a neural network

This module provides the functionality to solve 1st-order ordinary
differential equation initial value problems using a neural network.

This class creates a single-layer feedforward network with a number
of hidden nodes H (default is 10). These nodes are represented by
attributes w, u, and v, which represent the hidden node weights and
biases, and the output node weights for the hidden nodes.

The hidden nodes use the sigmoid transfer function.

Note that this class makes heavy use of NumPy arrays.

Example:
    Create an empty NNODE1IVP object.
        net = NNODE1IVP()
    Create an NNODE1IVP object for a ODE1IVP object.
        net = NNODE1IVP(ode1ivp_obj)
    Create an NNODE1IVP object for a ODE1IVP object, with 20 hidden nodes.
        net = NNODE1IVP(ode1ivp_obj, nhid=20)

Attributes:
    None

Methods:
    __init__() - Constructor
    __str__() - Create string version
    train() - Train the network
    run() - Run the trained network
    run_debug() - Run the trained network (debug version)
    run_derivative() - Run the trained derivative network
    run_derivative_debug() - Run the trained derivative network (debug)

Todo:
    * Add function annotations.
    * Add variable annotations.
    * Combine error and gradient code into a single function for speed.
"""


from math import sqrt
import numpy as np
from scipy.optimize import minimize

from nnde.differentialequation.ode.ode1ivp import ODE1IVP
import nnde.math.sigma as sigma
from nnde.neuralnetwork.slffnn import SLFFNN
from nnde.math.trainingdata import create_training_grid


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
s_v = np.vectorize(sigma.s)
s1_v = np.vectorize(sigma.s1)
s2_v = np.vectorize(sigma.s2)


class NNODE1IVP(SLFFNN):
    """Solve a 1st-order ODE IVP with a single-layer feedforward neural
    network."""

    # Public methods

    def __init__(self, eq, nhid=DEFAULT_NHID):
        """Constructor for NNODE1IVP objects.
        eq is an ODE1IVP object describing the equation to solve.
        nhid is the number of hidden nodes to use in the network
        (default is DEFAULT_NHID)."""

        # Save the differential equation object.
        self.eq = eq

        # Initialize all network parameters to 0.
        self.w = np.zeros(nhid)
        self.u = np.zeros(nhid)
        self.v = np.zeros(nhid)

        # Clear the result structure for minimize() calls.
        self.res = None

        # Initialize iteration counter.
        self.nit = 0

        # Create the parameter history array.
        self.phist = np.hstack((self.w.flatten(), self.u, self.v))

        # Initialize flags.
        self._debug = False
        self._verbose = False

        # Pre-vectorize (_v suffix) functions for efficiency.
        self.G_v = np.vectorize(self.eq.G)
        self.dG_dY_v = np.vectorize(self.eq.dG_dY)
        self.dG_ddYdx_v = np.vectorize(self.eq.dG_ddYdx)
        self.Yt_v = np.vectorize(self._Yt)
        self.dYt_dx_v = np.vectorize(self._dYt_dx)

    def __str__(self):
        """Create the string version of the object"""
        s = ''
        s += "%s\n" % self.eq.name
        s += "w = %s\n" % self.w
        s += "u = %s\n" % self.u
        s += "v = %s\n" % self.v
        return s.rstrip()

    def train(self, x, trainalg=DEFAULT_TRAINALG, opts=DEFAULT_OPTS):
        """Train the network to solve a 1st-order ODE IVP.
        trainalg is the desired training algorithm.
        opts is a dictionary of training options."""
        my_opts = dict(DEFAULT_OPTS)
        my_opts.update(opts)

        if trainalg == 'delta':
            self._train_delta(x, my_opts)
        elif trainalg in ('Nelder-Mead', 'Powell', 'CG', 'BFGS', 'Newton-CG'):
            self._train_minimize(x, trainalg, my_opts)
        else:
            print('ERROR: Invalid training algorithm (%s)!' % trainalg)
            exit(1)

    def run(self, x):
        """Compute the trained solution for each value in the
        array x."""
        w = self.w
        u = self.u
        v = self.v

        z = np.outer(x, w) + u
        s = s_v(z)
        N = s.dot(v)
        Yt = self.Yt_v(x, N)

        return Yt

    def run_derivative(self, x):
        """Compute the trained 1st derivative of the solution
        for each value in the array x."""
        w = self.w
        u = self.u
        v = self.v

        z = np.outer(x, w) + u
        s = s_v(z)
        s1 = s1_v(s)
        N = s.dot(v)
        dN_dx = s1.dot(v*w)
        dYt_dx = self.dYt_dx_v(x, N, dN_dx)

        return dYt_dx

    # Internal methods below this point

    def _Yt(self, x, N):
        """Trial function"""
        return self.eq.ic + x*N

    def _dYt_dx(self, x, N, dN_dx):
        """First derivative of trial function"""
        return x*dN_dx + N

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

        # Determine the number of training points, and change notation for
        # convenience.
        n = len(x)  # Number of training points
        H = len(self.v)
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
        w = np.random.uniform(wmin, wmax, H)
        u = np.random.uniform(umin, umax, H)
        v = np.random.uniform(vmin, vmax, H)

        # Initial parameter deltas are 0.
        dE_dw = np.zeros(H)
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

            # Log the current parameter values.
            self.phist = (
                np.vstack((self.phist, np.hstack((w.flatten(), u, v))))
            )

            # Compute the input, the sigmoid function, and its derivatives, for
            # each hidden node and training point.
            # x is nx1, w, u are 1xH
            # z, s, s1, s2 are nxH
            z = np.outer(x, w) + u
            s = s_v(z)
            s1 = s1_v(s)
            s2 = s2_v(s)

            # Compute the network output and its derivatives, for each
            # training point.
            # s, v are Hx1
            # N is scalar
            N = s.dot(v)
            dN_dx = s1.dot(v*w)
            dN_dw = s1*np.outer(x, v)
            dN_du = s1*v
            dN_dv = np.copy(s)
            d2N_dwdx = v*(s1 + s2*np.outer(x, w))
            d2N_dudx = v*s2*w
            d2N_dvdx = s1*w

            # Compute the value of the trial solution, its coefficients,
            # and derivatives, for each training point.
            Yt = self.Yt_v(x, N)
            dYt_dx = self.dYt_dx_v(x, N, dN_dx)
            # Temporary broadcast version of x.
            x_b = np.broadcast_to(x, (H, n)).T
            dYt_dw = x_b*dN_dw
            dYt_du = x_b*dN_du
            dYt_dv = x_b*dN_dv
            d2Yt_dwdx = x_b*d2N_dwdx + dN_dw
            d2Yt_dudx = x_b*d2N_dudx + dN_du
            d2Yt_dvdx = x_b*d2N_dvdx + dN_dv

            # Compute the value of the original differential equation for
            # each training point, and its derivatives.
            G = self.G_v(x, Yt, dYt_dx)
            dG_dYt = self.dG_dY_v(x, Yt, dYt_dx)
            dG_dYtdx = self.dG_ddYdx_v(x, Yt, dYt_dx)
            # Temporary broadcast versions of dG_dyt and dG_dytdx.
            dG_dYt_b = np.broadcast_to(dG_dYt, (H, n)).T
            dG_dYtdx_b = np.broadcast_to(dG_dYtdx, (H, n)).T
            dG_dw = dG_dYt_b*dYt_dw + dG_dYtdx_b*d2Yt_dwdx
            dG_du = dG_dYt_b*dYt_du + dG_dYtdx_b*d2Yt_dudx
            dG_dv = dG_dYt_b*dYt_dv + dG_dYtdx_b*d2Yt_dvdx

            # Compute the error function for this epoch.
            E = np.sum(G**2)

            # Compute the partial derivatives of the error with respect to the
            # network parameters.
            # Temporary boradcast version of G.
            G_b = np.broadcast_to(G, (H, n)).T
            dE_dw = 2*np.sum(G_b*dG_dw, axis=0)
            dE_du = 2*np.sum(G_b*dG_du, axis=0)
            dE_dv = 2*np.sum(G_b*dG_dv, axis=0)

            # Compute RMS error for this epoch.
            rmse = sqrt(E/n)
            if debug:
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
        H = len(self.v)
        wmin = my_opts['wmin']  # Network parameter limits
        wmax = my_opts['wmax']
        umin = my_opts['umin']
        umax = my_opts['umax']
        vmin = my_opts['vmin']
        vmax = my_opts['vmax']

        # Create the hidden node weights, biases, and output node weights.
        w = np.random.uniform(wmin, wmax, H)
        u = np.random.uniform(umin, umax, H)
        v = np.random.uniform(vmin, vmax, H)

        # Assemble the network parameters into a single 1-D vector for
        # use by the minimize() method.
        p = np.hstack((w, u, v))

        # Add the status callback.
        callback = self._iteration_callback
        self._debug = my_opts["debug"]
        self._verbose = my_opts["verbose"]

        # Minimize the error function to get the new parameter values.
        if trainalg in ('Nelder-Mead', 'Powell', 'CG', 'BFGS'):
            jac = None
        elif trainalg in ('Newton-CG',):
            jac = self._compute_error_gradient
        res = minimize(self._compute_error, p, method=trainalg, jac=jac,
                       args=(x), callback=callback)
        self.res = res

        # Unpack the optimized network parameters.
        self.w = res.x[0:H]
        self.u = res.x[H:2*H]
        self.v = res.x[2*H:3*H]

    def _compute_error(self, p, x):
        """Compute the error function using the current parameter values."""

        # Unpack the network parameters (hsplit() returns views, so no copies
        # made).
        (w, u, v) = np.hsplit(p, 3)

        # Compute the forward pass through the network.
        z = np.outer(x, w) + u
        s = s_v(z)
        s1 = s1_v(s)
        N = s.dot(v)
        dN_dx = s1.dot(v*w)
        Yt = self.Yt_v(x, N)
        dYt_dx = self.dYt_dx_v(x, N, dN_dx)
        G = self.G_v(x, Yt, dYt_dx)
        E = np.sum(G**2)

        return E

    def _compute_error_gradient(self, p, x):
        """Compute the gradient of the error function wrt network
        parameters."""

        # Fetch the number of training points.
        n = len(x)

        # Unpack the network parameters (hsplit() returns views, so no copies
        # made).
        H = len(self.v)
        (w, u, v) = np.hsplit(p, 3)

        # Compute the forward pass through the network.
        z = np.outer(x, w) + u
        s = s_v(z)
        s1 = s1_v(s)
        s2 = s2_v(s)

        # WARNING: Numpy and loop code below can give different results with
        # Newton-CG after a few iterations. The differences are very slight,
        # but they result in significantly different values for the weights
        # and biases. To avoid this for now, loop code has been retained for
        # some computations below.

        # N = s.dot(v)
        N = np.zeros(n)
        for i in range(n):
            for k in range(H):
                N[i] += s[i, k]*v[k]

        # dN_dx = s1.dot(v*w)
        dN_dx = np.zeros(n)
        for i in range(n):
            for k in range(H):
                dN_dx[i] += s1[i, k]*v[k]*w[k]

        # dN_dw = s1*np.outer(x, v)
        dN_dw = np.zeros((n, H))
        for i in range(n):
            for k in range(H):
                dN_dw[i, k] = s1[i, k]*x[i]*v[k]

        dN_du = s1*v
        dN_dv = s

        # d2N_dwdx = v*(s1 + s2*np.outer(x, w))
        d2N_dwdx = np.zeros((n, H))
        for i in range(n):
            for k in range(H):
                d2N_dwdx[i, k] = v[k]*(s1[i, k] + s2[i, k]*x[i]*w[k])

        d2N_dudx = v*s2*w
        d2N_dvdx = s1*w
        Yt = self._Yt(x, N)
        dYt_dx = self._dYt_dx(x, N, dN_dx)
        dYt_dw = np.broadcast_to(x, (H, n)).T*dN_dw
        dYt_du = np.broadcast_to(x, (H, n)).T*dN_du
        dYt_dv = np.broadcast_to(x, (H, n)).T*dN_dv
        d2Yt_dwdx = np.broadcast_to(x, (H, n)).T*d2N_dwdx + dN_dw
        d2Yt_dudx = np.broadcast_to(x, (H, n)).T*d2N_dudx + dN_du
        d2Yt_dvdx = np.broadcast_to(x, (H, n)).T*d2N_dvdx + dN_dv

        G = self.G_v(x, Yt, dYt_dx)
        dG_dYt = self.dG_dY_v(x, Yt, dYt_dx)
        dG_dYtdx = self.dG_ddYdx_v(x, Yt, dYt_dx)
        dG_dw = (np.broadcast_to(dG_dYt, (H, n)).T*dYt_dw
                 + np.broadcast_to(dG_dYtdx, (H, n)).T*d2Yt_dwdx)
        dG_du = (np.broadcast_to(dG_dYt, (H, n)).T*dYt_du
                 + np.broadcast_to(dG_dYtdx, (H, n)).T*d2Yt_dudx)
        dG_dv = (np.broadcast_to(dG_dYt, (H, n)).T*dYt_dv
                 + np.broadcast_to(dG_dYtdx, (H, n)).T*d2Yt_dvdx)

        dE_dw = 2*np.sum(np.broadcast_to(G, (H, n)).T*dG_dw, axis=0)
        dE_du = 2*np.sum(np.broadcast_to(G, (H, n)).T*dG_du, axis=0)
        dE_dv = 2*np.sum(np.broadcast_to(G, (H, n)).T*dG_dv, axis=0)

        jac = np.hstack((dE_dw, dE_du, dE_dv))

        return jac

    def _iteration_callback(self, xk):
        """Callback after each optimizer iteration"""
        if self._debug:
            print("nit =", self.nit)
        self.nit += 1

        # Log the current parameters.
        self.phist = np.vstack((self.phist, xk))
