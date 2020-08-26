# Standard run script for nnde jobs.

# This code assumes the presence of an analytical solution in the equation
# definition file.


import datetime
from math import sqrt
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from nnde.nnpde2diff import NNPDE2DIFF
from nnde.pde2diff import PDE2DIFF
from nnde.trainingdata import create_training_grid


# Run-specific parameters
EQUATION = "nnde.eq.diff1d_halfsine"
TRAINING_ALGORITHM = "BFGS"
H = 5
NX = 5
NY = 5
NZ = 5
NT = 5

# Maximum value for random number seed (+1, since half-range used).
SEED_MAX_PLUS_1 = 2**32


def main():

    # Save the start time.
    t_start = datetime.datetime.now()
    print("Starting at %s" % t_start)

    # Read the equation definition.
    eq = PDE2DIFF(EQUATION)

    # Print the run parameters.
    print("Equation: %s" % EQUATION)
    print("Training algorithm: %s" % TRAINING_ALGORITHM)
    print("Hidden node count: %d" % H)

    # Determine the dimensionality of the problem.
    m = len(eq.bc)
    print("Number of problem dimensions: %d" % m)
    if m == 2:
        print("NX = %d, NT = %d" % (NX, NT))
    elif m == 3:
        print("NX = %d, NY = %d, NT = %d" % (NX, NY, NT))
    elif m == 4:
        print("NX = %d, NY = %d, NZ = %d, NT = %d" % (NX, NY, NZ, NT))
    else:
        print("Invalid problem dimension %d!" % m, file=sys.stderr)
        sys.exit(1)

    # Create and save the training data.
    print("Creating training data.")
    if m == 2:
        x_train = np.array(create_training_grid([NX, NT]))
    elif m == 3:
        x_train = np.array(create_training_grid([NX, NY, NT]))
    elif m == 4:
        x_train = np.array(create_training_grid([NX, NY, NZ, NT]))
    n = len(x_train)
    np.savetxt("training_points.dat", x_train, fmt="%g")

    # Initialize the random number generator.
    seed = np.random.randint(0, SEED_MAX_PLUS_1)
    print("Random number generator seed: %s" % seed)
    np.random.seed(seed)

    # Options for training
    training_opts = {}
    training_opts["debug"] = True
    training_opts["nhid"] = H
    training_opts["verbose"] = True

    # Create and train the neural network.
    print("Creating network.")
    net = NNPDE2DIFF(eq, nhid=H)
    print("Training network.")
    t_trainstart = datetime.datetime.now()
    net.train(x_train, trainalg=TRAINING_ALGORITHM, opts=training_opts)
    t_trainend = datetime.datetime.now()
    t_trainelapsed = t_trainend - t_trainstart
    print("Elapsed training time (clock): %s" % t_trainelapsed)

    # Compute and save the trained results.
    print("Computing trained results.")
    Yt = net.run(x_train)
    delYt = net.run_gradient(x_train)
    del2Yt = net.run_laplacian(x_train)
    np.savetxt("Yt.dat", Yt)
    np.savetxt("delYt.dat", delYt)
    np.savetxt("del2Yt.dat", del2Yt)

    # If available, compute and save the analytical results and errors.

    # Analytical solution
    print("Computing analytical solution.")
    Ya = np.array(list(map(eq.Ya, x_train)))
    np.savetxt("Ya.dat", Ya)
    Yt_err = Yt - Ya
    np.savetxt("Yt_err.dat", Yt_err)

    # Analytical gradient
    print("Computing analytical gradient.")
    delYa = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            delYa[i, j] = eq.delYa[j](x_train[i])
    np.savetxt("delYa.dat", delYa)
    delYt_err = delYt - delYa
    np.savetxt("delYt_err.dat", delYt_err)

    # Analytical Laplacian
    print("Computing analytical Laplacian.")
    del2Ya = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            del2Ya[i, j] = eq.del2Ya[j](x_train[i])
    np.savetxt("del2Ya.dat", del2Ya)
    del2Yt_err = del2Yt - del2Ya
    np.savetxt("del2Yt_err.dat", del2Yt_err)

    # RMS errors in trained results
    Yt_rmserr = _rms(Yt_err)
    print("The RMS error of the trained solution is:", Yt_rmserr)
    delYt_rmserr = _rms(delYt_err)
    print("The RMS error of the trained gradient is:", delYt_rmserr)
    del2Yt_rmserr = _rms(del2Yt_err)
    print("The RMS error of the trained Laplacian is:", del2Yt_rmserr)

    # Save the network parameter history.
    np.savetxt("phist.dat", net.phist)

    # Reshape data for easier plotting.
    x = x_train[0::NX, 0]
    t = x_train[:NX, 1]
    Yerr = Yt_err.reshape(NX, NT).T
    delYerr = delYt_err.reshape(NX, NT, 2)
    del2Yerr = del2Yt_err.reshape(NX, NT, 2)

    # Create plots showing the error in the trained and derivatives.
    matplotlib.use("Agg")
    plt.figure(figsize=(12, 12))

    # Plot error in trained solution at each time.
    plt.subplot(3, 2, 1)
    for i in range(NT):
        plt.plot(x, Yerr[i], label="t=%s" % t[i])
    plt.xlabel("x")
    plt.ylabel("Absolute error in Yt")
    plt.grid()
    plt.legend()

    # Plot error in trained dY/dx at each time.
    plt.subplot(3, 2, 3)
    for i in range(NT):
        plt.plot(x, delYerr[i, :, 0], label="t=%s" % t[i])
    plt.xlabel("x")
    plt.ylabel("Absolute error in dYt/dx")
    plt.grid()
    plt.legend()

    # Plot error in trained dY/dt at each time.
    plt.subplot(3, 2, 4)
    for i in range(NT):
        plt.plot(x, delYerr[i, :, 1], label="t=%s" % t[i])
    plt.xlabel("x")
    plt.ylabel("Absolute error in dYt/dt")
    plt.grid()
    plt.legend()

    # Plot error in trained d2Y/dx2 at each time.
    plt.subplot(3, 2, 5)
    for i in range(NT):
        plt.plot(x, del2Yerr[i, :, 0], label="t=%s" % t[i])
    plt.xlabel("x")
    plt.ylabel("Absolute error in d2Yt/dx2")
    plt.grid()
    plt.legend()

    # Plot error in trained d2Y/dt2 at each time.
    plt.subplot(3, 2, 6)
    for i in range(NT):
        plt.plot(x, del2Yerr[i, :, 1], label="t=%s" % t[i])
    plt.xlabel("x")
    plt.ylabel("Absolute error in d2Yt/dt2")
    plt.grid()
    plt.legend()

    # Save the error plots.
    plt.savefig("errors.png")

    # Plot the errors as heat maps.
    (X, T) = np.meshgrid(x, t)
    plt.figure(figsize=(12, 12))

    # Plot error in trained solution.
    plt.subplot(3, 2, 1)
    plt.pcolor(X, T, Yerr, shading="auto")
    plt.title("Absolute error in trained solution")
    plt.grid()
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("t")
    plt.gca().set_aspect("equal")

    # Plot error in trained dY/dx.
    plt.subplot(3, 2, 3)
    plt.pcolor(X, T, delYerr[:, :, 0], shading="auto")
    plt.title("Absolute error in trained dY/dx")
    plt.grid()
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("t")
    plt.gca().set_aspect("equal")

    # Plot error in trained dY/dt.
    plt.subplot(3, 2, 4)
    plt.pcolor(X, T, delYerr[:, :, 1], shading="auto")
    plt.title("Absolute error in trained dY/dt")
    plt.grid()
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("t")
    plt.gca().set_aspect("equal")

    # Plot error in trained d2Y/dx2.
    plt.subplot(3, 2, 5)
    plt.pcolor(X, T, del2Yerr[:, :, 0], shading="auto")
    plt.title("Absolute error in trained d2Y/dx2")
    plt.grid()
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("t")
    plt.gca().set_aspect("equal")

    # Plot error in trained d2Y/dt2.
    plt.subplot(3, 2, 6)
    plt.pcolor(X, T, del2Yerr[:, :, 1], shading="auto")
    plt.title("Absolute error in trained d2Y/dt2")
    plt.grid()
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("t")
    plt.gca().set_aspect("equal")

    plt.savefig("error_heatmaps.png")

    # Make plots of the parameter history.
    nit = net.phist.shape[0]
    phist = net.phist.copy()
    whist = phist[:, 0:m*H].reshape(nit, m, H)
    uhist = phist[:, m*H:(m + 1)*H].reshape(nit, H)
    vhist = phist[:, (m + 1)*H:(m + 2)*H].reshape(nit, H)
    plt.figure(figsize=(12, 12))
    plt.subplot(4, 1, 1)
    for k in range(H):
        plt.plot(whist[:, 0, k])
    plt.xlabel("Iteration")
    plt.ylabel("w[0, k]")

    plt.subplot(4, 1, 2)
    for k in range(H):
        plt.plot(whist[:, 1, k])
    plt.xlabel("Iteration")
    plt.ylabel("w[1, k]")

    plt.subplot(4, 1, 3)
    for k in range(H):
        plt.plot(uhist[:, k])
    plt.xlabel("Iteration")
    plt.ylabel("u[k]")

    plt.subplot(4, 1, 4)
    for k in range(H):
        plt.plot(vhist[:, k])
    plt.xlabel("Iteration")
    plt.ylabel("v[k]")

    plt.savefig("phist.png")

    # Test the trained solution against the same number of random points
    # in the domain.
    x_test = np.array(np.random.uniform(size=(n, m)))
    Yt_test = net.run(x_test)
    delYt_test = net.run_gradient(x_test)
    del2Yt_test = net.run_laplacian(x_test)
    Ya_test = np.zeros(n)
    delYa_test = np.zeros((n, m))
    del2Ya_test = np.zeros((n, m))
    for i in range(n):
        Ya_test[i] = eq.Ya(x_test[i])
        for j in range(m):
            delYa_test[i, j] = eq.delYa[j](x_test[i])
            del2Ya_test[i, j] = eq.del2Ya[j](x_test[i])
    Yt_test_err = Yt_test - Ya_test
    delYt_test_err = delYt_test - delYa_test
    del2Yt_test_err = del2Yt_test - del2Ya_test
    print("For %d random points in the domain:" % n)
    print("The RMS solution error is: %g" % _rms(Yt_test_err))
    print("The RMS gradient error is: %g" % _rms(delYt_test_err))
    print("The RMS Laplacian error is: %g" % _rms(del2Yt_test_err))

    # Print end report.
    t_end = datetime.datetime.now()
    print("Ending at", t_end)
    t_elapsed = t_end - t_start
    print("Total elapsed time =", t_elapsed)


def _rms(x):
    """Compute the RMS value of array x."""
    return sqrt(np.sum(x**2)/x.size)


if __name__ == "__main__":
    main()
