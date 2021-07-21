# Script to generate figure for NNDE JOSS paper.


import math as m

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


# Width and height of a figure (a standard 8.5x11 inch page)
FIGURE_WIDTH_INCHES = 8.5
FIGURE_HEIGHT_INCHES = 11

# <HACK>
NT = 11
NX = 11
NY = 11
N_CMAP_COLORS = 16
# </HACK>


# Set to True to use LaTex rendering.
use_latex = True

def make_plots(variable_name, titles, cb_labels, t_labels, e):
    """Make a triplet of analytical, trained, and error plots."""

    # Compute the number of subplots to make.
    nt = len(t_labels)

    # Compute the number of rows and columns of subplots.
    n_rows = int(m.ceil(m.sqrt(nt)))
    n_cols = int(m.floor(m.sqrt(nt)))

    # Compute the value limits for each data set.
    (e_min, e_max) = (e.min(), e.max())

    # Plot the absolute error in the trained results.
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols,
                             figsize=(FIGURE_WIDTH_INCHES,
                                      FIGURE_HEIGHT_INCHES))
    for i in range(nt):
        ax = axes.flat[i]
        im = ax.imshow(e[i].T, origin='lower', extent=[0, 1, 0, 1], vmin=-4e-4, vmax=4e-4,
                       cmap=plt.get_cmap('viridis', N_CMAP_COLORS))
        if i >= (n_rows - 1)*n_cols:
            ax.set_xlabel('$x$')
        else:
            ax.tick_params(labelbottom=False)
        if i % n_cols == 0:
            ax.set_ylabel('$y$')
        else:
            ax.tick_params(labelleft=False)
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.text(0.05, 0.9, t_labels[i], color='white')
    # Hide unused subplots.
    for i in range(nt, n_rows*n_cols):
        axes.flat[i].axis('off')

    # Add the colorbar.
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cb = fig.colorbar(im, cax=cbar_ax)
    fig.text(0.85, 0.86, cb_labels[0])
    plt.savefig(variable_name + '_e.png')
    plt.close()


def main():

    # Use LaTex for equation rendering.
    if use_latex:
        plt.rcParams.update({
            'text.usetex': True
        })

    # Load the training data.
    x_train = np.loadtxt('training_points.dat')

    # N.B. RESULT FILES USE COLUMN ORDER (x y t).

    # Load the errors in the trained results.
    Yt_err = np.loadtxt('Yt_err.dat').reshape((NT, NX, NY))

    # Create the labels for each time plot.
    t = x_train[::NX*NY, 0]
    t_labels = ['$t = %s$' % tt for tt in t]

    # Create a MathJax string for the equation.
    equation_string = r'$\frac {\partial \psi} {dt} - 0.1 \left( \frac {\partial^2 \psi} {\partial x^2} + \frac {\partial^2 \psi} {\partial y^2} \right) = 0$'

    # Create plots in a buffer for writing to a file.
    matplotlib.use('Agg')

    # Create plots for the solution error.
    titles = [
        'Absolute error in neural network solution of\n%s' % equation_string
    ]
    cb_labels = [r'$\psi_t - \psi_a$']
    make_plots('Y', titles, cb_labels, t_labels, Yt_err)


if __name__ == "__main__":
    main()
