"""
trainingdata - Python module of functions useful for creating training data.

This module provides functions which automate the process of creating neural
network training data.

Examples:

    Create a set of 10 evenly-spaced training points in 1 dimension.
    x_train = create_training_grid(10)

    Create a 3x3x3 grid of training data, evenly spaced in the domain
    [[0, 1], [0, 1], [0, 1]].
        xyz_train = create_training_grid([3, 3, 3])

To do:
    * Add function annotations.
    * Add variable annotations.
"""


__all__ = []
__version__ = '0.0'
__author__ = 'Eric Winter (ewinter@stsci.edu)'


from itertools import repeat


def create_training_grid(n):
    """Create a grid of training data.  The input n is an integer, or a list
    containing the numbers of evenly-spaced data points to use in each
    dimension.  For example, for an (x, y, z) grid, with n = [3, 4, 5], we
    will get a grid with 3 points along the x-axis, 4 points along the
    y-axis, and 5 points along the z-axis, for a total of 3*4*5 = 60 points.
    The points along each dimension are evenly spaced in the range [0, 1].
    When there is m = 1 dimension, a list is returned, containing the evenly-
    spaced points in the single dimension.  When m > 1, a list of lists is
    returned, where each sub-list is the coordinates of a single point,
    in the order [x1, x2, ..., xm], where the coordinate order corresponds
    to the order of coordinate counts in the input list n. """

    # Determine the number of dimensions in the result.
    if isinstance(n, list):
        m = len(n)
    else:
        m = 1

    # Handle 1-D and (n>1)-D cases differently.
    if m == 1:
        X = [i/(n - 1) for i in range(n)]
    else:
        # Compute the evenly-spaced points along each dimension.
        x = [[i/(nn - 1) for i in range(nn)] for nn in n]

        # Assemble all possible point combinations.
        X = []
        p1 = None
        p2 = 1
        for j in range(m - 1):
            p1 = prod(n[j + 1:])
            XX = [xx for item in x[j] for xx in repeat(item, p1)]*p2
            X.append(XX)
            p2 *= n[j]
        X.append(x[-1]*p2)
        X = list(zip(*X))

    # Return the list of training points.
    return X


def prod(n):
    """Compute the product of the elements of a list of numbers."""
    p = 1
    for nn in n:
        p *= nn
    return p


if __name__ == '__main__':

    # Point counts along 1, 2, 3, 4 dimensions.
    n1 = 3
    n2 = [3, 4]
    n3 = [3, 4, 5]
    n4 = [3, 4, 5, 6]

    # Reference values for tests
    p1_ref = 3
    p2_ref = 12
    p3_ref = 60
    p4_ref = 360

    print('Testing point counts.')
    assert prod([n1]) == p1_ref
    assert prod(n2) == p2_ref
    assert prod(n3) == p3_ref
    assert prod(n4) == p4_ref

    print('Testing grid creatiion.')
    X1 = create_training_grid(n1)
    assert len(X1) == prod([n1])
    X2 = create_training_grid(n2)
    assert len(X2) == prod(n2)
    X3 = create_training_grid(n3)
    assert len(X3) == prod(n3)
    X4 = create_training_grid(n4)
    assert len(X4) == prod(n4)
