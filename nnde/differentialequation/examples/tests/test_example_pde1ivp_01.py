import numpy as np

from nnde.differentialequation.examples.example_pde1ivp_01 import (
    G, dG_dY, dG_ddelY,
    bc, delbc,
    Ya, delYa,
)


if __name__ == '__main__':

    # Values to use for testing.
    x_test = [0.4, 0.5]
    m = len(x_test)
    Y_test = 0.6
    delY_test = [0.7, 0.8]

    # Reference values for tests
    G_ref = 2.2
    bc_ref = [[0.479426, None], [-0.198669, None]]
    delbc_ref = [
        [[0, 0.877583], [None, None]],
        [[-0.490033, 0], [None, None]]
    ]
    dG_dY_ref = 0
    dG_ddelY_ref = [2, 1]
    Ya_ref = 0.29552
    delYa_ref = [-0.477668, 0.955336]

    print('Testing differential equation.')
    assert np.isclose(G(x_test, Y_test, delY_test), G_ref)

    print('Testing boundary conditions.')
    for j in range(m):
        assert np.isclose(bc[j][0](x_test), bc_ref[j][0])
        assert bc[j][1](x_test) is bc_ref[j][1]  # None since not used.

    print('Testing boundary condition gradients.')
    for j in range(m):
        for jj in range(m):
            assert np.isclose(delbc[j][0][jj](x_test), delbc_ref[j][0][jj])
            assert delbc[j][1][jj](x_test) is delbc_ref[j][1][jj]  # None

    print('Testing derivative of differential equation wrt solution.')
    assert np.isclose(dG_dY(x_test, Y_test, delY_test), dG_dY_ref)

    print('Testing derivative of differential equation wrt gradient '
          'components.')
    for j in range(m):
        assert np.isclose(dG_ddelY[j](x_test, Y_test, delY_test),
                          dG_ddelY_ref[j])

    print('Testing analytical solution.')
    assert np.isclose(Ya(x_test), Ya_ref)

    print('Testing analytical gradient.')
    for j in range(m):
        assert np.isclose(delYa[j](x_test), delYa_ref[j])
