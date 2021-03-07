import numpy as np

from nnde.differentialequation.examples.diff2d_halfsine_increase import (
    a, D,
    G, dG_dY, dG_ddelY, dG_ddel2Y,
    bc, f0, f1, g0, g1, Y0, delbc, del2bc,
    A, delA, del2A,
)


if __name__ == '__main__':

    # Test values
    xyt_test = (0.4, 0.5, 0.6)
    m = len(xyt_test)
    Y_test = 0.11
    delY_test = (0.22, 0.33, 0.44)
    del2Y_test = (-0.55, -0.66, -0.77)

    # # Reference values for tests.
    G_ref = 0.561
    bc_ref = ((0, 0.6),
              (0, 0),
              (0.951056516295154, None))
    delbc_ref = (((0, 0, 0), (0, 0, a)),
                 ((0, 0, 0), (0, 0, 0)),
                 ((0.970805519362733, 0, 0), (None, None, None)))
    del2bc_ref = (((0, 0, 0), (0, -5.921762640653615, 0)),
                  ((0, 0, 0), (0, 0, 0)),
                  ((-9.38655157891136, -9.38655157891136, 0),
                   (None, None, None)))
    dG_dY_ref = 0
    dG_ddelY_ref = (0, 0, 1)
    dG_ddel2Y_ref = (-D, -D, 0)
    A_ref = 0.6204226065180613
    delA_ref = [0.9883222077450933, 0, -0.5510565162951535]
    del2A_ref = [-3.7546206315645443, -6.12332568782599, 0]

    print("Testing differential equation.")
    assert np.isclose(G(xyt_test, Y_test, delY_test, del2Y_test), G_ref)

    print('Testing boundary conditions.')
    for j in range(m):
        for jj in range(2):
            if bc[j][jj] is not None:
                assert np.isclose(bc[j][jj](xyt_test), bc_ref[j][jj])

    print("Testing boundary condition continuity constraints.")
    assert np.isclose(f0([0, 0, 0]), g0([0, 0, 0]))
    assert np.isclose(f0([0, 0, 0]), Y0([0, 0, 0]))
    assert np.isclose(f1([1, 0, 0]), g0([1, 0, 0]))
    assert np.isclose(f1([1, 0, 0]), Y0([1, 0, 0]))
    assert np.isclose(f1([1, 1, 0]), g1([1, 1, 0]))
    assert np.isclose(f1([1, 1, 0]), Y0([1, 1, 0]))
    assert np.isclose(f0([0, 1, 0]), g1([0, 1, 0]))
    assert np.isclose(f0([0, 1, 0]), Y0([0, 1, 0]))
    # t=1 not used

    print('Testing boundary condition gradients.')
    for j in range(m):
        for jj in range(2):
            for jjj in range(m):
                if delbc[j][jj][jjj] is not None:
                    assert np.isclose(delbc[j][jj][jjj](xyt_test),
                                      delbc_ref[j][jj][jjj])

    print('Testing boundary condition Laplacians.')
    for j in range(m):
        for jj in range(2):
            for jjj in range(m):
                if del2bc[j][jj][jjj] is not None:
                    assert np.isclose(del2bc[j][jj][jjj](xyt_test),
                                      del2bc_ref[j][jj][jjj])

    print('Testing derivative of differential equation wrt solution.')
    assert np.isclose(dG_dY(xyt_test, Y_test, delY_test, del2Y_test),
                      dG_dY_ref)

    print('Testing derivative of differential equation wrt gradient '
          'components.')
    for j in range(m):
        assert np.isclose(dG_ddelY[j](xyt_test, Y_test, delY_test, del2Y_test),
                          dG_ddelY_ref[j])

    print('Testing derivative of differential equation wrt Laplacian '
          'components.')
    for j in range(m):
        assert np.isclose(dG_ddel2Y[j](xyt_test, Y_test, delY_test,
                                       del2Y_test),
                          dG_ddel2Y_ref[j])

    print("Testing optimized BC function.")
    A_ = A(xyt_test)
    assert np.isclose(A_, A_ref)

    print("Testing optimized BC function gradient.")
    delA_ = delA(xyt_test)
    for j in range(m):
        assert np.isclose(delA_[j], delA_ref[j])

    print("Testing optimized BC function Laplacian.")
    del2A_ = del2A(xyt_test)
    for j in range(m):
        assert np.isclose(del2A_[j], del2A_ref[j])
