import numpy as np

from nnde.differentialequation.examples.diff3d_halfsine import (
    D,
    G, dG_dY, dG_ddelY, dG_ddel2Y,
    bc, f0, f1, g0, g1, h0, h1, Y0, delbc, del2bc,
    A, delA, del2A,
    Ya, delYa, del2Ya,
)


if __name__ == '__main__':

    # Test values
    xyzt_test = [0.4, 0.5, 0.6, 0.7]
    m = len(xyzt_test)
    Y_test = 0.1138378166982095
    delY_test = [0.11620169660719469, 0, -0.11620169660719464,
                 -0.3370602650085157]
    del2Y_test = [-1.1235342166950524, -1.1235342166950524,
                  -1.1235342166950524, 0.9979954424881178]

    # Reference values for tests.
    G_ref = 0
    bc_ref = [[0, 0], [0, 0], [0, 0], [0.904508497187474, None]]
    delbc_ref = [[[0, 0, 0, 0], [0, 0, 0, 0]],
                 [[0, 0, 0, 0], [0, 0, 0, 0]],
                 [[0, 0, 0, 0], [0, 0, 0, 0]],
                 [[0.9232909152452285, 0, -0.923290915245228, 0],
                  [None, None, None, None]]]
    del2bc_ref = [[[0, 0, 0, 0], [0, 0, 0, 0]],
                  [[0, 0, 0, 0], [0, 0, 0, 0]],
                  [[0, 0, 0, 0], [0, 0, 0, 0]],
                  [[-8.927141044664213, -8.927141044664213,
                    -8.927141044664213, 0],
                   [None, None, None, None]]]
    dG_dY_ref = 0
    dG_ddelY_ref = [0, 0, 0, 1]
    dG_ddel2Y_ref = [-D, -D, -D, 0]
    A_ref = 0.2713525491562422
    delA_ref = [0.2769872745735686, 0, -0.2769872745735685,
                -0.9045084971874737]
    del2A_ref = [-2.6781423133992637, -2.6781423133992637,
                 -2.6781423133992637, 0]
    Ya_ref = 0.1138378166982095
    delYa_ref = [0.11620169660719469, 0, -0.11620169660719464,
                 -0.3370602650085157]
    del2Ya_ref = [-1.1235342166950524, -1.1235342166950524,
                  -1.1235342166950524, 0.9979954424881178]

    print("Testing differential equation.")
    assert np.isclose(G(xyzt_test, Y_test, delY_test, del2Y_test), G_ref)

    print('Testing boundary conditions.')
    for j in range(m):
        for jj in range(2):
            if bc[j][jj] is not None:
                assert np.isclose(bc[j][jj](xyzt_test), bc_ref[j][jj])

    print("Testing boundary condition continuity constraints.")
    assert np.isclose(f0([0, 0, 0, 0]), g0([0, 0, 0, 0]))
    assert np.isclose(f0([0, 0, 0, 0]), h0([0, 0, 0, 0]))
    assert np.isclose(f0([0, 0, 0, 0]), Y0([0, 0, 0, 0]))
    assert np.isclose(f1([1, 0, 0, 0]), g0([1, 0, 0, 0]))
    assert np.isclose(f1([1, 0, 0, 0]), h0([1, 0, 0, 0]))
    assert np.isclose(f1([1, 0, 0, 0]), Y0([1, 0, 0, 0]))
    assert np.isclose(f1([1, 1, 0, 0]), g1([1, 1, 0, 0]))
    assert np.isclose(f1([1, 1, 0, 0]), h0([1, 1, 0, 0]))
    assert np.isclose(f1([1, 1, 0, 0]), Y0([1, 1, 0, 0]))
    assert np.isclose(f0([0, 1, 0, 0]), g1([0, 1, 0, 0]))
    assert np.isclose(f0([0, 1, 0, 0]), h0([0, 1, 0, 0]))
    assert np.isclose(f0([0, 1, 0, 0]), Y0([0, 1, 0, 0]))
    assert np.isclose(f0([0, 0, 1, 0]), g0([0, 0, 1, 0]))
    assert np.isclose(f0([0, 0, 1, 0]), h1([0, 0, 1, 0]))
    assert np.isclose(f0([0, 0, 1, 0]), Y0([0, 0, 1, 0]))
    assert np.isclose(f1([1, 0, 1, 0]), g0([1, 0, 1, 0]))
    assert np.isclose(f1([1, 0, 1, 0]), h1([1, 0, 1, 0]))
    assert np.isclose(f1([1, 0, 1, 0]), Y0([1, 0, 1, 0]))
    assert np.isclose(f1([1, 1, 1, 0]), g1([1, 1, 1, 0]))
    assert np.isclose(f1([1, 1, 1, 0]), h1([1, 1, 1, 0]))
    assert np.isclose(f1([1, 1, 1, 0]), Y0([1, 1, 1, 0]))
    assert np.isclose(f0([0, 1, 1, 0]), g1([0, 1, 1, 0]))
    assert np.isclose(f0([0, 1, 1, 0]), h1([0, 1, 1, 0]))
    assert np.isclose(f0([0, 1, 1, 0]), Y0([0, 1, 1, 0]))
    # t=1 not used

    print('Testing boundary condition gradients.')
    for j in range(m):
        for jj in range(2):
            for jjj in range(m):
                if delbc[j][jj][jjj] is not None:
                    assert np.isclose(delbc[j][jj][jjj](xyzt_test),
                                      delbc_ref[j][jj][jjj])

    print('Testing boundary condition Laplacians.')
    for j in range(m):
        for jj in range(2):
            for jjj in range(m):
                if del2bc[j][jj][jjj] is not None:
                    assert np.isclose(del2bc[j][jj][jjj](xyzt_test),
                                      del2bc_ref[j][jj][jjj])

    print('Testing derivative of differential equation wrt solution.')
    assert np.isclose(dG_dY(xyzt_test, Y_test, delY_test, del2Y_test),
                      dG_dY_ref)

    print('Testing derivative of differential equation wrt gradient '
          'components.')
    for j in range(m):
        assert np.isclose(dG_ddelY[j](xyzt_test, Y_test, delY_test,
                                      del2Y_test),
                          dG_ddelY_ref[j])

    print('Testing derivative of differential equation wrt Laplacian '
          'components.')
    for j in range(m):
        assert np.isclose(dG_ddel2Y[j](xyzt_test, Y_test, delY_test,
                                       del2Y_test),
                          dG_ddel2Y_ref[j])

    print("Testing optimized BC function.")
    A_ = A(xyzt_test)
    assert np.isclose(A_, A_ref)

    print("Testing optimized BC function gradient.")
    delA_ = delA(xyzt_test)
    for j in range(m):
        assert np.isclose(delA_[j], delA_ref[j])

    print("Testing optimized BC function Laplacian.")
    del2A_ = del2A(xyzt_test)
    for j in range(m):
        assert np.isclose(del2A_[j], del2A_ref[j])

    print("Testing analytical solution.")
    assert np.isclose(Ya(xyzt_test), Ya_ref)

    print("Testing analytical solution gradient.")
    for j in range(m):
        assert np.isclose(delYa[j](xyzt_test), delYa_ref[j])

    print("Testing analytical solution Laplacian.")
    for j in range(m):
        assert np.isclose(del2Ya[j](xyzt_test), del2Ya_ref[j])
