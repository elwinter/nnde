import numpy as np

from nnde.differentialequation.examples.diff1d_halfsine_increase import (
    a, D,
    G, dG_dY, dG_ddelY, dG_ddel2Y,
    bc, f0, f1, Y0, delbc, del2bc,
    A, delA, del2A,
    Ya, delYa, del2Ya,
)


if __name__ == '__main__':

    # Test values
    x_test = (0.4, 0.5)
    m = len(x_test)
    Y_test = 0.5986958383019955
    delY_test = (0.6383788665776866, -0.536469420056056)
    del2Y_test = (-5.724723108734729, 0.5680753052926439)

    # Reference values for tests.
    G_ref = 0.03600289081741688   # Non-zero due to roundoff error?
    bc_ref = ((0, 0.05),
              (0.951057, None))
    delbc_ref = [[[0, 0], [0, a]],
                 [[0.970805519362733, 0], [None, None]]]
    del2bc_ref = [[[0, 0], [0, 0]],
                  [[-9.38655157891136, 0], [None, None]]]
    dG_dY_ref = 0
    dG_ddelY_ref = (0, 1)
    dG_ddel2Y_ref = (-D, 0)
    A_ref = 0.4955282581475768
    delA_ref = [0.5354027596813666, -0.9110565162951535]
    del2A_ref = [-4.69327578945568, 0]
    Ya_ref = 0.5986958383019955
    delYa_ref = [0.6383788665776866, -0.536469420056056]
    del2Ya_ref = [-5.724723108734729, 0.5680753052926439]

    print("Testing differential equation.")
    assert np.isclose(G(x_test, Y_test, delY_test, del2Y_test), G_ref)

    print('Testing boundary conditions.')
    for j in range(m):
        for jj in range(2):
            if bc[j][jj] is not None:
                assert np.isclose(bc[j][jj](x_test), bc_ref[j][jj])

    print("Testing boundary condition continuity constraints.")
    assert np.isclose(f0([0, 0]), Y0([0, 0]))
    assert np.isclose(f1([1, 0]), Y0([1, 0]))
    # t=1 not used

    print('Testing boundary condition gradients.')
    for j in range(m):
        for jj in range(2):
            for jjj in range(m):
                if delbc[j][jj][jjj] is not None:
                    assert np.isclose(delbc[j][jj][jjj](x_test),
                                      delbc_ref[j][jj][jjj])

    print('Testing boundary condition Laplacians.')
    for j in range(m):
        for jj in range(2):
            for jjj in range(m):
                if del2bc[j][jj][jjj] is not None:
                    assert np.isclose(del2bc[j][jj][jjj](x_test),
                                      del2bc_ref[j][jj][jjj])

    print('Testing derivative of differential equation wrt solution.')
    assert np.isclose(dG_dY(x_test, Y_test, delY_test, del2Y_test),
                      dG_dY_ref)

    print('Testing derivative of differential equation wrt gradient '
          'components.')
    for j in range(m):
        assert np.isclose(dG_ddelY[j](x_test, Y_test, delY_test, del2Y_test),
                          dG_ddelY_ref[j])

    print('Testing derivative of differential equation wrt Laplacian '
          'components.')
    for j in range(m):
        assert np.isclose(dG_ddel2Y[j](x_test, Y_test, delY_test, del2Y_test),
                          dG_ddel2Y_ref[j])

    print("Testing optimized BC function.")
    A_ = A(x_test)
    assert np.isclose(A_, A_ref)

    print("Testing optimized BC function gradient.")
    delA_ = delA(x_test)
    for j in range(m):
        assert np.isclose(delA_[j], delA_ref[j])

    print("Testing optimized BC function Laplacian.")
    del2A_ = del2A(x_test)
    for j in range(m):
        assert np.isclose(del2A_[j], del2A_ref[j])

    print("Testing analytical solution.")
    assert np.isclose(Ya(x_test), Ya_ref)

    print("Testing analytical solution gradient.")
    for j in range(m):
        assert np.isclose(delYa[j](x_test), delYa_ref[j])

    print("Testing analytical solution Laplacian.")
    for j in range(m):
        assert np.isclose(del2Ya[j](x_test), del2Ya_ref[j])
