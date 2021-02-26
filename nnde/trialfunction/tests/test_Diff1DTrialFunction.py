from math import cos, pi, sin
import unittest

from nnde.trialfunction.diff1dtrialfunction import Diff1DTrialFunction


# This is a 2-D problem.
# m = 2

# Test boundary conditions
bc = [[lambda xt: 0, lambda xt: 0],
    [lambda xt: sin(pi*xt[0]), None]]

# Test BC gradient
delbc = [[[lambda xt: 0, lambda xt: 0], [lambda xt: 0, lambda xt: 0]],
        [[lambda xt: pi*cos(pi*xt[0]), lambda xt: 0], [None, None]]]

# Test BC Laplacian
del2bc = [[[lambda xt: 0, lambda xt: 0], [lambda xt: 0, lambda xt: 0]],
        [[lambda xt: -pi**2*sin(pi*xt[0]), lambda xt: 0], [None, None]]]

# Test inputs.
xt_test = [0.4, 0.41]
N_test = 0.5
delN_test = [0.61, 0.62]
del2N_test = [0.71, 0.72]

# Reference values for tests.
A_ref = 0.5611233446141407
delA_ref = (0.5727752564240126, -0.9510565162951535)
del2A_ref = (-5.538065431557704, 0)
P_ref = 0.09839999999999999
delP_ref = (0.082, 0.24)
del2P_ref = [-0.82, 0]
Yt_ref = 0.6103233446141407
delYt_ref = [0.6737992564240126, -0.7700485162951536]
del2Yt_ref = [-5.778161431557704, 0.368448]


class TestBuilder(unittest.TestCase):

    def test___init__(self):
        Diff1DTrialFunction(bc, delbc, del2bc)

    def test_A(self):
        tf = Diff1DTrialFunction(bc, delbc, del2bc)
        self.assertAlmostEqual(tf.A(xt_test), A_ref)

    def test_delA(self):
        tf = Diff1DTrialFunction(bc, delbc, del2bc)
        for (a, b) in zip(tf.delA(xt_test), delA_ref):
            self.assertAlmostEqual(a, b)

    def test_del2A(self):
        tf = Diff1DTrialFunction(bc, delbc, del2bc)
        for (a, b) in zip(tf.del2A(xt_test), del2A_ref):
            self.assertAlmostEqual(a, b)

    def test_P(self):
        tf = Diff1DTrialFunction(bc, delbc, del2bc)
        self.assertAlmostEqual(tf.P(xt_test), P_ref)

    def test_delP(self):
        tf = Diff1DTrialFunction(bc, delbc, del2bc)
        for (a, b) in zip(tf.delP(xt_test), delP_ref):
            self.assertAlmostEqual(a, b)

    def test_del2P(self):
        tf = Diff1DTrialFunction(bc, delbc, del2bc)
        for (a, b) in zip(tf.del2P(xt_test), del2P_ref):
            self.assertAlmostEqual(a, b)

    def test_Yt(self):
        tf = Diff1DTrialFunction(bc, delbc, del2bc)
        self.assertAlmostEqual(tf.Yt(xt_test, N_test), Yt_ref)

    def test_delYt(self):
        tf = Diff1DTrialFunction(bc, delbc, del2bc)
        for (a, b) in zip(tf.delYt(xt_test, N_test, delN_test), delYt_ref):
            self.assertAlmostEqual(a, b)

    def test_del2Yt(self):
        tf = Diff1DTrialFunction(bc, delbc, del2bc)
        for (a, b) in zip(tf.del2Yt(xt_test, N_test, delN_test, del2N_test),
                          del2Yt_ref):
            self.assertAlmostEqual(a, b)


if __name__ == '__main__':
    unittest.main()
