from math import cos, pi, sin
import unittest

from nnde.trialfunction.diff2dtrialfunction import Diff2DTrialFunction


#     # This is a 3-D problem.
#     m = 3

# Test boundary conditions
bc = [[lambda xyt: 0, lambda xyt: 0],
        [lambda xyt: 0, lambda xyt: 0],
        [lambda xyt: 0.5*sin(pi*xyt[0])*sin(pi*xyt[1]), None]]

# Test BC gradient
delbc = [[[lambda xyt: 0, lambda xyt: 0, lambda xyt: 0],
            [lambda xyt: 0, lambda xyt: 0, lambda xyt: 0]],
            [[lambda xyt: 0, lambda xyt: 0, lambda xyt: 0],
            [lambda xyt: 0, lambda xyt: 0, lambda xyt: 0]],
            [[lambda xyt: 0.5*pi*cos(pi*xyt[0])*sin(pi*xyt[1]),
            lambda xyt: 0.5*pi*sin(pi*xyt[0])*cos(pi*xyt[1]),
            lambda xyt: 0],
            [None, None, None]]]

# Test BC Laplacian
del2bc = [[[lambda xyt: 0, lambda xyt: 0, lambda xyt: 0],
            [lambda xyt: 0, lambda xyt: 0, lambda xyt: 0]],
            [[lambda xyt: 0, lambda xyt: 0, lambda xyt: 0],
            [lambda xyt: 0, lambda xyt: 0, lambda xyt: 0]],
            [[lambda xyt: -0.5*pi**2*sin(pi*xyt[0])*sin(pi*xyt[1]),
            lambda xyt: -0.5*pi**2*sin(pi*xyt[0])*sin(pi*xyt[1]),
            lambda xyt: 0],
            [None, None, None]]]

# Test inputs.
xyt_test = (0.4, 0.41, 0.42)
N_test = 0.5
delN_test = (0.61, 0.62, 0.63)
del2N_test = (0.71, 0.72, 0.73)

# # Reference values for tests.
A_ref = 0.26485513452284254
delA_ref = (0.27035493897666385, 0.24173779408724125, -0.45664678366007333)
del2A_ref = (-2.614015401337761, -2.614015401337761, 0.0)
P_ref = 0.02438352
delP_ref = (0.0203196, 0.018144, 0.058056)
del2P_ref = (-0.203196, -0.2016, 0)
Yt_ref = 0.27704689452284254
delYt_ref = (0.29538868617666386, 0.26592757648724125, -0.41225716606007334)
del2Yt_ref = (-2.673511190137761, -2.6747607069377612, 0.0909505)


class TestBuilder(unittest.TestCase):

    def test___init__(self):
        Diff2DTrialFunction(bc, delbc, del2bc)

    def test_A(self):
        tf = Diff2DTrialFunction(bc, delbc, del2bc)
        self.assertAlmostEqual(tf.A(xyt_test), A_ref)

    def test_delA(self):
        tf = Diff2DTrialFunction(bc, delbc, del2bc)
        for (a, b) in zip(tf.delA(xyt_test), delA_ref):
            self.assertAlmostEqual(a, b)

    def test_del2A(self):
        tf = Diff2DTrialFunction(bc, delbc, del2bc)
        for (a, b) in zip(tf.del2A(xyt_test), del2A_ref):
            self.assertAlmostEqual(a, b)

    def test_P(self):
        tf = Diff2DTrialFunction(bc, delbc, del2bc)
        self.assertAlmostEqual(tf.P(xyt_test), P_ref)

    def test_delP(self):
        tf = Diff2DTrialFunction(bc, delbc, del2bc)
        for (a, b) in zip(tf.delP(xyt_test), delP_ref):
            self.assertAlmostEqual(a, b)

    def test_del2P(self):
        tf = Diff2DTrialFunction(bc, delbc, del2bc)
        for (a, b) in zip(tf.del2P(xyt_test), del2P_ref):
            self.assertAlmostEqual(a, b)

    def test_Yt(self):
        tf = Diff2DTrialFunction(bc, delbc, del2bc)
        self.assertAlmostEqual(tf.Yt(xyt_test, N_test), Yt_ref)

    def test_delYt(self):
        tf = Diff2DTrialFunction(bc, delbc, del2bc)
        for (a, b) in zip(tf.delYt(xyt_test, N_test, delN_test), delYt_ref):
            self.assertAlmostEqual(a, b)

    def test_del2Yt(self):
        tf = Diff2DTrialFunction(bc, delbc, del2bc)
        for (a, b) in zip(tf.del2Yt(xyt_test, N_test, delN_test, del2N_test),
                          del2Yt_ref):
            self.assertAlmostEqual(a, b)


if __name__ == '__main__':
    unittest.main()
