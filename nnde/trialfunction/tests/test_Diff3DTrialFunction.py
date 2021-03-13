from math import cos, pi, sin
import unittest

from nnde.trialfunction.diff3dtrialfunction import Diff3DTrialFunction


# Test boundary conditions
bc = [[lambda xyzt: 0, lambda xyzt: 0],
        [lambda xyzt: 0, lambda xyzt: 0],
        [lambda xyzt: 0, lambda xyzt: 0],
        [lambda xyzt: sin(pi*xyzt[0])*sin(pi*xyzt[1])*sin(pi*xyzt[2])/3,
        None]]

# Test BC gradient
delbc = [[[lambda xyzt: 0, lambda xyzt: 0, lambda xyzt: 0, lambda xyzt: 0],
            [lambda xyzt: 0, lambda xyzt: 0, lambda xyzt: 0,
            lambda xyzt: 0]],
            [[lambda xyzt: 0, lambda xyzt: 0, lambda xyzt: 0, lambda xyzt: 0],
            [lambda xyzt: 0, lambda xyzt: 0, lambda xyzt: 0,
            lambda xyzt: 0]],
            [[lambda xyzt: 0, lambda xyzt: 0, lambda xyzt: 0, lambda xyzt: 0],
            [lambda xyzt: 0, lambda xyzt: 0, lambda xyzt: 0,
            lambda xyzt: 0]],
            [[lambda xyzt: pi*cos(pi*xyzt[0])*sin(pi*xyzt[1])
            * sin(pi*xyzt[2])/3,
            lambda xyzt: pi*sin(pi*xyzt[0])*cos(pi*xyzt[1])
            * sin(pi*xyzt[2])/3,
            lambda xyzt: pi*sin(pi*xyzt[0])*sin(pi*xyzt[1])
            * cos(pi*xyzt[2])/3,
            lambda xyzt: 0],
            [None, None, None, None]]]

# Test BC Laplacian
del2bc = [[[lambda xyzt: 0, lambda xyzt: 0, lambda xyzt: 0,
            lambda xyzt: 0],
            [lambda xyzt: 0, lambda xyzt: 0, lambda xyzt: 0,
            lambda xyzt: 0]],
            [[lambda xyzt: 0, lambda xyzt: 0, lambda xyzt: 0,
            lambda xyzt: 0],
            [lambda xyzt: 0, lambda xyzt: 0, lambda xyzt: 0,
            lambda xyzt: 0]],
            [[lambda xyzt: 0, lambda xyzt: 0, lambda xyzt: 0,
            lambda xyzt: 0],
            [lambda xyzt: 0, lambda xyzt: 0, lambda xyzt: 0,
            lambda xyzt: 0]],
            [[lambda xyzt: -pi**2*sin(pi*xyzt[0])*sin(pi*xyzt[1])
            * sin(pi*xyzt[2])/3,
            lambda xyzt: -pi**2*sin(pi*xyzt[0])*sin(pi*xyzt[1])
            * sin(pi*xyzt[2])/3,
            lambda xyzt: -pi**2*sin(pi*xyzt[0])*sin(pi*xyzt[1])
            * sin(pi*xyzt[2])/3,
            lambda xyzt: 0],
            [None, None, None, None]]]

# Test inputs
xyzt_test = [0.4, 0.41, 0.42, 0.43]
N_test = 0.5
delN_test = [0.61, 0.62, 0.63, 0.64]
del2N_test = [0.71, 0.72, 0.73, 0.74]

# Reference values for tests.
A_ref = 0.16807414638994445
delA_ref = (0.17156426162049485, 0.15340413718842627, 0.13557262269283618,
            -0.29486692349113064)
del2A_ref = (-1.6588253349195334, -1.6588253349195334, -1.6588253349195334, 0)
P_ref = 0.00608125
delP_ref = (0.00506771, 0.00452511, 0.00399425, 0.0141424)
del2P_ref = (-0.0506771, -0.050279, -0.0499282, 0)
Yt_ref = 0.17111477133394445
delYt_ref = (0.17780767817217485, 0.1594370689189863, 0.14140093652227617,
             -0.28390370276281063)
del2Yt_ref = (-1.6736635846462535, -1.6739752141361735, -1.6743173439732935,
              0.0226025)


class TestBuilder(unittest.TestCase):

    def test___init__(self):
        Diff3DTrialFunction(bc, delbc, del2bc)

    def test_A(self):
        tf = Diff3DTrialFunction(bc, delbc, del2bc)
        self.assertAlmostEqual(tf.A(xyzt_test), A_ref)

    def test_delA(self):
        tf = Diff3DTrialFunction(bc, delbc, del2bc)
        for (a, b) in zip(tf.delA(xyzt_test), delA_ref):
            self.assertAlmostEqual(a, b)

    def test_del2A(self):
        tf = Diff3DTrialFunction(bc, delbc, del2bc)
        for (a, b) in zip(tf.del2A(xyzt_test), del2A_ref):
            self.assertAlmostEqual(a, b)

    def test_P(self):
        tf = Diff3DTrialFunction(bc, delbc, del2bc)
        self.assertAlmostEqual(tf.P(xyzt_test), P_ref)

    def test_delP(self):
        tf = Diff3DTrialFunction(bc, delbc, del2bc)
        for (a, b) in zip(tf.delP(xyzt_test), delP_ref):
            self.assertAlmostEqual(a, b)

    def test_del2P(self):
        tf = Diff3DTrialFunction(bc, delbc, del2bc)
        for (a, b) in zip(tf.del2P(xyzt_test), del2P_ref):
            self.assertAlmostEqual(a, b)

    def test_Yt(self):
        tf = Diff3DTrialFunction(bc, delbc, del2bc)
        self.assertAlmostEqual(tf.Yt(xyzt_test, N_test), Yt_ref)

    def test_delYt(self):
        tf = Diff3DTrialFunction(bc, delbc, del2bc)
        for (a, b) in zip(tf.delYt(xyzt_test, N_test, delN_test), delYt_ref):
            self.assertAlmostEqual(a, b)

    def test_del2Yt(self):
        tf = Diff3DTrialFunction(bc, delbc, del2bc)
        for (a, b) in zip(tf.del2Yt(xyzt_test, N_test, delN_test, del2N_test),
                          del2Yt_ref):
            self.assertAlmostEqual(a, b)


if __name__ == '__main__':
    unittest.main()
