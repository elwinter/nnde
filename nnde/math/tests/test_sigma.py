import unittest

from nnde.math.sigma import sigma, dsigma_dz, d2sigma_dz2, d3sigma_dz3
from nnde.math.sigma import d4sigma_dz4
from nnde.math.sigma import s, s1, s2, s3, s4


class TestBuilder(unittest.TestCase):

    def test_sigma(self):
        self.assertAlmostEqual(sigma(0), 0.5)
        self.assertAlmostEqual(sigma(1), 0.731058578630005)

    def test_dsigma_dz(self):
        self.assertAlmostEqual(dsigma_dz(0), 0.25)
        self.assertAlmostEqual(dsigma_dz(1), 0.196611933241482)

    def test_d2sigma_dz2(self):
        self.assertAlmostEqual(d2sigma_dz2(0), 0)
        self.assertAlmostEqual(d2sigma_dz2(1), -0.0908577476729484)

    def test_d3sigma_dz3(self):
        self.assertAlmostEqual(d3sigma_dz3(0), -0.125)
        self.assertAlmostEqual(d3sigma_dz3(1), -0.0353255805162356)

    def test_d4sigma_dz4(self):
        self.assertAlmostEqual(d4sigma_dz4(0), 0)
        self.assertAlmostEqual(d4sigma_dz4(1), 0.123506861366393)

    def test_s(self):
        self.assertAlmostEqual(s(0), 0.5)
        self.assertAlmostEqual(s(1), 0.731058578630005)

    def test_s1(self):
        self.assertAlmostEqual(s1(s(0)), 0.25)
        self.assertAlmostEqual(s1(s(1)), 0.196611933241482)

    def test_s2(self):
        self.assertAlmostEqual(s2(s(0)), 0)
        self.assertAlmostEqual(s2(s(1)), -0.0908577476729484)

    def test_s3(self):
        self.assertAlmostEqual(s3(s(0)), -0.125)
        self.assertAlmostEqual(s3(s(1)), -0.0353255805162356)

    def test_s4(self):
        self.assertAlmostEqual(s4(s(0)), 0)
        self.assertAlmostEqual(s4(s(1)), 0.123506861366393)


if __name__ == '__main__':
    unittest.main()
