import unittest

from nnde.differentialequation.differentialequation import DifferentialEquation
from nnde.exceptions.nndeexception import NNDEException


class TestBuilder(unittest.TestCase):

    def test___init__(self):
        with self.assertRaises(NNDEException):
            DifferentialEquation()

    def test_G(self):
        with self.assertRaises(NNDEException):
            DifferentialEquation.G(None)


if __name__ == '__main__':
    unittest.main()
