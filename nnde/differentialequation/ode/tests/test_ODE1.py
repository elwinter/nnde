import unittest

from nnde.differentialequation.ode.ode1 import ODE1
from nnde.exceptions.nndeexception import NNDEException


class TestBuilder(unittest.TestCase):

    def test___init__(self):
        with self.assertRaises(NNDEException):
            ODE1()

    def test_G(self):
        with self.assertRaises(NNDEException):
            ODE1.G(None, 0, 0, 0)


if __name__ == '__main__':
    unittest.main()
