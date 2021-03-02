import unittest

from nnde.differentialequation.pde.pde1 import PDE1
from nnde.exceptions.nndeexception import NNDEException


class TestBuilder(unittest.TestCase):

    def test___init__(self):
        with self.assertRaises(NNDEException):
            PDE1()

    def test_G(self):
        with self.assertRaises(NNDEException):
            PDE1.G(None, None, None, None)


if __name__ == '__main__':
    unittest.main()
