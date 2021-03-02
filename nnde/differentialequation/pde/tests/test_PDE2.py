import unittest

from nnde.differentialequation.pde.pde2 import PDE2
from nnde.exceptions.nndeexception import NNDEException


class TestBuilder(unittest.TestCase):

    def test___init__(self):
        with self.assertRaises(NNDEException):
            PDE2()

    def test_G(self):
        with self.assertRaises(NNDEException):
            PDE2.G(None, None, None, None, None)


if __name__ == '__main__':
    unittest.main()
