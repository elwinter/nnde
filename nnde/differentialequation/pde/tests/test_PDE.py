import unittest

from nnde.differentialequation.pde.pde import PDE
from nnde.exceptions.nndeexception import NNDEException


class TestBuilder(unittest.TestCase):

    def test___init__(self):
        with self.assertRaises(NNDEException):
            PDE()


if __name__ == '__main__':
    unittest.main()
