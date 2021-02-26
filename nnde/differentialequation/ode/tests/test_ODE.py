import unittest

from nnde.differentialequation.ode.ode import ODE
from nnde.exceptions.nndeexception import NNDEException


class TestBuilder(unittest.TestCase):

    def test___init__(self):
        with self.assertRaises(NNDEException):
            ODE()


if __name__ == '__main__':
    unittest.main()
