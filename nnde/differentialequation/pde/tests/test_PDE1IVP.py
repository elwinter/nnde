import unittest

from nnde.differentialequation.pde.pde1ivp import PDE1IVP
from nnde.exceptions.nndeexception import NNDEException


class TestBuilder(unittest.TestCase):

    def test___init__(self):
        PDE1IVP()
        # ADD CODE TO LOAD FROM MODULE.


if __name__ == '__main__':
    unittest.main()
