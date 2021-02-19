import unittest

from nnde.differentialequation.pde.pde1ivp import PDE1IVP


class TestBuilder(unittest.TestCase):

    def test___init__(self):
        PDE1IVP()


if __name__ == '__main__':
    unittest.main()
