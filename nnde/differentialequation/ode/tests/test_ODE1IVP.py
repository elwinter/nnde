import unittest

from nnde.differentialequation.ode.ode1ivp import ODE1IVP


class TestBuilder(unittest.TestCase):

    def test___init__(self):
        ODE1IVP()
        ODE1IVP()

if __name__ == '__main__':
    unittest.main()
