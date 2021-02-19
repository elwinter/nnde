import unittest

from nnde.differentialequation.ode.ode import ODE


class TestBuilder(unittest.TestCase):

    def test___init__(self):
        ODE()


if __name__ == '__main__':
    unittest.main()
