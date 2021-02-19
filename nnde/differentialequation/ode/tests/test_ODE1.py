import unittest

from nnde.differentialequation.ode.ode1 import ODE1


class TestBuilder(unittest.TestCase):

    def test___init__(self):
        ODE1()

    def test_G(self):
        with self.assertRaises(Exception):
            ODE1().G()


if __name__ == '__main__':
    unittest.main()
