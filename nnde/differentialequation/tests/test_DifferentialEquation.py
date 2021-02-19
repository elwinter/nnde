import unittest

from nnde.differentialequation.differentialequation import DifferentialEquation


class TestBuilder(unittest.TestCase):

    def test___init__(self):
        DifferentialEquation()

    def test_G(self):
        with self.assertRaises(Exception):
            DifferentialEquation().G()


if __name__ == '__main__':
    unittest.main()
