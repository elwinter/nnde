import unittest

from nnde.differentialequation.pde.pde1 import PDE1


class TestBuilder(unittest.TestCase):

    def test___init__(self):
        PDE1()

    def test_G(self):
        with self.assertRaises(Exception):
            PDE1().G()


if __name__ == '__main__':
    unittest.main()
