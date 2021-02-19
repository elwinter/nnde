import unittest

from nnde.differentialequation.pde.pde2 import PDE2


class TestBuilder(unittest.TestCase):

    def test___init__(self):
        PDE2()

    def test_G(self):
        with self.assertRaises(Exception):
            PDE2().G()


if __name__ == '__main__':
    unittest.main()
