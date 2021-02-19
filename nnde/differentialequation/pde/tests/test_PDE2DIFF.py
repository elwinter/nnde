import unittest

from nnde.differentialequation.pde.pde2diff import PDE2DIFF


class TestBuilder(unittest.TestCase):

    def test___init__(self):
        PDE2DIFF()

    def test_G(self):
        with self.assertRaises(Exception):
            PDE2DIFF().G()


if __name__ == '__main__':
    unittest.main()
