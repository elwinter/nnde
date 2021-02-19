import unittest

from nnde.differentialequation.pde.pde import PDE


class TestBuilder(unittest.TestCase):

    def test___init__(self):
        PDE()


if __name__ == '__main__':
    unittest.main()
