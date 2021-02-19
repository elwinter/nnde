import unittest

from nnde.math.kdelta import kdelta

class TestBuilder(unittest.TestCase):

    def test_kdelta(self):
        for i in range(-2, 3):
            for j in range(-2, 3):
                if i == j:
                    self.assertEqual(kdelta(i, j), 1)
                else:
                    self.assertEqual(kdelta(i, j), 0)


if __name__ == '__main__':
    unittest.main()
