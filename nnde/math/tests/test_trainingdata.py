import unittest

from nnde.math.trainingdata import create_training_grid, prod

class TestBuilder(unittest.TestCase):

    def test_create_training_grid(self):
        n1 = 3
        n2 = [3, 4]
        n3 = [3, 4, 5]
        n4 = [3, 4, 5, 6]
        X = create_training_grid(n1)
        self.assertEqual(len(X), n1)
        X = create_training_grid(n2)
        assert len(X) == n2[0]*n2[1]
        X = create_training_grid(n3)
        assert len(X) == n3[0]*n3[1]*n3[2]
        X = create_training_grid(n4)
        assert len(X) == n4[0]*n4[1]*n4[2]*n4[3]

    def test_prod(self):
        n1 = 3
        n2 = [3, 4]
        n3 = [3, 4, 5]
        n4 = [3, 4, 5, 6]
        self.assertEqual(prod([n1]), n1)
        self.assertEqual(prod(n2), n2[0]*n2[1])
        self.assertEqual(prod(n3), n3[0]*n3[1]*n3[2])
        self.assertEqual(prod(n4), n4[0]*n4[1]*n4[2]*n4[3])


if __name__ == '__main__':
    unittest.main()
