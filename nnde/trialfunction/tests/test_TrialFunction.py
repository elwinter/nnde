import unittest

from nnde.trialfunction.trialfunction import TrialFunction


class TestBuilder(unittest.TestCase):

    def test_Yt(self):
        with self.assertRaises(Exception):
            TrialFunction().Yt()


if __name__ == '__main__':
    unittest.main()
