import unittest

from nnde.exceptions.nndeexception import NNDEException
from nnde.neuralnetwork.slffnn import SLFFNN


class TestSLFFNN(unittest.TestCase):

    def test___init__(self):
        with self.assertRaises(NNDEException):
            SLFFNN()


if __name__ == '__main__':
    unittest.main()
