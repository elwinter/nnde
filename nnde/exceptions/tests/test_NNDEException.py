import unittest

from nnde.exceptions.nndeexception import NNDEException


class TestNNDEException(unittest.TestCase):

    def test___init__(self):
        NNDEException()


if __name__ == '__main__':
    unittest.main()
