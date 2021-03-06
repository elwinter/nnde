import unittest

from nnde.exceptions.nndeexception import NNDEException
from nnde.neuralnetwork.neuralnetwork import NeuralNetwork


class TestNeuralNetwork(unittest.TestCase):

    def test___init__(self):
        with self.assertRaises(NNDEException):
            NeuralNetwork()

    def test_train(self):
        with self.assertRaises(NNDEException):
            NeuralNetwork.train(None)

    def test_run(self):
        with self.assertRaises(NNDEException):
            NeuralNetwork.run(None)


if __name__ == '__main__':
    unittest.main()
