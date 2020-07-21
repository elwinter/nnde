"""
NeuralNetwork - Base class for neural networks

This module provides the base functionality for all neural network objects
used in the nnode software.

This class is currently an abstract base class. It must be subclassed
to be useful.

Example:
    Create an empty NeuralNetwork object.
        net = NeuralNetwork()

Attributes:
    None

Methods:
    __init__() - Constructor
    train() - Stub for training methods for subclasses
    run() - Stub for run methods for subclasses

Todo:
    * Add function annotations.
    * Add variable annotations.
"""


__all__ = []
__version__ = '0.0'
__author__ = 'Eric Winter (ewinter@stsci.edu)'


class NeuralNetwork:
    """Abstract base class for all neural network objects"""

    def __init__(self):
        """(abstract) Constructor for the neural network object."""
        pass

    def train(self):
        """(abstract) Train the neural network."""
        pass

    def run(self):
        """(abstract) Run the neural network using current parameters."""
        pass


if __name__ == '__main__':
    net = NeuralNetwork()
    print(net)
