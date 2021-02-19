"""
SLFFNN - Base class for single-layer feed-forward neural networks

This module provides the base functionality for all single-layer feed-forward
neural network objects used in the nnode software.

This class is currently an abstract base class. It must be subclassed
to be useful.

Example:
    Create an empty SLFFNN object.
        net = SLFNN()

Attributes:
    None

Methods:
    __init__() - Constructor

Todo:
    * Add function annotations.
    * Add variable annotations.
"""


__all__ = []
__version__ = '0.0'
__author__ = 'Eric Winter (ewinter@stsci.edu)'


from nnde.neuralnetwork import NeuralNetwork


class SLFFNN(NeuralNetwork):
    """Base class for all single-layer feed-forward neural network objects"""

    def __init__(self):
        """Constructor for SLFNN objects"""
        super().__init__()


if __name__ == '__main__':
    net = SLFFNN()
    print(net)
