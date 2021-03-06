"""Base class for single-layer feed-forward neural networks

This module provides the base functionality for all single-layer feed-forward
neural network objects used in the nnode software.

This class is currently an abstract base class. It must be subclassed
to be useful.

Attributes:
    None

Methods:
    None
"""


from nnde.neuralnetwork.neuralnetwork import NeuralNetwork


class SLFFNN(NeuralNetwork):
    """Base class for all single-layer feed-forward neural network objects"""
    pass
