"""Base class for neural networks

This module provides the base functionality for all neural network objects
used in the nnode software.

This class is currently an abstract base class. It must be subclassed
to be useful.

Attributes:
    None

Methods:
    __init__() - Constructor
    train() - Stub for training methods for subclasses
    run() - Stub for run methods for subclasses
"""


from nnde.exceptions.nndeexception import NNDEException


class NeuralNetwork:
    """Abstract base class for all neural network objects"""

    def __init__(self):
        """(abstract) Constructor for the neural network object."""
        raise NNDEException

    def train(self):
        """(abstract) Train the neural network."""
        raise NNDEException

    def run(self):
        """(abstract) Run the neural network using current parameters."""
        raise NNDEException
