"""Base class for trial functions

This module provides the base functionality for all trial function objects
used in the nnde software.

This class is currently an abstract base class. It must be subclassed
to be useful.

Example:
    Create an empty TrialFunction object.
        tf = TrialFunction()

Attributes:
    None

Methods:
    Yt(x, N) - Evaluate the trial function given coordinate vector x and
    scalar network output N
"""


class TrialFunction():
    """Trial function base class"""

    def Yt(self, x, N):
        """Trial function"""
        raise Exception
