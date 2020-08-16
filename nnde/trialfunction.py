"""
TrialFunction - Base class for trial functions

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

Todo:
    * Add function annotations.
    * Add variable annotations.
"""


__all__ = []
__version__ = '0.0'
__author__ = 'Eric Winter (ewinter@stsci.edu)'


class TrialFunction():
    """Trial function base class"""

    def __init__(self):
        """Constructor"""
        pass

    def Yt(self, x, N):
        """Trial function"""
        return None


if __name__ == '__main__':
    tf = TrialFunction()
    print(tf)
    assert tf.Yt(None, None) is None
