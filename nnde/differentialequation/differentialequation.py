"""Abstract base class for differential equations.

Deprecation Warning
-------------------
None

Extended Summary
----------------
This module defines the DifferentialEquation, which is an abstract
class defining methods by for all differential equation classes used
in the nnde package.

This class is an abstract base class. It must be subclassed to be
useful. All methods in this class will raise NNDEException if called.

Classes
-------
DifferentialEquation

Functions
---------
None

See Also
--------

Notes
-----

References
----------

Examples
--------

Authors
-------
Eric Winter (eric.winter62@gmail.com)
"""


from nnde.exceptions.nndeexception import NNDEException


class DifferentialEquation:
    """Abstract base class for differential equation objects.

    Deprecation Warning
    -------------------
    None

    Extended Summary
    ----------------
    This abstract base class defines the required methods for all
    differential equation classes in the nnde package.

    Parameters
    ----------

    Attributes
    ----------

    Returns
    -------

    Yields
    ------

    Receives
    --------

    Other Parameters
    ----------------
    Infrequently-used parameters go here.

    Raises
    ------

    Warns
    -----

    Warnings
    --------

    See Also
    --------

    Notes
    -----

    References
    ----------

    Examples
    --------
    """

    def __init__(self, *args, **kwargs):
        """Constructor for DifferentialEquation objects"""
        raise NNDEException

    def G(self, *args, **kwargs) -> float:
        """Differential equation implementation"""
        raise NNDEException
