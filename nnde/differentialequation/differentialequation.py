"""Abstract base class for differential equations.

Deprecation Warning
-------------------
None

Extended Summary
----------------
This module defines the DifferentialEquation, which is an abstract
class defining methods by for all differential equation classes used
in the nnde package.

Attributes
----------
DifferentialEquation : The base class for differential equations

Authors
-------
Eric Winter (eric.winter62@gmail.com)
"""


from nnde.exceptions.nndeexception import NNDEException


class DifferentialEquation:
    """Abstract base class for differential equation objects

    Attributes
    ----------
    G() - Function for differential equation, in the form G() = 0,
    where G() is a function of the independent variables x, the solution
    Y(x), and its derivatives.
    """

    def __init__(self, *args, **kwargs):
        """Initialize a new DifferentialEquation object.

        This abstract base class defines the required methods for all
        differential equation classes in the nnde package.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Raises
        ------
        NNDEException : If called
            This is an abstract method.
        """
        raise NNDEException

    def G(self, *args, **kargs):
        """The differential equation to solve as G() == 0.

        This method defines the interface for a generic differential
        equation.

        Parameters
        ----------
        *args : Arbitrary positional parameters.
            Should be overridden by equaiton-specific subclass.

        Returns
        -------
        g : float
            Value of differential equation, ideally 0.

        Raises
        ------
        NNDEException : If called
            This is an abstract method.

        Notes
        -----
        In general, this method will take as arguments, at a minimum:

        x : arraylike of float
            Values of independent variables of the differential equation.
        Y : float
            Value of dependent variable of the differential equation.
        dY_dx : arraylike of float
            Values for first derivatives of Y wrt each independent
            variable, in same order as Y.
        d2Y_dx2 : arraylike of float
            Values for second partial derivatives of Y wrt each pair of
            independent variables, in same order as 
        """
        raise NNDEException
