"""
sigma - Python module to implement the sigmoid function and derivatives

This module provides the sigma transfer function and derivatives of orders
1-4.  This function is frequently used as a transfer function in neural
network nodes.  Two versions of the functions are provided.  The first
version (functions named sigma, dsigma_dz, ...) are traditional functions of
the input z, and the functions are written in an unoptimized form for
clarity.  The second version (functions named s, s1, ...) are written in an
optimized form (except for s(), which is identical to sigma()).  In this
optimized form, the functions are rewritten to take advantage of algebraic
rearrangement to minimize the chance of overflow and range errors.  These
versions take s itself as an argument, rather than z.  This approach is
described in Menon et al (1996).

The sigma function is defined as:

sigma(z) = 1/(1 + exp(-z))

The derivatives are found by differentiation with respect to z.

Examples:

    Calculate the value of sigma for z=1.75.
        s = sigma.sigma(1.75)
    or
        s = sigma.s(1.75)

    Calculate the 1st through 4th derivatives of sigma at z=1.75
        s1 = sigma.dsigma_dz(1.75)
        s2 = sigma.d2sigma_dz2(1.75)
        s3 = sigma.d3sigma_dz3(1.75)
        s4 = sigma.d4sigma_dz4(1.75)
    or
        s = sigma.s(1.75)
        s1 = sigma.s1(s)
        s2 = sigma.s2(s)
        s3 = sigma.s3(s)
        s4 = sigma.s4(s)

Todo:
    * Adjust docstrings to PEP 257 conventions.
    * Add function annotations.
    * Add variable annotations.
    * Add code to auto-generate higher-order derivatives.

References:

https://en.wikipedia.org/wiki/Sigmoid_function

Menon, A., Mehrotra, K., Mohan, C., and Ranka, S., Neural Networks,
Volume 9, Number 5, pp. 819-835 (1996)
"""

__all__ = ['sigma', 'dsigma_dz', 'd2sigma_dz2', 'd3sigma_dz3', 'd4sigma_dz4',
           's', 's1', 's2', 's3', 's4']
__version__ = '0.0'
__author__ = 'Eric Winter (ewinter@stsci.edu)'


from math import exp
from numpy import isclose


# Standard versions

def sigma(z):
    """Compute the sigma function of z."""
    return 1/(1 + exp(-z))


def dsigma_dz(z):
    """Compute the first derivative of the sigma function of z."""
    return exp(-z)/(1 + exp(-z))**2


def d2sigma_dz2(z):
    """Compute the second derivative of the sigma function of z."""
    return 2*exp(-2*z)/(1 + exp(-z))**3 - exp(-z)/(1 + exp(-z))**2


def d3sigma_dz3(z):
    """Compute the third derivative of the sigma function of z."""
    return (6*exp(-3*z)/(1 + exp(-z))**4 - 6 * exp(-2*z)/(1 + exp(-z))**3
            + exp(-z)/(1 + exp(-z))**2)


def d4sigma_dz4(z):
    """Compute the fourth derivative of the sigma function of z."""
    return (24*exp(-4*z)/(1 + exp(-z))**5 - 36*exp(-3*z)/(1 + exp(-z))**4
            + 14*exp(-2*z)/(1 + exp(-z))**3 - exp(-z)/(1 + exp(-z))**2)


# Optimized versions

def s(z):
    """Compute the sigma function of z."""
    return 1/(1 + exp(-z))


def s1(s):
    """Compute the first derivative of the sigma function of z where
    s = s(z)."""
    return s - s**2


def s2(s):
    """Compute the second derivative of the sigma function of z where
    s = s(z)."""
    return 2*s**3 - 3*s**2 + s


def s3(s):
    """Compute the third derivative of the sigma function of z where
    s = s(z)."""
    return -6*s**4 + 12*s**3 - 7*s**2 + s


def s4(s):
    """Compute the fourth derivative of the sigma function of z where
    s = s(z)."""
    return 24*s**5 - 60*s**4 + 50*s**3 - 15*s**2 + s


if __name__ == '__main__':

    # Test value of z
    z_test = 1

    # Reference values for tests
    s_ref = 0.731058578630005
    s1_ref = 0.196611933241482
    s2_ref = -0.0908577476729484
    s3_ref = -0.0353255805162356
    s4_ref = 0.123506861366393

    print('Testing standard versions.')
    s_ = sigma(z_test)
    assert isclose(s_, s_ref)
    assert isclose(dsigma_dz(z_test), s1_ref)
    assert isclose(d2sigma_dz2(z_test), s2_ref)
    assert isclose(d3sigma_dz3(z_test), s3_ref)
    assert isclose(d4sigma_dz4(z_test), s4_ref)

    print('Testing optimized versions.')
    s_ = s(z_test)
    assert isclose(s_, s_ref)
    assert isclose(s1(s_), s1_ref)
    assert isclose(s2(s_), s2_ref)
    assert isclose(s3(s_), s3_ref)
    assert isclose(s4(s_), s4_ref)
