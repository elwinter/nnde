"""
PDE2DIFF - Base class for diffusion equations

This module provides the base functionality for all 1-, 2-, and 3-D diffusion
equation objects used in the nnode software.

This class is currently an abstract base class. It must be subclassed
to be useful.

Example:
    Create an empty PDE2DIFF object.
        pde2diff = PDE2DIFF()
    Create an PDE2DIFF object from a Python module.
        pde2diff = PDE2DIFF(modname)

The solution is assumed to be a function of m independent variables. In the
methods below, x is a vector of independent variables, and delY is the
Jacobian of the solution wrt the independent variables.

Attributes:
    None

Methods:
    None

Todo:
    * Add function annotations.
    * Add variable annotations.
"""


__all__ = []
__version__ = '0.0'
__author__ = 'Eric Winter (ewinter@stsci.edu)'


from importlib import import_module

from nnde.pde2 import PDE2


class PDE2DIFF(PDE2):
    """Base class for all diffusion equation objects"""

    def __init__(self, modname=None):
        super().__init__()
        self.name = None
        self.G = None
        self.bc = None
        self.delbc = None
        self.del2bc = None
        self.dG_dY = None
        self.dG_ddelY = None
        self.dG_ddel2Y = None
        self.A = None
        self.delA = None
        self.del2A = None
        self.Ya = None
        self.delYa = None
        self.del2Ya = None
        if modname:
            pdemod = import_module(modname)
            self.name = modname
            assert pdemod.G
            self.G = pdemod.G
            assert pdemod.bc
            self.bc = pdemod.bc
            assert pdemod.delbc
            self.delbc = pdemod.delbc
            assert pdemod.del2bc
            self.del2bc = pdemod.del2bc
            assert pdemod.dG_dY
            self.dG_dY = pdemod.dG_dY
            assert pdemod.dG_ddelY
            self.dG_ddelY = pdemod.dG_ddelY
            assert pdemod.dG_ddel2Y
            self.dG_ddel2Y = pdemod.dG_ddel2Y
            if hasattr(pdemod, 'A'):
                self.A = pdemod.A
            if hasattr(pdemod, 'delA'):
                self.delA = pdemod.delA
            if hasattr(pdemod, 'del2A'):
                self.del2A = pdemod.del2A
            if hasattr(pdemod, 'Ya'):
                self.Ya = pdemod.Ya
            if hasattr(pdemod, 'delYa'):
                self.delYa = pdemod.delYa
            if hasattr(pdemod, 'del2Ya'):
                self.del2Ya = pdemod.del2Ya


if __name__ == '__main__':
    pde2diff = PDE2DIFF('eq.diff1d_zero')
    print(pde2diff)
    print(pde2diff.A)
