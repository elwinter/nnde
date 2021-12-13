from importlib import import_module

from nnde.differentialequation.pde.pde2 import PDE2


class PDE2BVP(PDE2):

    def __init__(self, modname):
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
