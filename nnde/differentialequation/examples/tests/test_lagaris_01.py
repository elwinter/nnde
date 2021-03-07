import numpy as np

from nnde.differentialequation.examples.lagaris_01 import (
    G, dG_dY, dG_ddYdx, Ya, dYa_dx, ic
)


if __name__ == '__main__':
    assert G(0, 0, 0) == 0
    assert dG_dY(0, 0, 0) == 1
    assert dG_ddYdx(0, 0, 0) == 1
    assert np.isclose(Ya(0), ic)
    assert np.isclose(dYa_dx(0), -1)
