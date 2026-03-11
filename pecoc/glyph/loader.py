
from .cello import Glyph2D
from .cgo.base import Glyph3D

# ---------------------------------------------------------------------------
# Detect execution context so the module can behave appropriately in each.
# ---------------------------------------------------------------------------
_as_program      = (__name__ == "__main__")
_as_pymol_script = (__name__ == "pymol")
_as_module       = not (_as_program or _as_pymol_script)
_as_pymol_module = False

try:
    import pymol
    from pymol import cmd, cgo
    import ast

    if _as_module:
        _as_pymol_module = True
        print("Importing CGO drawing module.")
    else:
        print("Running CGO drawing module.")

    # Use PyMOL's native CGO loader when available.
    cgo_loader = cmd.load_cgo

except ImportError:
    # Running outside PyMOL (e.g. unit tests, offline analysis).
    cmd = None
    _as_pymol_script = False
    cgo_loader = None

    
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

    
def pltloader(glyph, ax=None, **plotopts):
    if plt is None:
        raise ImportError('matplotlib is required for 2D glyph rendering')
    if ax is not None:
        plt.sca(ax)
    x, y, c = glyph.mesh
    mesh = plt.pcolormesh(x, y, c, shading='gouraud')
    l0, = plt.plot(x[0], y[0], **plotopts)
    l1, = plt.plot(x[1], y[1], **plotopts)
    return [mesh, l0, l1]

    
def cgoloader(glyph, **plotopts):
    name = plotopts.pop('name') or 'pecoc_cgo'
    cgo_loader(glyph.definition, name, **plotopts)

    
def glyphloader(glyph, **plotopts):
    if isinstance(glyph, Glyph2D):
        return pltloader(glyph, **plotopts)
    elif isinstance(glyph, Glyph3D):
        return cgoloader(glyph, **plotopts)
