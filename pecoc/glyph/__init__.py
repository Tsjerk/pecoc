"""
pecoc.glyph
===========

2D and 3D glyph primitives for density visualization.

2D glyphs (matplotlib) are in the cello module.
3D glyphs (CGO/PyMOL) are in the cgo subpackage.
"""

from .loader import (
    glyphloader
)

from .cello import (
    Glyph2D,
    Ribbon,
    Cello,
    Ridge,
)

from .cgo.base import (
    Glyph3D,
    CGO,
    Value,
    Vertex,
    Normal,
    Primitive,
)

from .cgo.primitives import (
    V, X, N,
    Cylinder,
    Cone,
    Sphere,
    ColoredSphere,
    Sausage,
    Cylinder2,
    Ellipsoid,
    Triangle,
    Alpha,
    Linewidth,
    Widthscale,
    Arrow,
    Triangles,
    TriangleStrip,
    Tube,
    FramedTube,
    Mesh,
    read_stl,
)
