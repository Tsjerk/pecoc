"""
cello.py
========

2D ribbon-based glyphs for density and distribution visualization.

Provides a hierarchy of ribbon-shaped drawing primitives, intended as the
primary 2D rendering path for Pecoc feathers.  All classes share a common
transform algebra (+, -, *, @) that operates on vertex coordinates while
preserving color data.

Classes
-------
Ribbon
    Base class.  A general two-edge ribbon defined by two rows of (x, y)
    vertices and per-edge colors.  The two edges may be fully independent.

Cello
    Symmetric ribbon: top and bottom edges are mirror images around a
    central spine (y and -y).  Named for its characteristic silhouette.
    The natural glyph for a KDE density: shaft along x, half-width from
    the density value.

Ridge
    Asymmetric ribbon with one edge pinned to zero.  Useful for drawing
    one-sided distributions or stacked area charts.  The invert flag
    flips the filled side.

Typical usage
-------------
    feather = pecoc[0]
    glyph = Cello(feather.shaft, feather.barbs, c=feather.colors)
    x0, x1, c = glyph.mesh   # pass to pcolormesh or equivalent
"""

import numpy as np


class Glyph2D:
    pass


class Ribbon(Glyph2D):
    """
    A two-edge ribbon primitive for 2D density visualization.

    The ribbon is defined by two parallel sequences of (x, y) points
    (the top and bottom edges) and per-edge color arrays.  All geometric
    transforms act on the vertex array only; colors are carried along
    unchanged.

    Parameters
    ----------
    v : array-like, shape (2, n, 2)
        Vertex array.  ``v[0]`` is the first edge, ``v[1]`` the second.
        Each row holds ``n`` points of the form ``(x, y)``.
    y0 : array-like, shape (n,), optional
        y-coordinates of the first edge.  If provided together with ``y1``
        and a 1D ``v`` (interpreted as x-coordinates), the vertex array is
        constructed from ``(v, y0)`` and ``(v, y1)``.
    y1 : array-like, shape (n,), optional
        y-coordinates of the second edge.  See ``y0``.
    c0 : array-like, shape (n, 3) or (n, 4), optional
        Colors for the first edge.  Defaults to black if not supplied.
    c1 : array-like, shape (n, 3) or (n, 4), optional
        Colors for the second edge.  Defaults to ``c0`` if not supplied.

    Attributes
    ----------
    v : ndarray, shape (2, n, 2)
        Vertex array.
    c : ndarray, shape (2, n, n_channels)
        Color array, one row per edge.
    """

    def __init__(self, v, y0=None, y1=None, c0=None, c1=None):
        if y0 is None:
            self.v = np.asarray(v)
        else:
            x = np.asarray(v)
            self.v = np.asarray(((x, y0), (x, y1))).transpose((0, 2, 1))

        n = self.v.shape[1]
        if c0 is None:
            c0 = np.zeros((n, 3))
        if c1 is None:
            c1 = c0

        self.c = np.asarray((c0, c1))

    # ---------- derived geometry ----------

    @property
    def mesh(self):
        """
        Decompose the ribbon into x, y, and color arrays for pcolormesh.

        Returns
        -------
        x : ndarray, shape (2, n)
            x-coordinates of both edges.
        y : ndarray, shape (2, n)
            y-coordinates of both edges.
        c : ndarray, shape (2, n, n_channels)
            Colors of both edges.
        """
        return self.v[:, :, 0], self.v[:, :, 1], self.c

    @property
    def base(self):
        """
        Full vertex array, shape (2, n, 2).

        Returns
        -------
        ndarray
        """
        return self.v

    @property
    def outline(self):
        """
        Closed outline of the ribbon as flat x and y arrays.

        Traces the first edge forward and the second edge in reverse,
        forming a closed polygon suitable for ``plt.fill`` or
        ``plt.plot``.

        Returns
        -------
        x : ndarray, shape (2n,)
        y : ndarray, shape (2n,)
        """
        xb = np.concatenate([self.v[0, :, 0], self.v[1, ::-1, 0]])
        yb = np.concatenate([self.v[0, :, 1], self.v[1, ::-1, 1]])
        return xb, yb

    # ---------- transformations ----------

    def _new(self, v):
        return Ribbon(v, c0=self.c[0], c1=self.c[1])

    def __add__(self, shift):
        """Translate vertices by ``shift``."""
        return self._new(self.v + shift)

    def __sub__(self, shift):
        """Translate vertices by ``-shift``."""
        return self._new(self.v - shift)

    def __mul__(self, s):
        """Scale vertices by scalar or array ``s``."""
        return self._new(self.v * s)

    def __matmul__(self, A):
        """Apply matrix ``A`` to vertices."""
        return self._new(self.v @ A)


class Cello(Ribbon):
    """
    Symmetric ribbon glyph, mirrored around the x-axis.

    Both edges are ``+y`` and ``-y``, producing a shape reminiscent of
    a cello body.  The natural choice for rendering a KDE density, where
    the shaft runs along x and the half-width represents the density value.

    Parameters
    ----------
    x : array-like, shape (n,) 
        x-coordinates of the spine (1D).
    y : array-like, shape (n,)
        Half-widths.  The ribbon spans from ``-y`` to ``+y``.
    c : array-like, shape (n, 3) or (n, 4)
        Colors applied identically to both edges.

    Attributes
    ----------
    base : ndarray, shape (n, 2)
        Central spine of the ribbon, i.e. the mean of both edges.
    """

    def __init__(self, x, y, c):
            v = np.array(((y, x), (-y, x))).transpose((0, 2, 1))
            super().__init__(v, c0=c, c1=c)

    @property
    def base(self):
        """
        Central spine as an (n, 2) array of (x, y) points.

        Returns
        -------
        ndarray, shape (n, 2)
        """
        return self.v.mean(axis=0)


class Ridge(Ribbon):
    """
    Asymmetric ribbon glyph with one edge pinned to zero.

    One edge follows ``y``; the other is held at zero, producing a
    one-sided filled shape.  Useful for stacked or directional density
    plots.

    Parameters
    ----------
    x : array-like, shape (n,) or array-like, shape (2, n, 2)
        x-coordinates (1D), or a pre-built vertex array (2D).
    y : array-like, shape (n,), optional
        Height of the free edge.  If None, ``x`` is a full vertex array.
    c : array-like, shape (n, 3) or (n, 4), optional
        Colors applied identically to both edges.
    invert : bool, optional
        If False (default), the ribbon fills from 0 up to ``y``.
        If True, it fills from ``-y`` down to 0.
    """

    def __init__(self, x, y=None, c=None, invert=False):
        if invert:
            super().__init__(x, np.zeros_like(x), -np.asarray(y), c, c)
        else:
            super().__init__(x, y, np.zeros_like(x), c, c)

