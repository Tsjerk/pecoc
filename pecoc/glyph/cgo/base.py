"""
cgo_base.py
===========
Core machinery for renderer-agnostic geometric object construction.

This module provides three layers of abstraction:

CGO -- a NumPy-backed container for a flat array of drawing commands.
    Tracks which positions in the array hold vertex coordinates and which
    hold surface normals, enabling bulk geometric transforms via standard
    arithmetic operators:
        cgo + vector    -- translate vertices
        cgo * scalar    -- scale vertices
        cgo @ matrix    -- rotate vertices and normals
        cgo1 + cgo2     -- concatenate two objects into one

Field descriptors -- lightweight objects that describe the role of each
    field in a primitive's data layout:
        Value(name)     -- plain data (radii, colours, flags); never transformed
        Vertex(name)    -- 3-D coordinates; transformed by +, -, *, @
        Normal(name)    -- 3-D directions; transformed by @ only
    Multiplying a descriptor scales its size: 3*Value('rgb') occupies 3 floats.

Primitive -- a factory function that takes an ordered sequence of constants
    and descriptors mirroring the flat array layout of one primitive, and
    returns a CGO subclass whose constructor accepts the settable fields as
    positional arguments.  The returned class supports batched instantiation:
    passing (N, 3) arrays constructs N primitives in one call.

This module has no dependency on any specific renderer.  The CGO array layout
follows PyMOL conventions by default, but the operator algebra and factory
machinery are general.  Renderer-specific loading (e.g. cmd.load_cgo) is
handled by importing PyMOL when available and falling back gracefully when not.

Usage contexts
--------------
* Imported as a module from cgo_primitives or user code
* Sourced inside a running PyMOL session (``__name__ == "pymol"``)
* Run directly as a script (``__main__``)
"""

# Code was human generated. Documentation was written by Claude.ai.

import numpy as np

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


# ---------------------------------------------------------------------------
# CGO – base container for compiled graphical objects
# ---------------------------------------------------------------------------

class Glyph3D:
    def draw(self, ax=None, **plotopts):
        from .loader import pltloader
        return cgoloader(self, **plotopts)


class CGO(Glyph3D):
    """
    A NumPy-backed container for a PyMOL Compiled Graphical Object (CGO).

    A CGO is ultimately a flat 1-D array of floats that PyMOL interprets as a
    sequence of drawing commands.  This class keeps track of which positions
    in that array hold vertex coordinates (``cindices``) and which hold surface
    normals (``nindices``), enabling efficient bulk transforms via NumPy.

    Parameters
    ----------
    definition : array-like
        Flat array of CGO floats (opcodes interleaved with data).
    objects : array-like of int
        Start index of each primitive within *definition*.
    cindices : array-like of int
        Indices into *definition* that contain vertex coordinate values.
    nindices : array-like of int
        Indices into *definition* that contain surface normal values.

    Notes
    -----
    Arithmetic operators (+, -, *) act only on vertex coordinates.
    The matrix-multiplication operator (@) additionally rotates normals,
    making it suitable for applying rotation matrices.
    """

    def __init__(self, definition, objects, cindices, nindices):
        self._definition = np.asarray(definition)
        self.objects  = objects
        self.cindices = cindices.astype(int)
        self.nindices = nindices.astype(int)

    # ------------------------------------------------------------------
    # PyMOL I/O
    # ------------------------------------------------------------------

    def draw(self, loader=cgo_loader, **kwargs):
        """
        Send this CGO to PyMOL for rendering.

        Parameters
        ----------
        loader : callable, optional
            Function used to load the CGO array.  Defaults to
            ``cmd.load_cgo`` from the active PyMOL session.
        **kwargs
            Additional keyword arguments forwarded to *loader*
            (e.g. ``name`` for the PyMOL object name).
        """
        loader(self.definition, **kwargs)

    def write(self, filename):
        """
        Write the CGO definition to a plain-text file.

        Each primitive is written as a space-separated line of floats,
        making the output human-readable and easy to diff.

        Parameters
        ----------
        filename : str
            Path to the output file.
        """
        with open(filename, 'w') as out:
            # Split the flat array back into per-primitive chunks and write
            # each one on its own line.
            for part in np.split(self.definition, self.objects):
                print(*part, file=out)

    # ------------------------------------------------------------------
    # Properties: live views into the flat definition array
    # ------------------------------------------------------------------

    @property
    def definition(self):
        """
        The fully assembled CGO float array, ready for PyMOL.

        Vertex coordinates and normals are written into the correct positions
        of the underlying array before it is returned, so any in-place
        modifications to ``vertices`` or ``normals`` are reflected here.

        Returns
        -------
        numpy.ndarray
            1-D array of CGO floats.
        """
        self._definition[self.cindices] = self.vertices.flatten()
        self._definition[self.nindices] = self.normals.flatten()
        return self._definition

    @property
    def vertices(self):
        """
        Vertex coordinates as an (N, 3) array.

        Returns
        -------
        numpy.ndarray, shape (N, 3)
        """
        return self._definition[self.cindices].reshape((-1, 3))

    @property
    def normals(self):
        """
        Surface normals as an (N, 3) array.

        Returns
        -------
        numpy.ndarray, shape (N, 3)
        """
        return self._definition[self.nindices].reshape((-1, 3))

    def __len__(self):
        """Return the total number of floats in the CGO definition array."""
        return len(self._definition)

    # ------------------------------------------------------------------
    # Arithmetic helpers
    # ------------------------------------------------------------------

    def _op(self, other, fun):
        """
        Apply a binary operation to vertex coordinates (and optionally normals).

        A new CGO is returned; the original is not modified.

        Parameters
        ----------
        other : scalar or array-like
            Right-hand operand (e.g. a translation vector or scale factor).
        fun : str
            Name of the dunder method to call on the vertex/normal arrays
            (e.g. ``'__add__'``, ``'__mul__'``, ``'__matmul__'``).

        Returns
        -------
        CGO
            Transformed copy of this object.

        Notes
        -----
        ``__matmul__`` is the only operation that also transforms normals.
        For pure rotations represented as (3, 3) matrices this keeps the
        normals geometrically consistent with the rotated vertices.
        All other operators (translation, uniform scaling) leave normals
        untouched because translating or scaling direction vectors is
        generally incorrect.
        """
        definition = self._definition.copy()
        # Apply the operation to vertex coordinates.
        definition[self.cindices] = getattr(self.vertices, fun)(other)
        # For matrix multiplication (rotation), also transform the normals.
        if fun in ['__matmul__']:
            definition[self.nindices] = getattr(self.normals, fun)(other)
        return CGO(definition, self.objects, self.cindices, self.nindices)

    def __add__(self, other):
        """
        Translate vertices by *other*, or concatenate two CGO objects.

        Parameters
        ----------
        other : CGO or scalar or array-like
            * If *other* is a :class:`CGO`, the two objects are concatenated
              into a single CGO containing all primitives from both.
            * Otherwise, *other* is treated as a translation vector (or
              scalar) and added element-wise to every vertex coordinate.

        Returns
        -------
        CGO
        """
        if not isinstance(other, CGO):
            return self._op(other, '__add__')

        # Concatenate two CGOs: adjust all index arrays of *other* so they
        # point into the combined definition array.
        definition = np.concatenate((self.definition, other.definition))
        objects    = np.concatenate((self.objects,  other.objects  + len(self)))
        cindices   = np.concatenate((self.cindices, other.cindices + len(self)))
        nindices   = np.concatenate((self.nindices, other.nindices + len(self)))
        return CGO(definition, objects, cindices, nindices)

    def __sub__(self, other):
        """
        Translate vertices by *-other* (element-wise subtraction).

        Parameters
        ----------
        other : scalar or array-like

        Returns
        -------
        CGO
        """
        return self._op(other, '__sub__')

    def __mul__(self, other):
        """
        Scale vertices by *other* (element-wise multiplication).

        Parameters
        ----------
        other : scalar or array-like

        Returns
        -------
        CGO
        """
        return self._op(other, '__mul__')

    def __matmul__(self, other):
        """
        Rotate this CGO by a (3, 3) rotation matrix.

        Both vertex coordinates *and* surface normals are transformed so that
        shading remains correct after rotation.

        Parameters
        ----------
        other : numpy.ndarray, shape (3, 3)
            Rotation matrix.

        Returns
        -------
        CGO
        """
        return self._op(other, '__matmul__')

    def __str__(self):
        """
        Return a human-readable, per-primitive representation.

        Each primitive occupies one line, with its float values
        space-separated.
        """
        return '\n'.join(str(p) for p in np.split(self._definition, self.objects[1:]))


# ---------------------------------------------------------------------------
# Field descriptors
# ---------------------------------------------------------------------------

class Value:
    """
    Descriptor for one or more plain data floats in a CGO primitive.

    Plain values are never modified by geometric transforms (translation,
    rotation, scaling).  Use for opcodes, radii, cap flags, colour triples
    that are not spatially meaningful, etc.

    Parameters
    ----------
    par : str
        Field name, used as the keyword in the primitive's field map.

    Attributes
    ----------
    size : int
        Number of consecutive floats this field occupies (default 1).
        Scaled by the ``__rmul__`` operator, e.g. ``3 * Value('rgb')``.
    """

    _size = 1

    def __init__(self, par):
        self.par  = par
        self.size = self._size

    def __rmul__(self, other):
        """Return a copy of this descriptor with size scaled by *other*."""
        x      = self.__class__(self.par)
        x.size = other * self.size
        return x


class Vertex(Value):
    """
    Descriptor for a 3-D vertex coordinate in a CGO primitive.

    Vertex fields are transformed by all arithmetic operators:
    translation (+/-), scaling (*), and rotation (@).

    Default size is 3 (x, y, z).  Multiplying scales the size, which is
    useful for storing multiple vertices in one named field, though it is
    usually clearer to declare them separately.
    """
    _size = 3


class Normal(Value):
    """
    Descriptor for a 3-D surface normal (or orientation vector) in a CGO
    primitive.

    Normal fields are transformed only by rotation (@), not by translation
    or scaling, because direction vectors are not affected by those operations.

    Default size is 3.  Multiplying scales the size:
        ``3 * Normal('orientation')``  produces a size-9 field suitable for
        storing a full 3x3 orientation matrix (e.g. for Ellipsoid).
    """
    _size = 3


# ---------------------------------------------------------------------------
# Primitive – a factory that manufactures typed CGO subclasses
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Primitive factory
# ---------------------------------------------------------------------------

def Primitive(*spec):
    """
    Factory that creates a CGO subclass for a specific primitive type.

    Parameters
    ----------
    *spec : float, int, Value, Vertex, or Normal
        Ordered sequence of field descriptors mirroring the flat CGO array
        layout for one primitive.  Each element is one of:

        float / int
            A constant value baked into the template (e.g. a CGO opcode or
            a fixed flag).  Not exposed as a settable constructor argument.
        Value(name) / n * Value(name)
            One or more plain data floats.  Passed as constructor arguments;
            never modified by geometric transforms.
        Vertex(name) / n * Vertex(name)
            Three (or 3n) coordinate floats.  Passed as constructor
            arguments; transformed by translation, scaling, and rotation.
        Normal(name) / n * Normal(name)
            Three (or 3n) direction floats.  Passed as constructor
            arguments; transformed by rotation only.

    Returns
    -------
    type
        A new class (subclass of :class:`CGO`) whose constructor accepts one
        positional argument per non-constant field (in declaration order).
        The class supports batched instantiation: if any argument has a
        leading batch dimension, all arguments are broadcast to that size.

    Examples
    --------
    >>> V, X, N = Value, Vertex, Normal
    >>> Sphere   = Primitive(7.0, X('center'), V('radius'))
    >>> Cylinder = Primitive(9.0, X('start'), X('end'), V('radius'),
    ...                      3*V('rgb1'), 3*V('rgb2'))
    >>> Ellipsoid = Primitive(18.0, X('center'), V('radius'),
    ...                       3*N('orientation'))

    Notes
    -----
    The first element of *spec* is conventionally a float opcode.  The
    generated class name reflects this: ``Primitive_9`` for opcode 9.0.

    Constant floats/ints in *spec* occupy positions in the template but are
    not exposed as constructor arguments, so the constructor argument count
    equals the number of descriptor objects in *spec*.
    """

    # ------------------------------------------------------------------
    # Build the per-primitive template array.
    # Constants contribute 1 slot; descriptors contribute descriptor.size.
    # ------------------------------------------------------------------
    template_size = sum(
        1 if isinstance(v, (int, float)) else v.size
        for v in spec
    )
    template = np.zeros(template_size)

    fields  = {}  # name → slice(start, end) for each settable field
    coords  = []  # flat offsets of Vertex positions within one template
    normals = []  # flat offsets of Normal positions within one template

    position = 0
    for thing in spec:
        if isinstance(thing, (int, float)):
            # Bake constant directly into the template (opcode, flag, etc.)
            template[position] = thing
            position += 1
            continue

        newpos = position + thing.size

        # Always store as a slice (even size-1 fields) for uniform indexing
        # in __init__ when filling out[:, slc] = value.
        fields[thing.par] = slice(position, newpos)

        if isinstance(thing, Vertex):
            coords.extend(range(position, position + thing.size))
        elif isinstance(thing, Normal):
            normals.extend(range(position, position + thing.size))
        # Plain Value: no entry in coords or normals.

        position = newpos

    # Convert to NumPy arrays now so that broadcasting in __init__ works
    # correctly.  Empty arrays (for primitives with no Vertex/Normal fields)
    # are handled explicitly below.
    _template_data = template
    _fields_data   = fields
    _coords_data   = np.array(coords,  dtype=int)
    _normals_data  = np.array(normals, dtype=int)

    # ------------------------------------------------------------------
    # Dynamically defined class for this primitive type
    # ------------------------------------------------------------------

    class _Primitive(CGO):
        """
        A batch of CGO primitives of a single type.

        Instantiated by the enclosing :func:`Primitive` factory; do not
        construct directly.

        Parameters
        ----------
        *args
            One positional argument per non-constant field declared in the
            spec, in declaration order.  Each argument may be:

            * a scalar or 1-D array of length 3 (single primitive), or
            * a 2-D array of shape (n, field_size) (batch of n primitives).

            The batch size *n* is inferred from the first argument whose
            array representation has ``ndim > 1``.  All other arguments are
            broadcast to that size automatically by NumPy.
        name : str, optional
            Label used when loading the CGO into PyMOL.
        """

        _template = _template_data
        _fields   = _fields_data
        _coords   = _coords_data
        _normals  = _normals_data

        def __init__(self, *args, name=None):
            self.name = name

            # ----------------------------------------------------------
            # Determine the batch size n from the first argument that has
            # more than one dimension when viewed as a NumPy array.
            # Scalars and 1-D arrays of length 3 correspond to n=1.
            # ----------------------------------------------------------
            n = 1
            for v in args:
                arr = np.asarray(v)
                if arr.ndim > 1:
                    n = arr.shape[0]
                    break

            # Broadcast the 1-D template to (n, template_size) so each
            # primitive in the batch gets its own independent copy.
            out = np.broadcast_to(self._template, (n, self._template.size)).copy()

            # Fill each settable field; slices are always slice objects so
            # out[:, slc] = value works uniformly for any field width.
            for (field, slc), value in zip(self._fields.items(), args):
                out[:, slc] = value

            # Start offset of each primitive in the ravelled output array.
            objects = len(self._template) * np.arange(n)

            # Absolute index arrays for coordinates and normals.
            # Guard against empty _coords/_normals (e.g. Alpha, Linewidth).
            if len(self._coords):
                cindices = (objects[:, None] + self._coords).ravel()
            else:
                cindices = np.array([], dtype=int)

            if len(self._normals):
                nindices = (objects[:, None] + self._normals).ravel()
            else:
                nindices = np.array([], dtype=int)

            super().__init__(out.ravel(), objects, cindices, nindices)

    _Primitive.__name__ = f"Primitive_{int(template[0])}"
    return _Primitive

