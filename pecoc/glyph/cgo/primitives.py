"""
primitives.py
=================

Concrete CGO primitive types and compound geometric constructors.

Imports the core machinery (CGO, Primitive, Value, Vertex, Normal) from
cgo_base and uses it to define all drawable objects.  Nothing in this module
is renderer-specific; the CGO array layout follows PyMOL opcode conventions
but the objects themselves are plain NumPy-backed data.

Fixed primitives (classes produced by the Primitive factory)
------------------------------------------------------------
Each is a CGO subclass whose constructor accepts positional arguments
matching the field declarations below, with batched (N-at-a-time) support:

    Cylinder(start, end, radius, rgb1, rgb2)
    Cone(start, end, radius1, radius2, rgb1, rgb2, cap1, cap2)
    Sphere(center, radius)
    ColoredSphere(rgb, center, radius)
    Sausage(start, end, radius, rgb1, rgb2)
    Cylinder2(start, end, radius, cap1, cap2, rgb1, rgb2)
    Ellipsoid(center, radius, orientation)
    Triangle(v1, v2, v3, n1, n2, n3, c1, c2, c3)
    Alpha(value)
    Linewidth(width)
    Widthscale(scale)

Variable-length primitives (functions returning a CGO)
------------------------------------------------------
These cannot be expressed as fixed-layout Primitive subclasses because
their array size depends on the number of vertices.  They build their
Primitive class dynamically and cache it by size:

    TriangleStrip(vertices, normals, colors)
    Triangles(vertices, normals, colors)

Compound constructors (functions returning a CGO)
-------------------------------------------------
Built from one or more of the above:

    Arrow(start, end, radius, rgb)          -- Cylinder + Cone
    Tube(vertices, radii, colors)           -- Cone or Cylinder2 chain
    Mesh(vertices, faces, normals, colors)  -- Triangles block from index array
    FramedTube(vertices, radii, colors)     -- TriangleStrip tube with Bishop frame

Convenience aliases
-------------------
    V = Value    -- plain data
    X = Vertex   -- coordinates
    N = Normal   -- directions  (note: do not use N as a variable name)
"""


import numpy as np
from .base import CGO, Value, Vertex, Normal, Primitive


# ---------------------------------------------------------------------------
# CGO opcodes
# ---------------------------------------------------------------------------
_BEGIN    = 2.0
_END      = 3.0
_VERTEX   = 4.0
_NORMAL   = 5.0
_COLOR    = 6.0
_SPHERE   = 7.0
_CYLINDER = 9.0
_TRIANGLE = 17.0
_ALPHA    = 25.0
_CONE     = 27.0

# GL primitive modes (passed as the value after BEGIN)
_GL_POINTS         = 0.0
_GL_LINES          = 1.0
_GL_LINE_STRIP     = 2.0
_GL_LINE_LOOP      = 3.0
_GL_TRIANGLES      = 4.0
_GL_TRIANGLE_STRIP = 5.0
_GL_TRIANGLE_FAN   = 6.0


# ---------------------------------------------------------------------------
# Convenience aliases for descriptor classes
# ---------------------------------------------------------------------------

V = Value   # plain data  (radii, colours, flags)
X = Vertex  # coordinates (transformed by +, -, *, @)
N = Normal  # directions  (transformed by @ only)


# ---------------------------------------------------------------------------
# Concrete primitive types
# ---------------------------------------------------------------------------
# Each Primitive() call produces a class whose opcode is baked into position 0
# of the template.  Field order mirrors the PyMOL CGO array layout exactly.
#
# Constructor argument order matches the spec order, skipping constants:
#   Cylinder(start, end, radius, rgb1, rgb2)
#   Cone(start, end, radius1, radius2, rgb1, rgb2, cap1, cap2)
#   etc.
# ---------------------------------------------------------------------------

# Two endpoints, a uniform radius, and per-endpoint colours.
Cylinder = Primitive(
    9.0,
    X('start'), X('end'),
    V('radius'),
    3*V('rgb1'), 3*V('rgb2'),
)

# Two endpoints, independent radii (enabling taper), per-endpoint colours,
# and independent caps at each end (1=capped, 0=open).
Cone = Primitive(
    27.0,
    X('start'), X('end'),
    V('radius1'), V('radius2'),
    3*V('rgb1'), 3*V('rgb2'),
    V('cap1'), V('cap2'),
)

# Sphere with an explicit colour opcode (6.0) preceding the sphere opcode
# (7.0).  Use when the colour must be set as part of the primitive rather
# than relying on PyMOL's current colour state.
ColoredSphere = Primitive(
    6.0, 3*V('rgb'),
    7.0, X('center'), V('radius'),
)

# Simple sphere; colour is taken from PyMOL's current state.
Sphere = Primitive(
    7.0,
    X('center'), V('radius'),
)

# Rounded cylinder ("sausage"): like Cylinder but with hemispherical caps.
Sausage = Primitive(
    14.0,
    X('start'), X('end'),
    V('radius'),
    3*V('rgb1'), 3*V('rgb2'),
)

# Alternative cylinder with explicit cap control (opcode 15).
Cylinder2 = Primitive(
    15.0,
    X('start'), X('end'),
    V('radius'),
    V('cap1'), V('cap2'),
    3*V('rgb1'), 3*V('rgb2'),
)

# Ellipsoid: centre, radius, and a 3x3 orientation matrix stored as 9
# Normal floats so that rotation (@) correctly reorients the ellipsoid.
Ellipsoid = Primitive(
    18.0,
    X('center'), V('radius'),
    3*N('orientation'),
)

# Triangle with per-vertex normals and per-vertex colours.
# The fundamental surface-mesh building block; batch over faces.
# Normals are declared with N() so rotation (@) keeps shading correct.
Triangle = Primitive(
    17.0,
    X('v1'), X('v2'), X('v3'),      # three vertex positions
    N('n1'), N('n2'), N('n3'),      # three surface normals
    3*V('c1'), 3*V('c2'), 3*V('c3'), # three RGB colour triples
)

# Single-float state setters: no Vertex or Normal fields, so cindices and
# nindices will be empty and arithmetic operators are effectively no-ops.

# Transparency for all subsequent primitives (0.0 = opaque, 1.0 = invisible).
Alpha = Primitive(
    25.0,
    V('value'),
)

# OpenGL line width in pixels (does not affect ray-tracing).
Linewidth = Primitive(
    16.0,
    V('width'),
)

# Ray-tracing width scale (proportional multiplier on line radius).
Widthscale = Primitive(
    20.0,
    V('scale'),
)

# ---------------------------------------------------------------------------
# More complex types
# ---------------------------------------------------------------------------

def Arrow(start, end, radius, rgb, *, cone_fraction=0.25, cone_radius_factor=2.0):
    """
    Build a CGO arrow from a cylinder shaft and a cone head.

    The arrow points from *start* to *end*.  The cone occupies the last
    ``cone_fraction`` of the total length; its base radius is
    ``radius * cone_radius_factor``.

    Parameters
    ----------
    start, end : array-like, shape (3,) or (N, 3)
        Tail and tip coordinates.  Batched (N > 1) inputs are supported.
    radius : float or array-like
        Shaft radius.
    rgb : array-like, shape (3,) or (N, 3)
        Colour applied uniformly to both shaft and head.
    cone_fraction : float, optional
        Fraction of total length occupied by the cone head (default 0.25).
    cone_radius_factor : float, optional
        Cone base radius as a multiple of the shaft radius (default 2.0).

    Returns
    -------
    CGO
        Combined shaft + head CGO (can be further transformed with +, *, @).
    """
    start  = np.asarray(start)
    end    = np.asarray(end)
    radius = np.asarray(radius)
    rgb    = np.asarray(rgb)

    # Split point between shaft and cone.
    split = start + (1.0 - cone_fraction) * (end - start)

    shaft = Cylinder(start, split, radius, rgb, rgb)
    head  = Cone(
        split, end,
        radius * cone_radius_factor,  # wide base
        np.zeros_like(radius),        # pointed tip (radius=0)
        rgb, rgb,
        np.ones_like(radius),         # cap the base
        np.zeros_like(radius),        # leave the tip open
    )
    return shaft + head


# ---------------------------------------------------------------------------
# Mesh structures (TRIANGLE_STRIP, TRIANGLE_FAN) 
# ---------------------------------------------------------------------------

def _triangles_spec(n):
    """
    Build a Primitive spec for a TRIANGLES block of exactly n vertices.

    The CGO layout is:
        BEGIN(2.0), TRIANGLES(6.0),
        [NORMAL_OP(5.0), n_i, COLOR_OP(6.0), c_i, VERTEX_OP(4.0), v_i] * n,
        END(3.0)

    Vertices are listed in triples; each triple defines one triangle.
    n must therefore be a multiple of 3.
    """
    assert n % 3 == 0
    spec = [_BEGIN, _GL_TRIANGLES]
    for i in range(n):
        spec += [_NORMAL, N(f'n{i}'),
                 _COLOR,  3*V(f'c{i}'),
                 _VERTEX, X(f'v{i}')]
    spec += [_END]
    return spec


_triangles_cache = {}

def _get_triangles_class(n):
    """Return a cached Primitive class for a TRIANGLES block of n vertices."""
    if n not in _triangles_cache:
        _triangles_cache[n] = Primitive(*_triangles_spec(n))
    return _triangles_cache[n]


def Triangles(vertices, normals, colors):
    """
    Build a CGO TRIANGLES block from a flat list of per-vertex data.

    Vertices are interpreted in triples; each consecutive triple defines
    one triangle.  Unlike the standalone Triangle primitive (opcode 17),
    this uses a BEGIN/TRIANGLES/END block which avoids per-face opcode
    overhead and is more efficient for large meshes.

    Parameters
    ----------
    vertices : array-like, shape (N, 3)
        Vertex positions, N must be a multiple of 3.
    normals : array-like, shape (N, 3)
        Per-vertex surface normals.
    colors : array-like, shape (N, 3)
        Per-vertex RGB colors.

    Returns
    -------
    CGO
    """
    vertices = np.asarray(vertices, dtype=float)
    normals  = np.asarray(normals,  dtype=float)
    colors   = np.asarray(colors,   dtype=float)
    n        = len(vertices)

    TrianglesClass = _get_triangles_class(n)
    args = [x for triple in zip(normals, colors, vertices) for x in triple]
    return TrianglesClass(*args)


# This is a CGO mesh, below is a base mesh that can be read in from or written to STL
# and can be converted to CGO mesh. A bit meshy at the moment.

def _Mesh(vertices, faces, normals=None, colors=None, default_color=(1.0, 1.0, 1.0)):
    """..."""
    vertices = np.asarray(vertices, dtype=float)
    faces    = np.asarray(faces,    dtype=int)

    # Expand face indices to flat per-vertex arrays (3 vertices per face)
    v = vertices[faces].reshape(-1, 3)          # (F*3, 3)

    if normals is None:
        edge1      = v[1::3] - v[0::3]
        edge2      = v[2::3] - v[0::3]
        face_norms = np.cross(edge1, edge2)
        lengths    = np.linalg.norm(face_norms, axis=1, keepdims=True)
        face_norms /= np.where(lengths > 0, lengths, 1.0)
        n = np.repeat(face_norms, 3, axis=0)    # (F*3, 3) flat shading
    else:
        n = np.asarray(normals)[faces].reshape(-1, 3)

    if colors is None:
        c = np.broadcast_to(default_color, v.shape).copy()
    else:
        colors = np.asarray(colors)
        c = (np.broadcast_to(colors, vertices.shape).copy()[faces].reshape(-1, 3)
             if colors.shape == (3,) else colors[faces].reshape(-1, 3))

    return Triangles(v, n, c)


# The actual Mesh can also read/write from/to STL

def _is_ascii_stl(filename):
    """
    Determine whether an STL file is ASCII or binary format.

    The STL spec says ASCII files start with 'solid', but many binary files
    also start with 'solid', so the check combines the header text with the
    expected file size for a binary file of the stated face count.

    Parameters
    ----------
    filename : str

    Returns
    -------
    bool
        True if the file is ASCII, False if binary.
    """
    import os
    with open(filename, 'rb') as f:
        header     = f.read(80)
        count_data = f.read(4)

    if len(count_data) < 4:
        # File too short to be valid binary; assume ASCII.
        return True

    import struct
    n_faces       = struct.unpack('<I', count_data)[0]
    expected_size = 80 + 4 + n_faces * 50   # header + count + faces
    actual_size   = os.path.getsize(filename)

    # A binary STL has exactly this size; ASCII will not match.
    if actual_size == expected_size:
        return False

    # Fall back to header text check.
    return header.lstrip().startswith(b'solid')


def _read_stl_flat(filename):
    """
    Read an STL file and return flat (non-indexed) vertex and normal arrays.

    Each triangle contributes three consecutive rows to the vertex array
    and three identical rows (the face normal) to the normal array.
    The caller is responsible for deduplication if an indexed representation
    is needed.

    Parameters
    ----------
    filename : str

    Returns
    -------
    vertices : ndarray, shape (F*3, 3)
    normals  : ndarray, shape (F*3, 3)
        Per-face normals repeated three times per face (flat shading).
    """
    if _is_ascii_stl(filename):
        return _read_stl_ascii(filename)
    else:
        return _read_stl_binary(filename)


def _read_stl_binary(filename):
    """
    Read a binary STL file into flat vertex and normal arrays.

    Binary STL layout::

        bytes  0– 79  : header (ignored)
        bytes 80– 83  : uint32 face count
        repeat n_faces:
            float32[3] : face normal
            float32[3] : vertex 0
            float32[3] : vertex 1
            float32[3] : vertex 2
            uint16     : attribute byte count (ignored)

    Uses a structured NumPy dtype to read all faces in one call.
    """
    dtype = np.dtype([
        ('normal',   '<f4', (3,)),
        ('vertices', '<f4', (3, 3)),
        ('attr',     '<u2'),
    ])
    with open(filename, 'rb') as f:
        f.read(80 + 4)                           # skip header and face count
        faces = np.frombuffer(f.read(), dtype=dtype)

    normals  = np.repeat(faces['normal'].astype(float), 3, axis=0)  # (F*3, 3)
    vertices = faces['vertices'].astype(float).reshape(-1, 3)        # (F*3, 3)
    return vertices, normals


def _read_stl_ascii(filename):
    """
    Read an ASCII STL file into flat vertex and normal arrays.

    ASCII STL layout::

        solid <name>
          facet normal nx ny nz
            outer loop
              vertex x y z
              vertex x y z
              vertex x y z
            endloop
          endfacet
        endsolid
    """
    vertices = []
    normals  = []
    normal   = None

    with open(filename, 'r') as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == 'facet':
                normal = list(map(float, parts[2:5]))
            elif parts[0] == 'vertex':
                vertices.append(list(map(float, parts[1:4])))
                normals.append(normal)

    return np.array(vertices), np.array(normals)


def read_stl(filename, color=(1.0, 1.0, 1.0)):
    """
    Read an STL file and return it as a CGO object.

    Both ASCII and binary STL formats are supported.  STL stores one normal
    per triangle (flat shading); the normal is assigned to all three vertices
    of each face.  Per-vertex normals (smooth shading) can be obtained
    afterwards by post-processing with a normal-averaging step, but that
    information is not present in the STL format itself.

    Parameters
    ----------
    filename : str
        Path to the .stl file.
    color : tuple of float, optional
        RGB colour applied uniformly to all faces (default: white).

    Returns
    -------
    CGO
        A Triangles-based CGO ready for rendering or further transformation.

    Notes
    -----
    STL normals are often unreliable (some exporters write zero normals and
    expect the reader to recompute them).  If the normals in the file are
    all zero, they are recomputed from the vertex positions using the same
    cross-product method as Mesh().
    """
    vertices, normals = _read_stl_flat(filename)
    # recompute if all-zero normals
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    if np.all(lengths < 1e-6):
        e1 = vertices[1::3] - vertices[0::3]
        e2 = vertices[2::3] - vertices[0::3]
        fn = np.cross(e1, e2)
        fn /= np.linalg.norm(fn, axis=1, keepdims=True).clip(1e-12)
        normals = np.repeat(fn, 3, axis=0)
    colors = np.broadcast_to(color, vertices.shape).copy()
    return Triangles(vertices, normals, colors)


class Mesh:
    """
    A triangle mesh with vertex, face, normal and color data.

    Serves as a geometry container that can be converted to CGO for
    rendering, or exported to standard formats (OBJ, STL).  Unlike a
    raw CGO, a Mesh retains the indexed face structure, so per-vertex
    normal averaging, face selection, and format export are all possible
    after construction.

    Parameters
    ----------
    vertices : array-like, shape (V, 3)
        Vertex positions.
    faces : array-like, shape (F, 3)
        Triangle face indices into vertices.
    normals : array-like, shape (V, 3), optional
        Per-vertex normals.  Computed from face geometry if not supplied.
    colors : array-like, shape (V, 3) or (3,), optional
        Per-vertex RGB colors.  Broadcast to all vertices if shape is (3,).
    default_color : tuple, optional
        Fallback color when colors is None (default: white).

    Attributes
    ----------
    vertices : ndarray, shape (V, 3)
    faces    : ndarray, shape (F, 3)
    normals  : ndarray, shape (V, 3)
    colors   : ndarray, shape (V, 3)
    """
    
    def __init__(self, vertices, faces, normals=None, colors=None,
                 default_color=(1.0, 1.0, 1.0), strip_order=None):
        self.vertices = np.asarray(vertices, dtype=float)
        self.faces    = np.asarray(faces,    dtype=int)

        self.normals  = (self._compute_normals()
                         if normals is None
                         else np.asarray(normals, dtype=float))

        if colors is None:
            self.colors = np.broadcast_to(default_color,
                                          self.vertices.shape).copy()
        else:
            colors = np.asarray(colors, dtype=float)
            self.colors = (np.broadcast_to(colors, self.vertices.shape).copy()
                           if colors.shape == (3,)
                           else colors)

        self.strip_order = strip_order

    def _compute_normals(self):
        """
        Compute smooth per-vertex normals by angle-weighted averaging.
        
        Winding order is checked by comparing each face normal against the
        vector from the mesh centroid to the face centre.  If the majority
        of face normals point inward, all winding orders are flipped before
        averaging, ensuring outward-facing normals for ray tracing.
        """
        V  = len(self.vertices)
        v  = self.vertices[self.faces]           # (F, 3, 3)
        
        # Per-face normals (unnormalised)
        e1 = v[:, 1] - v[:, 0]
        e2 = v[:, 2] - v[:, 0]
        fn = np.cross(e1, e2)                    # (F, 3)
        
        # Check winding: compare face normals to centroid→face_centre vectors
        centroid    = self.vertices.mean(axis=0)
        face_centres = v.mean(axis=1)            # (F, 3)
        outward     = face_centres - centroid    # (F, 3)
        dots        = np.einsum('fi,fi->f', fn, outward)
        
        if np.sum(dots < 0) > np.sum(dots > 0):
            # Majority of normals point inward — flip all faces
            self.faces = self.faces[:, ::-1]
            fn = -fn
            
        # Angle-weighted averaging
        def angle(a, b):
            cos = np.einsum('fi,fi->f', a, b) / (
                np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)
            return np.arccos(np.clip(cos, -1, 1))

        # Recompute v after possible face flip
        v  = self.vertices[self.faces]
        w0 = angle(v[:,1]-v[:,0], v[:,2]-v[:,0])
        w1 = angle(v[:,0]-v[:,1], v[:,2]-v[:,1])
        w2 = angle(v[:,0]-v[:,2], v[:,1]-v[:,2])
        
        normals = np.zeros_like(self.vertices)
        for i, w in enumerate([w0, w1, w2]):
            np.add.at(normals, self.faces[:, i], fn * w[:, None])

        lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        normals /= np.where(lengths > 0, lengths, 1.0)

        return normals

    def flip_normals(self):
        """
        Reverse face winding order and negate all normals.
        
        Use when the centroid heuristic in _compute_normals() produces
        inward-facing normals, e.g. for hollow objects or highly concave
        surfaces where the centroid lies outside the mesh.
        
        Returns
        -------
        Mesh
            A new Mesh with flipped normals; the original is not modified.
        """
        return Mesh(
            self.vertices,
            self.faces[:, ::-1],
            normals  = -self.normals,
            colors   = self.colors,
        )

    def to_cgo(self):
        if self.strip_order is not None:
            return self._to_cgo_strips(*self.strip_order)
        return self._to_cgo_triangles()

    def _to_cgo_triangles(self):
        v = self.vertices[self.faces].reshape(-1, 3)
        n = self.normals[self.faces].reshape(-1, 3)
        c = self.colors[self.faces].reshape(-1, 3)
        return Triangles(v, n, c)

    def _to_cgo_strips(self, n_rings, ring_size):
        """Emit one TriangleStrip per adjacent ring pair."""
        S1 = ring_size + 1   # including closing vertex
        verts  = self.vertices.reshape(n_rings, ring_size, 3)
        norms  = self.normals.reshape(n_rings, ring_size, 3)
        colors = self.colors.reshape(n_rings, ring_size, 3)

        # Close rings
        verts  = np.concatenate([verts,  verts[:, :1, :]],  axis=1)
        norms  = np.concatenate([norms,  norms[:, :1, :]],  axis=1)
        colors = np.concatenate([colors, colors[:, :1, :]], axis=1)

        # Interleave adjacent rings into strip order
        strip_v = np.empty((n_rings-1, 2*S1, 3))
        strip_n = np.empty((n_rings-1, 2*S1, 3))
        strip_c = np.empty((n_rings-1, 2*S1, 3))

        strip_v[:, 0::2] = verts[:-1];  strip_v[:, 1::2] = verts[1:]
        strip_n[:, 0::2] = norms[:-1];  strip_n[:, 1::2] = norms[1:]
        strip_c[:, 0::2] = colors[:-1]; strip_c[:, 1::2] = colors[1:]

        StripClass = _get_strip_class(2 * S1)
        strips = [StripClass(*[x for triple in zip(sn, sc, sv) for x in triple])
                  for sn, sc, sv in zip(strip_n, strip_c, strip_v)]
        return sum(strips[1:], strips[0])

    def write_stl(self, filename):
        """
        Write this mesh to a binary STL file.

        Parameters
        ----------
        filename : str
            Output path.
        """
        import struct
        F = len(self.faces)
        v = self.vertices[self.faces]            # (F, 3, 3)

        # Use per-face normals (average of vertex normals at each face)
        fn = self.normals[self.faces].mean(axis=1)  # (F, 3)
        fn /= np.linalg.norm(fn, axis=1, keepdims=True).clip(1e-12)

        with open(filename, 'wb') as f:
            f.write(b'\0' * 80)                  # header
            f.write(struct.pack('<I', F))
            for i in range(F):
                f.write(struct.pack('<3f', *fn[i]))
                for j in range(3):
                    f.write(struct.pack('<3f', *v[i, j]))
                f.write(struct.pack('<H', 0))    # attribute

    def write_obj(self, filename):
        """
        Write this mesh to a Wavefront OBJ file.

        Includes vertex positions, per-vertex normals, and face definitions.
        Colors are not written since OBJ requires a separate MTL file for
        material properties.

        Parameters
        ----------
        filename : str
            Output path.
        """
        with open(filename, 'w') as f:
            for v in self.vertices:
                f.write(f"v  {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            for n in self.normals:
                f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
            # OBJ indices are 1-based; face format is v//vn
            for face in self.faces:
                i, j, k = face + 1
                f.write(f"f {i}//{i} {j}//{j} {k}//{k}\n")

    @classmethod
    def from_stl(cls, filename, color=(1.0, 1.0, 1.0)):
        """
        Construct a Mesh from an STL file.

        The flat per-face STL layout is converted to an indexed representation:
        vertices are deduplicated and faces are built as index triples.
        Per-vertex normals are computed by angle-weighted averaging after
        loading, replacing the flat per-face STL normals with smooth ones.

        Parameters
        ----------
        filename : str
            Path to the .stl file.
        color : tuple, optional
            Uniform RGB color for all vertices (default: white).

        Returns
        -------
        Mesh
        """
        vertices_flat, _ = _read_stl_flat(filename)
        vertices, inverse = np.unique(vertices_flat, axis=0, return_inverse=True)
        faces = inverse.reshape(-1, 3)
        return cls(vertices, faces, normals=None, colors=color)

    def __add__(self, other):
        """
        Combine two meshes into one by concatenating vertices and faces.

        Parameters
        ----------
        other : Mesh

        Returns
        -------
        Mesh
        """
        offset   = len(self.vertices)
        vertices = np.concatenate([self.vertices, other.vertices])
        faces    = np.concatenate([self.faces, other.faces + offset])
        normals  = np.concatenate([self.normals,  other.normals])
        colors   = np.concatenate([self.colors,   other.colors])
        return Mesh(vertices, faces, normals=normals, colors=colors)



# Tubes (straight and splined)

def Tube(vertices, radii, colors):
    V = np.asarray(vertices)
    R = np.asarray(radii)
    C = np.asarray(colors)
    n = len(V)

    cap1 = np.zeros(n-1)
    cap1[0]  = 1

    cap2 = np.zeros(n-1)
    cap2[-1] = 1

    print(V.shape, R.shape, C.shape, cap1.shape, cap2.shape)
    
    if len(R) ==  - 1:
        return Cylinder2(V[:-1], V[1:], R, C[:-1], C[1:], cap1, cap2)
    
    if len(R) != n:
        raise ValueError(f"Expected {n} or {n-1} radii, got {len(R)}.")

    return Cone(V[:-1], V[1:], R[:-1], R[1:], C[:-1], C[1:], cap1, cap2)
        

def _strip_spec(n):
    """
    Build a Primitive spec for a triangle strip of exactly n vertices.

    The spec encodes the PyMOL-required per-vertex interleaving:
        BEGIN, TRIANGLE_STRIP,
        [NORMAL_OP, n_i, COLOR_OP, c_i, VERTEX_OP, v_i] * n,
        END
    where opcodes are constants (floats) and data fields are descriptors.
    """
    spec = [_BEGIN, _GL_TRIANGLE_STRIP]
    for i in range(n):
        spec += [_NORMAL, N(f'n{i}'),
                 _COLOR,  3*V(f'c{i}'),
                 _VERTEX, X(f'v{i}')]
    spec += [_END]
    return spec


_strip_cache = {}

def _get_strip_class(n):
    """Return a cached Primitive class for triangle strips of length n."""
    if n not in _strip_cache:
        _strip_cache[n] = Primitive(*_strip_spec(n))
    return _strip_cache[n]


def TriangleStrip(vertices, normals, colors):
    """
    Build a CGO triangle strip of arbitrary length.

    The appropriate Primitive class is retrieved from a cache keyed on
    strip length, so repeated calls with the same number of vertices
    (e.g. all segments of a FramedTube) reuse the same class without
    regenerating the spec.

    Parameters
    ----------
    vertices : array-like, shape (N, 3)
    normals  : array-like, shape (N, 3)
    colors   : array-like, shape (N, 3)

    Returns
    -------
    CGO
    """
    vertices = np.asarray(vertices, dtype=float)
    normals  = np.asarray(normals,  dtype=float)
    colors   = np.asarray(colors,   dtype=float)
    n = len(vertices)

    StripClass = _get_strip_class(n)

    # Interleave into positional args matching field order: n0, c0, v0, n1, c1, v1, ...
    args = [x for i in range(n) for x in (normals[i], colors[i], vertices[i])]

    return StripClass(*args)



def FramedTube(vertices, radii, colors, n_sides=8):
    """
    A smooth tube tessellated as triangle strips around a centreline.

    Uses a Bishop (parallel transport) frame to avoid the twisting and
    undefined behaviour of Frenet-Serret frames near inflection points.
    Each segment between adjacent centreline points becomes one
    TriangleStrip; all segments share a cached Primitive class since
    they have identical length (2 * (n_sides + 1) vertices).

    Parameters
    ----------
    vertices : array-like, shape (N, 3)
        Centreline points.
    radii : array-like, shape (N,)
        Cross-section radius at each centreline point.
    colors : array-like, shape (N, 3)
        Per-centreline-point RGB color, constant around each ring.
    n_sides : int, optional
        Number of polygon sides per cross-section ring (default 8).

    Returns
    -------
    CGO
        Triangle-strip mesh tube as a single CGO instance.
    """
    vertices = np.asarray(vertices, dtype=float)
    radii    = np.asarray(radii,    dtype=float)
    colors   = np.asarray(colors,   dtype=float)
    N        = len(vertices)

    # ------------------------------------------------------------------
    # Tangents by finite differences
    # ------------------------------------------------------------------
    tangents        = np.zeros_like(vertices)
    tangents[1:-1]  = vertices[2:] - vertices[:-2]  # central differences
    tangents[0]     = vertices[1]  - vertices[0]    # forward
    tangents[-1]    = vertices[-1] - vertices[-2]   # backward
    tangents       /= np.linalg.norm(tangents, axis=1, keepdims=True)

    # ------------------------------------------------------------------
    # Bishop (parallel transport) frame
    # ------------------------------------------------------------------
    def perp(v):
        """Arbitrary unit vector perpendicular to v."""
        w = np.array([1, 0, 0]) if abs(v[0]) < 0.9 else np.array([0, 1, 0])
        c = np.cross(v, w)
        return c / np.linalg.norm(c)

    rnormals   = np.zeros_like(vertices)
    rbinormals = np.zeros_like(vertices)

    rnormals[0]   = perp(tangents[0])
    rbinormals[0] = np.cross(tangents[0], rnormals[0])

    for i in range(1, N):
        n    = rnormals[i-1] - np.dot(rnormals[i-1], tangents[i]) * tangents[i]
        norm = np.linalg.norm(n)
        rnormals[i]   = n / norm if norm > 1e-6 else perp(tangents[i])
        rbinormals[i] = np.cross(tangents[i], rnormals[i])

    # ------------------------------------------------------------------
    # Cross-section rings: shape (N, S+1, 3), closed by repeating first vertex
    # ------------------------------------------------------------------
    angles = np.linspace(0, 2 * np.pi, n_sides, endpoint=False)
    cos_a  = np.cos(angles)
    sin_a  = np.sin(angles)

    ring_verts = (vertices[:, None, :]
                  + radii[:, None, None]
                  * (cos_a[None, :, None] * rnormals[:, None, :]
                   + sin_a[None, :, None] * rbinormals[:, None, :]))

    ring_norms = (cos_a[None, :, None] * rnormals[:, None, :]
                + sin_a[None, :, None] * rbinormals[:, None, :])

    ring_colors = np.broadcast_to(colors[:, None, :], (N, n_sides, 3)).copy()

    # Vertices: all ring points flattened
    mesh_verts = ring_verts.reshape(-1, 3)        # (N*(S+1), 3)
    mesh_norms = ring_norms.reshape(-1, 3)
    mesh_colors = ring_colors.reshape(-1, 3)

    # Faces: two triangles per quad between adjacent rings
    S1 = n_sides + 1
    i  = np.arange(N-1)
    j  = np.arange(S1-1)                          # exclude closing vertex pair

    # Base index of vertex [ring i, side j]
    base  = (i[:, None] * S1 + j[None, :]).ravel()       # (N-1)*(S1-1) quads
    # Four corners of each quad
    A = base
    B = base + 1
    C = base + S1
    D = base + S1 + 1

    faces = np.concatenate([
        np.stack([A, B, C], axis=1),   # triangle 1
        np.stack([B, D, C], axis=1),   # triangle 2
    ])

    return Mesh(mesh_verts, faces, normals=mesh_norms, colors=mesh_colors, strip_order=(N, n_sides))



