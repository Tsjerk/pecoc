"""
Colorinator: flexible multi-channel color mapping with non-uniform control points.

Provides the Colorinator class, which interpolates colors along a gradient
defined by explicit control points. Unlike matplotlib colormaps, control points
can represent any meaningful monotonic parameterization (quantiles, distances,
physical quantities), and the number of channels is unrestricted — supporting
RGB, RGBA, or arbitrary extra channels for use in custom renderers.

Predefined colormaps
--------------------
BWR       : blue – white – red (diverging)
PRGn      : purple – white – green (diverging)
BGW       : blue – turquoise – white
RYW       : red – yellow – white
PMG       : purple – magenta – green
Spectral  : blue – cyan – green – yellow – red
HWC       : hotpink – white – chartreuse
BOX       : five-color scale anchored to boxplot quantiles (0, 0.25, 0.5, 0.75, 1)
"""

import numpy as np

# Boxplot with marker for median
#
#                      |--------[============||============]--------|
BOXPLOT = (          0.00,    0.25,      0.48,0.52,    0.75,    1.00)
BOXRADI = (               0.10,    0.30,    0.20,    0.30,    0.10   )
FIVECOL = np.array(((0,1,1), (0,1,0),     (1,1,0),     (1,0,0), (1,0,1)))


class Colorinator:
    """Maps data values to colors with support for non-uniform color scales.
    
    Useful for creating color gradients where averaging produces meaningful
    intermediate colors (e.g., blue-white-red for negative-neutral-positive).

    points define the control points of the color scale. These are not
    restricted to quantiles — they can represent any monotonic
    parameterization (e.g., distances, component magnitudes, or
    normalized coordinates).
    
    Parameters
    ----------
    colors : array-like of shape (n_colors, 3) or (n_colors, 4)
        RGB or RGBA color values in [0, 1] range.
    points : array-like of shape (n_colors,), optional
        Control positions of colors. If None, colors are evenly spaced
        from 0 to 1.
    """
    def __init__(self, colors, points=None):
        q = np.arange(len(colors)) / (len(colors) - 1)
        if points is None:
            points = q
        if len(colors) != len(points):
            colors = np.array([ np.interp(points, q, c) for c in np.asarray(colors).T ]).T
        self.colors = np.asarray(colors)
        self.points = points

    def __call__(self, points, pmin=None, pmax=None):
        return self.map(points, pmin, pmax)
        
    def __len__(self):
        return len(self.colors)

    def __getitem__(self, item):
        return self.colors[item]

    def __str__(self):
        return '\n'.join(f'{q:6.4f}: {c}' for q, c in zip(self.points, self.colors)) 
    
    def map(self, points, pmin=None, pmax=None):
        """Map data values to colors.
        
        Values are normalized to [0, 1] based on the range of the input,
        then interpolated against the color gradient.
        
        Parameters
        ----------
        points : array-like
            Data values to map to colors.
        
        Returns
        -------
        ndarray of shape (n_points, n_channels)
            Color values for each input point.
        """
        points = np.asarray(points).astype(float)
        if pmin is None:
            pmin = points.min()
        if pmax is None:
            pmax = points.max()
        points = (points - pmin) / (pmax - pmin)
        return np.array([ np.interp(points, self.points, c) for c in self.colors.T ]).T
    
    
BWR = Colorinator([
    (0.0, 0.0, 1.0),   # blue
    (1.0, 1.0, 1.0),   # white
    (1.0, 0.0, 0.0)    # red
])

PRGn = Colorinator([
    (0.45, 0.00, 0.55),  # purple
    (1.00, 1.00, 1.00),  # white
    (0.00, 0.55, 0.35)   # green
])

BGW = Colorinator([
    (0.00, 0.20, 0.90),  # blue
    (0.20, 0.80, 0.60),  # turquoise
    (1.00, 1.00, 1.00)   # white
])

RYW = Colorinator([
    (0.80, 0.00, 0.00),  # red
    (1.00, 0.80, 0.00),  # yellow
    (1.00, 1.00, 1.00)   # white
])

PMG = Colorinator([
    (0.30, 0.00, 0.60),  # purple
    (0.90, 0.00, 0.70),  # magenta
    (0.00, 0.70, 0.30)   # green
])

Spectral = Colorinator([
    (0.00, 0.00, 0.90),  # blue
    (0.00, 0.70, 1.00),  # cyan
    (0.00, 1.00, 0.30),  # green
    (1.00, 1.00, 0.00),  # yellow
    (1.00, 0.00, 0.00)   # red
])

HWC = Colorinator([
    (1.00, 0.00, 0.50),  # hotpink
    (1.00, 1.00, 1.00),  # white
    (0.50, 1.00, 0.00)   # chartreuse
])

# Perceptually uniform sequential (viridis-like)
Viridis = Colorinator([
    (0.267, 0.005, 0.329),  # dark purple
    (0.253, 0.405, 0.600),  # blue
    (0.163, 0.698, 0.498),  # teal-green
    (0.478, 0.821, 0.318),  # yellow-green
    (0.993, 0.906, 0.144),  # yellow
])

# Heat / density
Heat = Colorinator([
    (0.00, 0.00, 0.00),  # black
    (0.80, 0.00, 0.00),  # red
    (1.00, 0.80, 0.00),  # yellow
    (1.00, 1.00, 1.00),  # white
])

# Dark sequential (deep blue to white, lightness-monotonic for 3D rendering)
Blues = Colorinator([
    (0.03, 0.06, 0.35),  # navy
    (0.10, 0.45, 0.80),  # cornflower
    (0.60, 0.85, 1.00),  # light blue
    (1.00, 1.00, 1.00),  # white
])

# Equivalent warm sequential
Reds = Colorinator([
    (0.35, 0.03, 0.03),  # dark red
    (0.85, 0.25, 0.10),  # orange-red
    (1.00, 0.80, 0.60),  # peach
    (1.00, 1.00, 1.00),  # white
])

# Van Gogh: swirling night blues and starlight yellows
VanGogh = Colorinator([
    (0.05, 0.05, 0.20),  # deep midnight blue
    (0.10, 0.20, 0.55),  # cobalt blue
    (0.15, 0.45, 0.70),  # cerulean
    (0.975, 0.95, 0.60),  # 
    (0.95, 0.85, 0.20),  # starlight yellow
    (0.60, 0.55, 0.20),  # ochre
])

# Peacock tail feather: dark root through iridescent eye to golden tip
PEACOCK_PTS = (0.00, 0.15, 0.35, 0.50, 0.65, 0.80, 1.00)
Peacock = Colorinator([
    (0.05, 0.05, 0.05),  # dark root
    (0.00, 0.20, 0.30),  # dark teal shaft
    (0.00, 0.55, 0.55),  # deep cyan
    (0.00, 0.40, 0.20),  # iridescent green eye
    (0.00, 0.65, 0.80),  # electric blue eye ring
    (0.80, 0.75, 0.10),  # golden bronze tip
    (0.95, 0.90, 0.60),  # pale gold edge
], points=PEACOCK_PTS)

# Signature diverging: sky blue – white – burnt orange
SBW = Colorinator([
    (0.15, 0.55, 0.90),  # sky blue
    (1.00, 1.00, 1.00),  # white
    (0.85, 0.40, 0.05),  # burnt orange
])

BOX = Colorinator(colors=FIVECOL, points=BOXPLOT)


