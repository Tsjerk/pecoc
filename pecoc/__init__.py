"""
pecoc — Pattern-Encoded Color-Ordered Correlation
==================================================

PECOC is a visualization encoding in which the values of a reference
variable are mapped to a continuous color gradient and mixed according
to their kernel-density-weighted correspondence with a target variable.
The resulting color pattern reflects the structural nature of the
relationship — not as a scalar, but as a shape that is immediately
readable by the human visual system.

The central insight is that the visual system detects color-order
patterns rapidly and preattentively — in under 200 ms, without serial
scanning — a capability that is largely untapped by conventional
correlation analysis. PECOC recruits this capability for data analysis.

Canonical pattern classes
-------------------------
Five pattern classes cover the most common relational structures:

  direct   : smooth gradient across the density; monotone relationship
  reverse  : gradient reverses at the midpoint; non-monotone (arc/loop)
  dwell    : one color dominates broadly before a late transition; sigmoid
             with midpoint late in the range
  shift    : rapid early transition to a stable second state; sigmoid with
             midpoint early in the range
  mixed    : all colors present without spatial structure; independence or
             high-frequency oscillation

These classes form a vocabulary that transfers across display types:
once learned from a cello plot, the same reading applies to scatter
marginals and 3D structural glyphs.

The cello plot
--------------
A cello plot is a violin-style marginal density visualization enriched
by PECOC encoding. It is named for carrying more harmonic depth than
a violin plot — more relational information in the same visual space.
Each bin of the KDE is filled with the weighted average color of the
observations contributing to it, so the color pattern directly encodes
the structural relationship between the reference and target variables.

Cello plots support layered perception: a novice reads position and
shape (distribution); an expert additionally reads the color-order
pattern (correlation structure). Both audiences are served by the same
figure, with no information hidden from either.

Quick start
-----------
    import numpy as np
    from pecoc import Pecoc, SBW

    # X: array of series (n_series, n_points); y: reference variable
    p = Pecoc(X, y, cmap=SBW)

    # Draw a cello plot of the first series
    p[0].cello()

    # Draw a ridge plot of all series
    p.plot_ridge()

Classes
-------
Pecoc           : main interface; computes KDE feathers for an array of series
Feather         : single KDE result (shaft, barbs, colors) with glyph methods
Colorinator     : multi-channel color map with explicit non-uniform control points

Colormaps
---------
The default colormap for PECOC is SBW (sky blue – white – burnt orange),
a diverging map whose endpoints have approximately equal luminance,
making it accessible to readers with red-green color vision deficiency.
BWR (blue – white – red) is the classical alternative described in the
paper; SBW is recommended for publication figures.

  SBW      : sky blue – white – burnt orange  [recommended default]
  BWR      : blue – white – red               [classical PECOC colormap]
  PRGn     : purple – white – green           [diverging]
  BGW      : blue – turquoise – white         [sequential warm]
  RYW      : red – yellow – white             [sequential warm]
  PMG      : purple – magenta – green         [diverging, high contrast]
  HWC      : hotpink – white – chartreuse     [diverging, vivid]
  Spectral : blue – cyan – green – yellow – red
  VanGogh  : midnight blue – cobalt – cerulean – ochre – starlight yellow
  Peacock  : dark root – teal – cyan – iridescent green – electric blue – gold
  Viridis  : perceptually uniform dark purple to yellow  [sequential]
  Heat     : black – red – yellow – white                [density/heat]
  Blues    : navy to white                               [sequential]
  Reds     : dark red to white                           [sequential]
  BOX      : five-color scale anchored to boxplot quantiles (0, .25, .5, .75, 1)

Glyphs (2D, via matplotlib)
---------------------------
  Cello    : symmetric ribbon; width encodes density (the cello plot glyph)
  Ridge    : asymmetric ribbon; height encodes density (ridge / joy plot)
  Ribbon   : general two-edge ribbon base class

Glyphs (3D, via CGO / PyMOL)
-----------------------------
CGO objects can be constructed and written to file without PyMOL;
PyMOL is only required for live rendering via .draw().

  Tube        : tapered cylinder chain along a centreline
  FramedTube  : smooth triangle-strip tube with Bishop parallel-transport frame
  Mesh        : indexed triangle mesh with STL and OBJ I/O

References
----------
Wassenaar, T.A. (...). A New Look at Correlations: Pattern-Encoded
  Color-Ordered Correlation (PECOC) and Cello Plots for Multivariate
  Data Visualization. https://github.com/Tsjerk/cello

Cleveland, W.S. and McGill, R. (1984). Graphical perception: theory,
  experimentation, and application to the development of graphical methods.
  Journal of the American Statistical Association, 79(387), 531-554.

Tukey, J.W. (1977). Exploratory Data Analysis. Addison-Wesley.
"""

__all__ = [
    # Core
    'Pecoc', 'Feather',
    'weighted_kde', 'determine_bandwidth',
    'recursive_gaussian_1d', 'young_vliet_coeffs',
    # Colorinator
    'Colorinator',
    'SBW', 'BWR', 'PRGn', 'BGW', 'RYW', 'PMG', 'HWC',
    'Spectral', 'VanGogh', 'Peacock',
    'Viridis', 'Heat', 'Blues', 'Reds', 'BOX',
    # 2D glyphs
    'Glyph2D', 'Ribbon', 'Cello', 'Ridge',
    # 3D glyphs
    'Glyph3D', 'Tube', 'FramedTube', 'Mesh',
]

from .colorinator import (
    Colorinator,
    BWR, SBW, PRGn, BGW, RYW, PMG, HWC,
    Spectral, VanGogh, Peacock,
    Viridis, Heat, Blues, Reds,
    BOX,
)

from .pecoc import (
    determine_bandwidth,
    young_vliet_coeffs,
    recursive_gaussian_1d,
    weighted_kde,
    Feather,
    Pecoc,
)

from .glyph import (
    Glyph2D,
    Ribbon,
    Cello,
    Ridge,
    Glyph3D,
    Tube,
    FramedTube,
    Mesh,
)

