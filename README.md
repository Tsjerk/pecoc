# pecoc — Pattern-Encoded Color-Ordered Correlation

**pecoc** is a Python library for visualizing correlation structure as color-order patterns. Instead of reducing a relationship to a scalar coefficient, PECOC encodes the form of the relationship — whether it is monotone, non-monotone, sigmoidal, or absent — as a color pattern that the human visual system can detect rapidly and preattentively.

The primary display is the **cello plot**: a violin-style density visualization enriched by PECOC encoding, named for carrying more harmonic depth than a violin plot. The same encoding is portable to scatter marginals, bar charts, and three-dimensional structural glyphs for molecular visualization.

The method is described in full in:

> Wassenaar, T.A. (2024). *A New Look at Correlations: Pattern-Encoded Color-Ordered Correlation (PECOC) and Cello Plots for Multivariate Data Visualization.*

---

## The idea in one paragraph

Map the values of a reference variable X to a diverging color gradient (blue → white → orange by default). At each position in the density of a target variable Y, blend the colors of all observations contributing to that position, weighted by their kernel density contribution. The resulting color pattern reveals the structural relationship between X and Y: a smooth gradient means monotone correlation; a reversed gradient means a non-monotone arc; a uniform mixture means independence. This is not a new scalar measure — it is a new way of seeing.

---

## Canonical pattern classes

| Pattern | What it looks like | What it means |
|---|---|---|
| **direct** | smooth gradient across the density | monotone positive (or negative) relationship |
| **reverse** | gradient reverses at the midpoint | non-monotone arc or loop |
| **dwell** | one color dominates broadly, then transitions late | sigmoid with midpoint late in the range |
| **shift** | rapid early transition, then stable | sigmoid with midpoint early in the range |
| **mixed** | all colors throughout, no spatial structure | independence or high-frequency oscillation |

These five classes form a vocabulary that transfers across display types. Once learned in the context of a cello plot, the same reading applies to scatter marginals and 3D structural glyphs.

---

## Installation

```bash
pip install pecoc
```

Or from source:

```bash
git clone https://github.com/Tsjerk/cello
cd cello
pip install -e .
```

**Requirements:** Python ≥ 3.8, NumPy ≥ 1.24, SciPy ≥ 1.10.

Optional: `matplotlib` for 2D rendering; `pymol-open-source` for 3D CGO rendering.

---

## Quick start

```python
import numpy as np
from pecoc import Pecoc, SBW

# X: one or more data series, shape (n_series, n_points) or (n_points,)
# y: reference variable, shape (n_points,)
rng = np.random.default_rng(0)
n = 300
y = np.linspace(0, 1, n)
X = np.vstack([
    -np.cos(np.pi * y) + 0.1 * rng.standard_normal(n),   # direct
     np.cos(2 * np.pi * y) + 0.1 * rng.standard_normal(n), # reverse
])

p = Pecoc(X, y, cmap=SBW)

# Cello plot of the first series
p[0].cello()

# Ridge plot of the second series
p[1].ridge()

# Ensemble cello plot of all series
p.plot_cello()
```

---

## Colormaps

The default colormap is **SBW** (sky blue – white – burnt orange), a diverging map whose endpoints have approximately equal luminance. This makes it accessible to readers with red-green color vision deficiency, since the encoding lies along the blue-yellow perceptual axis.

**BWR** (blue – white – red) is the classical PECOC colormap described in the paper and is available as an alternative.

```python
from pecoc import SBW, BWR, Spectral, VanGogh, Peacock, Viridis, Heat
```

All colormaps are instances of `Colorinator`, which supports explicit non-uniform control points and an arbitrary number of channels (RGB, RGBA, or custom).

---

## Glyphs

### 2D (matplotlib)

| Glyph | Description |
|---|---|
| `Cello` | symmetric ribbon; width encodes density |
| `Ridge` | asymmetric ribbon with one edge at zero; height encodes density |
| `Ribbon` | general two-edge ribbon base class |

```python
feather = p[0]
c = feather.cello()          # returns a Cello object and draws it
r = feather.ridge(draw=False) # build without drawing
```

All 2D glyphs support transform algebra:

```python
c2 = c * 2           # scale
c3 = c + np.array([1, 0])  # translate
c4 = c @ R           # rotate (returns Ribbon — symmetry is not preserved)
```

### 3D (CGO / PyMOL)

CGO objects can be constructed and written to file without PyMOL installed. PyMOL is only required for live rendering via `.draw()`.

```python
from pecoc import FramedTube, Mesh

tube = feather.tube()       # Bishop-framed smooth tube
tube.write('output.cgo')    # write for later use in PyMOL
tube.draw(name='density')   # render directly if PyMOL is active
```

---

## The `Pecoc` object

```python
p = Pecoc(
    X,                  # data series, shape (n_series, n_points) or (n_points,)
    y       = None,     # reference variable; uniform color if None
    cmap    = SBW,      # Colorinator or array-like
    pmin    = None,     # color scale minimum; inferred from y if None
    pmax    = None,     # color scale maximum; inferred from y if None
    bw      = None,     # bandwidth for density KDE; 'global' uses all series
    cbw     = None,     # bandwidth for color KDE; defaults to bw
    bins    = None,     # number of histogram bins; derived from bins_per_bw
    x_range = None,     # (xmin, xmax) or 'global'
    bins_per_bw = 2,    # bin resolution; 2 is appropriate for interpolated renderers
)
```

Each series in `X` produces a `Feather`:

```python
feather = p[0]
shaft, barbs, colors = feather   # unpack: bin centers, density, weighted colors
```

---

## Design notes

**Why not a scalar?** Pearson's r, Spearman's ρ, MIC, and HSIC all reduce a relationship to a number. A monotone relationship and a non-monotone arc can yield identical Spearman correlations. PECOC does not replace these measures — it reveals what they cannot: the *form* of the relationship.

**Why color-order patterns?** Detecting that a color gradient runs smoothly from one end of a density to the other is a preattentive task, completed in under 200 ms without conscious scanning. This is categorically different from reading a color value, which is low in Cleveland and McGill's perceptual hierarchy. PECOC exploits this distinction.

**Layered perception.** A cello plot supports multiple perceptual layers: position and scale (accessible to all readers), distributional shape (modality, skew), and color-order pattern (correlation structure). A novice reads the first two; an expert reads all three. The same figure serves both audiences.

**Caption design.** Captions for cello plots should follow the same structure as a well-constructed sentence: familiar before unfamiliar, simple before complex. First describe what is plotted and on which axes; then distributional features; then the color encoding and reference variable; finally the observed pattern and its interpretation.

---

## Citation

If you use pecoc in your work, please cite:

```
Wassenaar, T.A. (2024). A New Look at Correlations: Pattern-Encoded
Color-Ordered Correlation (PECOC) and Cello Plots for Multivariate
Data Visualization. https://github.com/Tsjerk/cello
```

---

## License

See [LICENSE](LICENSE).
