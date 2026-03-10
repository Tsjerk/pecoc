import numpy as np
from scipy.signal import lfilter

from .colorinator import *


def determine_bandwidth(values, hint=None, rule='scott'):
    """
    Determine bandwidth(s) for kernel density estimation.
    
    Parameters
    ----------
    values : array_like
        Input data. If 1-D, treated as single series. If 2-D, each row
        is a separate series.
    hint : None, float, 'global', 'local', or array_like, optional
        Bandwidth specification:
          - None: compute per-series bandwidth (equivalent to 'local')
          - float or int: use this value for all series
          - 'global': compute single bandwidth from all data combined
          - 'local': compute separate bandwidth for each series
          - array_like of length n: explicit bandwidth per series
    rule : {'scott', 'silverman'}, default='scott'
        Rule for automatic bandwidth calculation.
    
    Returns
    -------
    tuple of float
        Bandwidth values, one per series (or repeated if global/fixed).
    
    Raises
    ------
    ValueError
        If hint is an array with incorrect length or an unrecognized value.
    """
    values = np.asarray(values)
    
    ## Handle multiple series
    
    if values.ndim > 1:
        n = len(values)
        if hint is None or hint == 'global':
            # Global bandwidth: compute from all data
            return n*(determine_kde_bw(values.flatten(), None, rule), )
        elif hint == 'local':
            # Local bandwidth: compute per series (None or 'local')
            return tuple(determine_kde_bw(series, None, rule) for series in values)
        elif isinstance(hint, (float, int)):
            # Fixed bandwidth for all series
            return n*(hint, )
        elif isinstance(hint, (np.ndarray, list, tuple)) and len(hint) == n:
            # Explicit bandwidth per series
            return tuple(hint)
        raise ValueError(
            f"Invalid hint: {hint}. Must be a number, 'global', 'local', None, "
            f"or an array of length {n}."
        )
            
    ## Handle single series

    if isinstance(hint, (int, float)):
        return hint
    
    if 'cott'.startswith(rule[1:]):
        return values.std() * values.size**(-1/5)
    elif 'ilverman'.startswith(rule[1:]):
        std = values.std()
        iqr = (np.percentile(values, 75) - np.percentile(values, 25)) / 1.349
        return (values.size * 3/4)**(-1/5) * min(std, iqr)

    raise ValueError(f"Unknown rule: {rule}. Use 'sc[ott]' or 'si[lverman]'.")


def young_vliet_coeffs(sigma):
    """
    Compute Young-van Vliet recursive Gaussian filter coefficients.
    
    Based on: I.T. Young, L.J. van Vliet, M. van Ginkel, 
    "Recursive Gabor filtering", IEEE Trans. Sig. Proc., 
    vol. 50, pp. 2799-2805, 2002.
    """
    if sigma < 0.5:
        raise ValueError('Sigma must be >= 0.5')
    
    # Compute q parameter
    if sigma < 3.556:
        q = -0.2568 + 0.5784 * sigma + 0.0561 * sigma**2
    else:
        q = 2.5091 + 0.9804 * (sigma - 3.556)
    
    q2 = q * q
    
    # m constants
    m0, m1, m2 = 1.16680, 1.10783, 1.40586
    m1sq, m2sq = m1**2, m2**2
    
    # Compute coefficients
    scale = 1.0 / ((m0 + q) * (m1sq + m2sq + 2.0 * m1 * q + q2))
    
    b1 = q * (2.0*m0*m1 + m1sq + m2sq + (2.0*m0 + 4.0*m1)*q + 3.0*q2) * scale
    b2 = -q2 * (m0 + 2.0*m1 + 3.0*q) * scale
    b3 = q2 * q * scale
    
    # B (normalization for output)
    B = (m0 * (m1sq + m2sq) * scale)**2
    a = np.array([1.0, -b1, -b2, -b3])

    return a, B


def recursive_gaussian_1d(signal, sigma):
    """
    Apply Young-van Vliet recursive Gaussian filter.
    """
    a, B = young_vliet_coeffs(sigma)
    b0 = 1.0 / a.sum()
    
    # Forward pass then backward pass and normalize
    return B * lfilter([b0], a, lfilter([b0], a, signal)[::-1])[::-1]


def weighted_kde(x, weights, bw='scott', wbw=None, bins=None, x_range=None, bins_per_bw=5):
    """
    Compute 1D KDE with weights using recursive Gaussian filtering.
    
    Parameters
    ----------
    x : array-like, shape (n,)
        1D positions
    weights : array
        Each ROW of weights is used for a weighted KDE of x
    bw : float or str, default='scott'
        Bandwidth. If float, use that value. If 'scott' or 'silverman',
        estimate using respective rule.
    bins : int, default=256
        Number of bins for discretization
    x_range : tuple or None
        (xmin, xmax) for binning. If None, use data range with 5% padding.
    
    Returns
    -------
    bin_centers : ndarray, shape (bins,)
        Bin center positions
    density : ndarray, shape (bins,)
        KDE density at each bin (normalized to integrate to 1)
    weighted_densities : ndarray, shape (W, bins)
        Sum of input values weighted by local density
    """
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError("x must be 1D")

    weights = np.atleast_2d(weights)
    if weights.shape[1] != x.shape[0]:
        raise ValueError("Weights must have same length as x")
    
    # Estimate bandwidth if needed
    bw = determine_bandwidth(x, method=bw) 
    wbw = bw if wbw is None else estimate_bandwidth(x, method=wbw) 
    
    # Determine range of KDE
    if x_range is not None:
        x_min, x_max = x_range
    else:
        x_min, x_max = x.min(), x.max()
    x_min, x_max = x_min - 3*max(bw, wbw), x_max() + 3*max(bw, wbw)

    # Check binsize
    if bins is None:
        bins = int(bins_per_bw * (x_max - x_min) / min(bw, wbw))
    binsize = (x_max - x_min) / bins
    
    # Create bins
    bin_edges = np.linspace(x_min, x_max, bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    digitized = (bins * (x - x_min)/(x_max - x_min)).astype(int)

    bin_counts = np.bincount(digitized, minlength=bins).astype(float)

    bin_weight_sums = np.array([
        np.bincount(digitized, weights=w, minlength=bins) for w in weights
    ])

    # Convert bandwidth from data units to bin units
    sigma = bw / binsize
    wsigma = wbw / binsize
    
    # Apply recursive Gaussian filter to counts
    smoothed_counts = recursive_gaussian_1d(bin_counts, sigma)
    
    # Apply same filter to each color channel
    smoothed_colors = np.zeros_like(bin_weight_sums)
    for w in range(len(weights)):
        smoothed_weights[:, w] = recursive_gaussian_1d(bin_weight_sums[:, w], wsigma)
    
    # Normalize density to integrate to 1
    density = smoothed_counts / (smoothed_counts.sum() * binsize)
    
    # Compute weighted average colors
    # Avoid division by zero
    weighted = np.zeros_like(smoothed_weights)
    mask = smoothed_counts > 1e-10
    weighted[mask] = smoothed_weights[mask] / smoothed_counts[mask, np.newaxis]
    
    # For bins with no data, could interpolate or leave as zero
    # Here we leave as zero (black for RGB)
    
    return bin_centers, density, weighted


class Feather:
    """
    A single KDE result: bin positions, density, and weighted colors.

    Produced by Pecoc for each input series. Supports indexing to select
    a subset of bins, e.g. for rendering a subrange.

    Attributes
    ----------
    shaft : ndarray, shape (bins,)
        Bin center positions (the KDE x-axis).
    barbs : ndarray, shape (bins,)
        KDE density at each bin, normalized to integrate to 1.
    colors : ndarray, shape (bins, n_channels)
        Per-bin weighted average color, interpolated by the KDE-smoothed
        weight distribution.
    """
    def __init__(self, shaft, barbs, colors):
        self.shaft = shaft
        self.barbs = barbs
        self.colors = colors

    def __len__(self):
        return len(self.shaft)

    def __getitem__(self, item):
        return Feather(self.shaft[item], self.barbs[item], self.colors[item])

    # Drawing...
    

class Pecoc:
    """
    KDE-based density estimator with weighted color mapping.

    For each series in X, computes a 1D KDE using recursive Gaussian
    filtering (Young-van Vliet) and a corresponding color distribution
    derived from per-point weights mapped through a Colorinator.

    Parameters
    ----------
    X : array-like, shape (n_series, n_points) or (n_points,)
        Input data. A 1D array is treated as a single series.
    y : array-like, shape (n_points,), optional
        Values mapped to colors via cmap. If None, colors are uniform.
    cmap : Colorinator or array-like
        Color map. If not a Colorinator, wrapped in one automatically.
    pmin, pmax : float, optional
        Color scale limits. If None, inferred from y.
    bw : float, str, or None
        Bandwidth for density KDE. Use 'gscott' or 'gsilverman' for
        global bandwidth estimation from all series combined and
        'scott' or 'silverman' for per-series estimation.
    cbw : float, str, or None
        Bandwidth for color KDE. Defaults to bw if None.
    bins : int or None
        Number of histogram bins. If None, derived from bins_per_bw.
    x_range : tuple or 'global' or None
        (xmin, xmax) for binning. 'global' uses the range of all X
        combined (plus bandwidth margin). None computes per series.
    bins_per_bw : float
        Controls bin resolution: bins = bins_per_bw * range / bw.
        Default 1 is suitable for interpolated renderers (pcolormesh, CGO tube);
        use 5 for direct point rendering.
        interpolated renderers (pcolormesh, CGO tube).
    """
    def __init__(self, X, y=None, cmap=SBW, pmin=None, pmax=None,
                 bw=None, cbw=None, bins=None, x_range=None, bins_per_bw=1):

        X = np.atleast_2d(X)
        
        if x_range == 'global':
            x_range = (X.min(), X.max())

        if isinstance(bw, str) and bw.startswith('g'):
            bw = determine_bandwidth(X.ravel(), bw[1:])

        if isinstance(cbw, str) and cbw.startswith('g'):
            cbw = determine_bandwidth(X.ravel(), cbw[1:])
            
        if not hasattr(cmap, 'map'):
            cmap = Colorinator(cmap)
        self.cmap = cmap
        colors = None if y is None else cmap.map(np.asarray(y), pmin, pmax)

        self.feathers = [
            Feather(*weighted_kde(xi, colors, bw, cbw, bins, x_range, bins_per_bw))
            for xi in X
        ]
        
    def __getitem__(self, item):
        return self.feathers[item]



