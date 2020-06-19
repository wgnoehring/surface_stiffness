#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Functions for working with Brillouin zone representations."""
import numpy as np


def extract_path(values, x_grid, y_grid, fraction=1.0, true_edge_lengths=None):
    """Extract values on a path through a quadrant of the Brillouin zone.

    Extract the values on the path
    :math:`\\bar{\Gamma}`-:math:`\\bar{X}`-:math:`\\bar{M}`-:math:`\\bar{\Gamma}`,
    or a scaled version of this path. The path connects the points
    :math:`\\bar{\Gamma}`, :math:`\\bar{X}`, and :math:`\\bar{M}`, where
    :math:`\\bar{\Gamma}` is the origin of the Brillouin zone (zero
    wavector), :math:`\\bar{X}` is the midpoint of the right edge
    (or the lower right corner of the upper right quadrant), and
    :math:`\\bar{M}` is the upper right corner. The path is closed.

    If :math:`n` is the number of rows or columns, then the path
    consists of :math:`3n-3` unique points. We add the starting point
    :math:`\\bar{\Gamma}` at the end, so the returned path has :math:`3n-2` points.

    When determining the path coordinate, we need to be careful because the
    values are associated with pixels. Their coordinates are the coordinates of
    the centroid of the pixel. It follows that the values at the boundary of the
    Brillouin zone are not available. The boundary in a direction corresponds
    to the edge of the last pixel in that direction, not its centroid. For
    example, the points of the segment :math:`\\bar{\Gamma}`-:math:`\\bar{X}`
    have x-coordinates in :math:`[0, l_x)`, where :math:`l_x` is the
    length of the Brillouin zone in x-direction. The offset for the
    segment :math:`\\bar{X}`-:math:`\\bar{M}` is :math:`l_x`, however.

    Parameters
    ----------
    values: numpy.ndarray
        A quadrant of the Brillouin zone as a square array. The lower
        left corner of the array (index= -1, 0) should be the origin.
        The x-coordinate should increase with increasing column number,
        and the y-coordinate should increase with decreasing row number.
    x_grid: numpy.ndarray
        x-coordinates associated with values
    y_grid: numpy.ndarray
        y-coordinates
    fraction: float
        Fraction of the quadrant of the Brillouin zone that is
        traversed. If fraction < 1.0, then the path is shrunk so that
        the endpoints are at a distance lx*fraction and ly*fraction
        from the origin. Only pixels within this zone are considered.

    Returns
    -------
    path: numpy.ndarray
        Array with shape :code:`(N, 4)`, where :code:`N` is the number
        of points of the path. The first two columns are the x- and
        y-coordinates of the points, the third column contains the
        associated values from :code:`values`, and the fourth colum
        is a monotonically increasing coordinate along the path.
    
    Examples
    --------
    >>> component = np.real(stiffness[:, :, voigt_index])
    >>> xx, yy = generate_wavevectors(*component.shape)
    >>> imid = component.shape[0] // 2
    >>> quadrant = component[imid:, imid:]
    >>> xx_quadrant = xx[imid:, imid:]
    >>> yy_quadrant = yy[imid:, imid:]
    >>> path = extract_path(
    >>>     np.flipud(quadrant),
    >>>     xx_quadrant,
    >>>     np.flipud(yy_quadrant),
    >>>     fraction=1.0
    >>> )
    """
    assert x_grid.shape[0] == x_grid.shape[1]
    n = x_grid.shape[0]
    dx = x_grid[0, 1] - x_grid[0, 0]
    dy = y_grid[-2, 0] - y_grid[-1, 0]
    if true_edge_lengths is None:
        true_edge_length_x = x_grid[0, -1] + 0.5 * dx
        true_edge_length_y = y_grid[0, 0] + 0.5 * dy
    else:
        true_edge_length_x, true_edge_length_y = true_edge_lengths
    if fraction == 1.0:
        path = np.zeros((3 * n - 2, 4), dtype=values.dtype)
        # bottom row
        s = np.full(shape=n, fill_value=n - 1), np.arange(0, n, 1, dtype=int)
        grids = (x_grid, y_grid, values)
        for i, grid in zip(range(3), grids):
            path[:n, i] = grid[s]
        # rightmost column
        s = np.arange(n - 2, -1, -1, dtype=int), np.full(n - 1, fill_value=n - 1)
        for i, grid in zip(range(3), grids):
            path[n : 2 * n - 1, i] = grid[s]
        # Diagonal north-east to south-west
        s = np.arange(1, n, 1, dtype=int), np.arange(n - 2, -1, -1, dtype=int)
        for i, grid in zip(range(3), grids):
            path[2 * n - 1 :, i] = grid[s]
        # Coordinate along path
        path[:n, 3] = path[:n, 0]
        path[n : 2 * n - 1, 3] = path[n : 2 * n - 1, 1] + true_edge_length_x
        diagonal_offset_x = true_edge_length_x - x_grid[-2, -2]
        diagonal_offset_y = true_edge_length_y - y_grid[1, 1]
        diagonal_offset = np.sqrt(diagonal_offset_x ** 2 + diagonal_offset_y ** 2)
        # If we operated on a lattice with regular
        # spacing, we could write the diagonal as follows
        # diagonal_offset = (true_edge_length_x - x_grid[-2, -2]) * np.sqrt(2.0)
        # diagonal_length = np.sqrt(dx ** 2 + dy ** 2)
        # path[2 * n - 1 :, 3] = (
        #    np.arange(1, n) * diagonal_length
        #    + diagonal_offset
        #    + true_edge_length_x
        #    + true_edge_length_y
        # )
        diagonal_steps = np.diff(
            path[2 * n - 1 :, 0:2], axis=0, prepend=np.atleast_2d(path[2 * n - 1, 0:2])
        )
        path[2 * n - 1 :, 3] = (
            np.cumsum(np.linalg.norm(diagonal_steps, axis=1))
            + diagonal_offset
            + true_edge_length_x
            + true_edge_length_y
        )
    else:
        rows, cols = np.where(
            (x_grid <= fraction * true_edge_length_x)
            & (y_grid <= fraction * true_edge_length_y)
        )
        rslice = slice(rows[0], rows[-1] + 1)
        cslice = slice(cols[0], cols[-1] + 1)
        path = extract_path(
            values[rslice, cslice],
            x_grid[rslice, cslice],
            y_grid[rslice, cslice],
            fraction=1.0,
            true_edge_lengths=(
                fraction * true_edge_length_x,
                fraction * true_edge_length_y,
            ),
        )
    return path


def generate_wavevectors(nrows, ncols):
    """Generate wavevectors.
    
    Consider a discretization of the Brillouin zone into :code:`nrows`
    by :code:`ncols` pixels. This function generates a meshgrid
    of the components of the wavevectors at the pixel centroids.

    Parameters
    ----------
    nrows : int
        Number of pixel rows (pixels along x)
    ncols : int
        Number of columns

    Returns
    -------
    xx : np.ndarray
        x-components of wavevectors
    yy : np.ndarray
        y-components of wavevectors
    """
    # In the stiffness array, the y-coordinate varies with row
    # number and the x-coordinate varies with column number. Thus,
    # we have to define a meshgrid with cartesian (xy) indexing.
    x_freq = np.fft.fftshift(np.fft.fftfreq(ncols)) * np.pi
    y_freq = np.fft.fftshift(np.fft.fftfreq(nrows)) * np.pi
    xx, yy = np.meshgrid(x_freq, y_freq, sparse=False, indexing="xy")
    return xx, yy
