#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Functions for calculating surface stiffnesses."""
import numpy as np
from matrix import  load_atomistic_stiffness, invert_grid_of_flattened_matrices


def calculate_stiffness(greens_functions, config, num_stddev=0, mask=None):
    r"""Calculate stiffness by inversion of elastic Greens functions

    We assume that the atoms are arranged in a two-dimensional simple cubic
    lattice. Let :math:`N` be the number of atoms along the edge. The input
    array :code:`greens_functions` contains the elastic Greens functions.
    Each :math:`3\times{}3` block contains the Greens functions of one pair
    of sites and, corresponding to one point in the Brillouin zone. Holding
    one site fixed, we can obtain data for all points in the Brillouin zone
    by varying the other site. Varying the first site, we obtain :math:`N^2`
    :math:`3\times{}3` matrices for each Brillouin zone point. This function
    first calculates the average at each Brillouin zone point and then inverts
    the :math:`3\times{}3` matrices to obtain stiffness matrices.

    The :math:`N^2` :math:`3\times{}3` matrices at a given Brillouin zone
    point take the same values in case of a pure crytal. In the case of 
    an alloy (or some other disorder), the values differ. We can compute
    a confidence interval of the variation in stiffness by adding and 
    subtracting a multiple of the standard deviation of the Green's 
    functions before inversion. Since the Greens functions are complex-valued,
    we use the following approach. Noting that the variance of a complex
    number is :math:`Var[z] = Var[Re(z)] + Var[Im(z)]`, where :math:`Re`
    and :math:`Im` indicate the real and imaginary parts, respectively, 
    we compute :math:`z +- m(\sigma(Re(z)) + i * \sigma(Im(z)))`, where
    :math:`\sigma` is the standard deviation, and :math:`m` is the number
    of standard deviations to add.

    Parameters
    ----------
    greens_functions: array-like
        Array with shape :code:`((N*N*3), (N*N*3))`, where each block of shape
        :code:`(3,3)` contains the Greens functions for one atomic site.
    config: surface_stiffness.configurations.Configuration
        Configuration of crystal and material, which provides the information
        of how to reshape a row in greens_functions to obtain an array of shape
        :code:`(N,N)`, whose components are arranged according to the order of
        the corresponding sites in space, or the corresponding wavevectors in
        reciprocal space.
    num_stddev: int
        Number of standard deviations to add/subtract from the average Greens
        functions to obtain a confidence interval.
    mask: array-like
        Array of ones and zeros, or None. If None, then the average of the 
        Greens functions runs over all sites. If not None, then the average
        runs only over sites with a one in :code:`mask`.

    Returns
    -------
    mean_stiff: numpy.ndarray
        :code:`((N*3), (N*3))` array containing the inverses of the
        average Greens functions. Each block of shape :code:`(3,3)`
        the inverse of the corresponding block in the :code:`((N*3),(N*3))` 
        of average Greens functions.
    upper_stiff: numpy.ndarray
        Upper confidence limit, array with the same shape as :code:`mean_stiff`
    lower_stiff: numpy.ndarray
        Lower confidence limit, array with the same shape as :code:`mean_stiff`

    """
    if num_stddev < 0:
        raise ValueError("number of standard deviations for confidence interval must be greater or equal to zero")
    mean_both, = load_atomistic_stiffness(
        greens_functions,
        reshape=config.crystal.reshape,
        statistics=[np.average],
        part="both",
        indexing="matrix",
        mask=mask,
    )
    mean_stiff = invert_grid_of_flattened_matrices(mean_both)
    upper_stiff = lower_stiff = None
    if num_stddev > 0:
        std_both, = load_atomistic_stiffness(
            greens_functions,
            reshape=config.crystal.reshape,
            statistics=[np.std],
            part="both",
            indexing="matrix",
            mask=mask,
        )
        mean_real, std_real = load_atomistic_stiffness(
            greens_functions,
            reshape=config.crystal.reshape,
            statistics=[np.average, np.std],
            part="real",
            indexing="matrix",
            mask=mask,
        )
        mean_imag, std_imag = load_atomistic_stiffness(
            greens_functions,
            reshape=config.crystal.reshape,
            statistics=[np.average, np.std],
            part="imag",
            indexing="matrix",
            mask=mask,
        )
        fluctuation = std_real + 1.0j * std_imag
        upper = mean_both + num_stddev * fluctuation
        lower = mean_both - num_stddev * fluctuation
        upper_stiff = invert_grid_of_flattened_matrices(upper)
        lower_stiff = invert_grid_of_flattened_matrices(lower)
    return mean_stiff, upper_stiff, lower_stiff
