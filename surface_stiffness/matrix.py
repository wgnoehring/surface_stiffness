#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Functions for working with matrices."""
import sys
from abc import ABC
import numpy as np
from numpy import ma
import accupy

matrix_indices_for_voigt_index = {
    0: (0, 0),
    1: (1, 1),
    2: (2, 2),
    3: (1, 2),
    4: (0, 2),
    5: (0, 1),
}
"""Matrix indices corresponding to a given Voigt index as key-value pairs."""

voigt_index_for_matrix_indices = {
    (0, 0) : 0,
    (1, 1) : 1,
    (2, 2) : 2,
    (1, 2) : 3,
    (2, 1) : 3,
    (0, 2) : 4,
    (2, 0) : 4,
    (0, 1) : 5,
    (1, 0) : 5,
}
"""Indices in Voigt notation for matrix indices as key-value pairs."""

label_for_component = {0: "xx", 1: "yy", 2: "zz", 3: "yz", 4: "xz", 5: "xy"}
"""Subscript strings for the indices of a vector in Voigt notation as key-value pairs"""


def convert_matrix_to_voigt_vector(matrix):
    """Convert a 3×3 matrix into a Voigt vector.

    Parameters
    ----------
    matrix: array-like
        Symmetric 3×3 matrix

    Returns
    -------
    vector: numpy.ndarray
        Vector with 6 elements
    """
    vector = np.zeros(6, dtype=matrix.dtype)
    for i, j in zip(*np.triu_indices(3)):
        voigt_index = voigt_indices_for_matrix_index[(i, j)]
        vector[voigt_index] = matrix[i, j]
    return vector


def convert_voigt_vector_to_matrix(vector):
    """Convert a Voigt vector into a 3×3 matrix.


    Parameters
    ----------
    vector: numpy.ndarray
        Vector with 6 elements

    Returns
    -------
    matrix: array-like
        Symmetric 3×3 matrix
    """
    matrix = np.zeros((3, 3), dtype=vector.dtype)
    for i, j in np.ndindex(3, 3):
        voigt_index = voigt_indices_for_matrix_index[(i, j)]
        matrix[i, j] = vector[voigt_index]
    return matrix


def fourier_transform_symmetric_square_block_matrix(matrix, reshape, block_size=3):
    """Calculate the Fourier transform of a symmetric square block matrix.

    Parameters
    ----------
    matrix: numpy.ndarray 
        Symmetric square two-dimensional array which can be decomposed into blocks.
    reshape: Reshape
    block_size: int
        Size of the square blocks.

    Returns
    -------
    matrix_fft: numpy.ndarray
        Array with the same shape as :code:`matrix` containing the Fourier transformed
        components of the :code:`block_size` × :code:`block_size` blocks in :code:`matrix`
    """
    matrix_fft = np.zeros(matrix.shape, dtype=np.complex_)
    assert(
        matrix.ndim == 2 and 
        matrix.shape[0] == matrix.shape[1] and 
        matrix.shape[0]%block_size == 0
    )
    num_blocks = int(matrix.shape[0] / block_size)
    print(f"input matrix for FFT is partitioned into {num_blocks}x{num_blocks} blocks of size {block_size}x{block_size}")
    for block_index in range(num_blocks):
        sys.stdout.write(f'taking FFT of block column {block_index+1}\r')
        sys.stdout.flush()
        # We make no assumption about symmetry of 3x3 blocks
        for i in range(block_size):
            for j in range(block_size):
                row = block_size * block_index + i
                values_on_grid = reshape.vector_to_grid(matrix[row, j::block_size])
                values_on_grid = np.roll(values_on_grid, -(block_index // (int(np.sqrt(num_blocks)))), axis=0)
                values_on_grid = np.roll(values_on_grid, -block_index, axis=1)
                traffo_on_grid = np.fft.fftshift(np.fft.fft2(values_on_grid))
                matrix_fft[row, j::block_size] = reshape.grid_to_vector(traffo_on_grid)
    sys.stdout.write('\n')
    return matrix_fft


def calculate_blockwise_inverse(matrix, block_size=3):
    """Invert sub-blocks

    Invert blocks of a matrix.
    
    Parameters
    ----------
    matrix: array-like
        Matrix which can be decomposed into square blocks of size :code:`block_size` × :code:`block_size`
    block_size: int
        Size of the blocks

    Returns
    -------
    blockwise_inverse: numpy.ndarray
        Array with the same shape as :code:`matrix` containing the inverted blocks.
    """
    blockwise_inverse = np.zeros_like(matrix)
    num_blocks = int(blockwise_inverse.shape[0] / block_size)
    for i in range(num_blocks): 
        for j in range(num_blocks): 
            sys.stdout.write(f'inverting block {i*num_blocks+j+1} out of {num_blocks*num_blocks}\r')
            sys.stdout.flush()
            slice_i = slice(block_size*i, block_size*(i+1), 1)
            slice_j = slice(block_size*j, block_size*(j+1), 1)
            blockwise_inverse[slice_i, slice_j] = np.linalg.inv(matrix[slice_i, slice_j])
    sys.stdout.write("\n")
    return blockwise_inverse


def extract_local_stiffness(stiff, atom_index, indices, reshape, part="real", indexing="voigt"):
    """Extract stiffness for a specific site.

    Parameters
    ----------
    stiff: array-like
       Matrix which for a system of :math:`N\\times{}N` atoms has shape :math:`3N\\times{}3N` and
       contains :math:`N\\times{}N` :math:`3\\times 3`stiffness matrices for pairs of atoms.
    atom_index: int
        Index of the central site. Stiffness components with this site as first site are selected.
    indices: tuple or int
        Matrix indices or equivalent index in Voigt notation of the component to extract in each
        selected :math:`3\\times{}3` stiffness matrix, see parameter :code:`indexing`.
    reshape: Reshape
    part: string 
        'real' for real part or 'imag' for imaginary part of the complex stiffness, 
        otherwise 'both'
    indexing: string
        :code:`indexing='voigt'` means that :code:`indices` must be :code:`int` and is 
        interpreted as the Voigt index of the matrix element. :code:`indexing='matrix'` 
        means that :code:`indices` is a tuple containing the row and column indices of 
        the component to extract.

    Returns
    -------
    components: numpy.ndarray
        :math:`N\\times{}N` array of stiffness components.
    """
    block_size = 3
    if indexing == "voigt":
        i, j = matrix_indices_for_voigt_index[indices]
    elif indexing == "matrix": 
        i, j = indices
    else:
        raise ValueError("unknown indexing scheme")
    row = block_size * atom_index + i
    row = stiff[row, j::block_size]
    if part == "real":
        return np.real(reshape.vector_to_grid(row))
    elif part == "imag":
        return np.imag(reshape.vector_to_grid(row))
    elif part == "both":
        return reshape.vector_to_grid(row)
    else:
        raise ValueError


def load_atomistic_stiffness(stiff, reshape, statistics=None, atom_index=0, part="real", mask=None, indexing="voigt"):
    """Load atomistic stiffness from a file 

    The result still needs to be divided by the area per atom.

    Parameters
    ----------
    stiff: array-like
       Matrix which for a system of :math:`N\\times{}N` atoms has shape :math:`3N\\times{}3N` and
       contains :math:`N\\times{}N` :math:`3\\times 3`stiffness matrices for pairs of atoms.
    reshape: Reshape
    statistics: list of numpy statistics routines
        The routines will be applied to the (N, N, N*N) array which
        contains the (N, N) stiffness matrices of the N*N surface atoms,
        in order to calculate the sample statistic. The statistic is an
        array of shape (N, N). The calculation is performed for all six
        Voigt indices, therefore the shape of the result is (N, N, 6).
    atom_index: int
        If statistics is None, then only the (N, N, 6) components
        for the atom with array index atom_index will be extracted,
        i.e. the stiffness for this atom as central atom.
    part: "real", or "imag", or "both"
        Whether to extract the real or imaginary parts, or both
    indexing: string
        "voigt" means that only the six components of Voigt notation will be
        loaded, which is sufficient if the stiffness matrices are symmetric.
        "matrix" means that all nine stiffness components will be loaded.

    Returns
    -------
    arr: tuple of arrays of shape (N, N, P)
        N is the number of atoms along the edge of the configuration.
        If Ns statistics were requested, then the tuple contains the
        resulting Ns arrays. If no statistics were requested, then
        the tuple contains only the array for atom index atom_index.
        If :code:`indexing='voigt'`, then the (statistics of the) 
        upper triangular matrix are returned in Voigt notation and
        :code:`P=6`. If :code:`indexing='matrix'`, then 
        (statistics of) all elements are returned.

    """
    assert stiff.ndim == 2 and stiff.shape[0] == stiff.shape[1]
    if mask is not None:
        zeros = ma.zeros 
    else:
        zeros = np.zeros
    # get required dtype
    tmp = extract_local_stiffness(
        stiff, 0, 0, reshape, part=part
    )
    required_dtype = tmp.dtype

    if indexing == "voigt": 
        indices = tuple(range(6))
        num_indices = 6
    elif indexing == "matrix":
        indices = tuple(np.ndindex(3, 3))
        num_indices = 9
    else: 
        raise ValueError(f"indexing {indexing} not supported")

    num_atoms = reshape.grid_shape[0] * reshape.grid_shape[1]
    if statistics is not None:
        output = []
        for op in statistics:
            print(f"calculating {op.__name__}")
            stat = zeros((reshape.grid_shape[0], reshape.grid_shape[1], num_indices), dtype=required_dtype)
            for ii in range(num_indices):
                variables = zeros(
                    (reshape.grid_shape[0], reshape.grid_shape[1], num_atoms),
                    dtype=required_dtype
                )
                if mask is not None:
                    variables.mask = zeros(variables.shape)
                for atom_index in range(num_atoms):
                    variables[:, :, atom_index] = extract_local_stiffness(
                        stiff, atom_index, indices[ii], reshape, part=part, indexing=indexing
                    )
                    if mask is not None:
                        variables.mask[:, :, atom_index] = mask[atom_index]
                stat[:, :, ii] = op(variables, axis=2)
            output.append(stat)
        return tuple(output)
    else:
        print(f"extracting data for atom index {atom_index}")
        arr = zeros((reshape.grid_shape[0], reshape.grid_shape[1], num_indices), dtype=required_dtype)
        for ii in range(num_indices):
            arr[:, :, ii] = extract_local_stiffness(
                stiff, atom_index, indices[ii], reshape, part=part, indexing=indexing
            )
        return (arr,)


def histogram_stiffness(stiff, reshape, indices, part="real", num_bins=100, mask=None, indexing="voigt"):
    """Generate histograms of stiffness components.

    Parameters
    ----------
    stiff: array-like
       Matrix which for a system of :math:`N\\times{}N` atoms has shape :math:`3N\\times{}3N` and
       contains :math:`N\\times{}N` :math:`3\\times 3`stiffness matrices for pairs of atoms.
    indices: int
        Index in voigt notation of the component of the 
        :math:`3\\times{}3` stiffness matrix 
    indices: tuple or int
        Matrix indices or equivalent index in Voigt notation of the component of the 
        :math:`3\\times{}3` stiffness matrices that should be histogrammed. 
        See parameter :code:`indexing`
    part: string 
        'real' for real part or 'imag' for imaginary part of the complex stiffness
    num_bins: int
        Number of histogram bins
    mask: array-like
        Masks values in the stiffness matrix
    indexing: string
        :code:`indexing='voigt'` means that :code:`indices` must be :code:`int` and
        is interpreted as the Voigt index of the matrix element. :code:`indexing='matrix'`
        means that :code:`indices` is a tuple containing the row and column indices of the 
        component to histogram.

    Returns
    -------
    histograms: array-like
        For :math:`N\times N` surface atoms this is a 
        :math:`N\times N\timesM` array, where :math:`M` is
        the number of bins
    bin_edges: array-like
        For :math:`N\times N` surface atoms this is a 
        :math:`N\times N\timesM+1` array, where :math:`M` is
        the number of bins
    """
    assert stiff.ndim == 2 and stiff.shape[0] == stiff.shape[1]
    if mask is not None:
        zeros = ma.zeros 
    else:
        zeros = np.zeros
    num_atoms = reshape.grid_shape[0] * reshape.grid_shape[1]
    variables = zeros(
        (reshape.grid_shape[0], reshape.grid_shape[1], num_atoms), dtype=float,
    )
    histograms = zeros(
        (reshape.grid_shape[0], reshape.grid_shape[1], num_bins), dtype=float
    )
    bin_edges = zeros(
        (reshape.grid_shape[0], reshape.grid_shape[1], num_bins+1), dtype=float
    )
    if mask is not None:
        variables.mask = zeros(variables.shape)

    for atom_index in range(num_atoms):
        variables[:, :, atom_index] = extract_local_stiffness(
            stiff, atom_index, indices, reshape, part=part, indexing=indexing
        )
        if mask is not None:
            variables.mask[:, :, atom_index] = mask[atom_index]
    for (i, j) in np.ndindex((reshape.grid_shape[0], reshape.grid_shape[1])):
        histograms[i, j, :], bin_edges[i, j, :] = np.histogram(
            variables[i, j, :], bins=num_bins, density=True
        )
    return histograms, bin_edges


def invert_grid_of_flattened_matrices(array, epsilon=1e-13):
    """Invert a grid of flattened matrices

    Given a :math:`M×N×P` array, interpret the values along the 
    third dimension as the elements of a :math:`3×3` matrix. 
    Invert the matrices and return the inverses as 
    :math:`M×N×P` array.

    Parameters
    ----------
    array: array-like
        M×N×P array of inverses. If :code:`P=6`, then interpret 
        :code:`array[i, j, :]` as the Voigt vector representation
        of a symmetric matrix. If :code:`P=9`, then assume that 
        :code:`array[i, j, :]` is the flattened array obtained by 
        :code:`numpy.ravel`.
    epsilon: float
        Matrices whose 2-norm is less or equal than epsilon are
        assumed to be zero and not inverted. The inverse is zero.

    Returns 
    -------
    inverse: numpy.ndarray
        M×N×P array of inverses

    """
    inverse = np.zeros_like(array)
    if array.shape[2] == 6:
        invert = lambda x: invert_voigt_representation(x)
    elif array.shape[2] == 9:
        invert = lambda x: np.linalg.inv(x.reshape(3, 3)).ravel()
    else:
        raise ValueError(f"invalid number of elements {array.shape[2]}")
    for i, j in np.ndindex(*array.shape[:2]):
        if np.linalg.norm(array[i, j, :]) > epsilon:
            inverse[i, j, :]  = invert(array[i, j, :])
    return inverse


def invert_voigt_representation(vector):
    """Invert the symmetric matrix corresponding to a Voigt vector.

    Parameters
    ----------
    vector: array-like
        Vector with six elements representing a matrix in Voigt notation.

    Returns 
    -------
    inverse: numpy.ndarray
        Vector with six elements representing the inverse matrix.

    """ 
    matrix = convert_voigt_vector_to_matrix(vector)
    inverse = np.linalg.inv(matrix)
    return convert_matrix_to_voigt_vector(inverse)


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




def bootstrap_block_matrix(matrix, block_size=3, num_samples=100, rng=None):
    """Resample a block matrix.

    This function is useful for performing a bootstrap.
    """
    if rng is None:
        rng = np.random.default_rng()
    num_blocks = matrix.shape[0] // block_size
    row_tiling = np.tile(np.arange(block_size, dtype=int), num_blocks)
    samples = np.zeros((num_samples, matrix.shape[0], matrix.shape[1]), dtype=matrix.dtype)
    for i in range(num_samples):
        sampled_blocks = rng.integers(num_blocks, size=num_blocks)
        rows = np.repeat(sampled_blocks, 3, axis=0) + row_tiling
        samples[i, :, :] = matrix[rows, :]
    return samples


class Reshape(ABC):
    """Reshape a vector into a grid."""

    def vector_to_grid(self):
        raise NotImplementedError

    def grid_to_vector(self):
        raise NotImplementedError


class OrderedVectorToSquareGrid(Reshape):

    def __init__(self, edge_length):
        """Transform between an ordered vector and a square grid."""
        self.grid_shape = (edge_length, edge_length)

    def vector_to_grid(self, x):
        return np.reshape(x, (-1, self.grid_shape[1]))

    def grid_to_vector(self, x):
        return np.ravel(x)
