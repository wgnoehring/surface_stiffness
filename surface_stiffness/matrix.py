#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Functions for working with matrices."""
import numpy as np
from numpy import ma
from .units import eva3_to_gpa, gpa_to_eva3

matrix_indices_for_voigt_index = {
    0: (0, 0),
    1: (1, 1),
    2: (2, 2),
    3: (1, 2),
    4: (0, 2),
    5: (0, 1),
}
"""Indices in Voigt notation for matrix indices as key-value pairs."""

label_for_component = {0: "xx", 1: "yy", 2: "zz", 3: "yz", 4: "xz", 5: "xy"}
"""Subscript strings for the indices of a vector in Voigt notation as key-value pairs"""


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


def extract_local_stiffness(stiff, atom_index, voigt_index, reshape, part="real"):
    """Extract stiffness for a specific site.

    Parameters
    ----------
    stiff: array-like
       Matrix which for a system of :math:`N\\times{}N` atoms has shape :math:`3N\\times{}3N` and
       contains :math:`N\\times{}N` :math:`3\\times 3`stifness matrices for pairs of atoms.
    atom_index: int
        Index of the central site. Stiffness components with this site as first site are selected.
    voigt_index: int
        Index in voigt notation of the component to extract
        in each selected :math:`3\\times{}3` stiffness matrix
    reshape: Reshape
    part: string 
        'real' for real part or 'imag' for imaginary part of the complex stiffness

    Returns
    -------
    components: numpy.ndarray
        :math:`N\\times{}N` array of stiffness components.
    """
    block_size = 3
    i, j = matrix_indices_for_voigt_index[voigt_index]
    row = block_size * atom_index + i
    row = stiff[row, j::block_size]
    if part == "real":
        return np.real(reshape.vector_to_grid(row))
    elif part == "imag":
        return np.imag(reshape.vector_to_grid(row))
    else:
        raise NotImplementedError


def load_atomistic_stiffness(path, statistics=None, atom_index=0, part="real", mask=None):
    """Load atomistic stiffness from a file 

    The result still needs to be divided by the area per atom.

    Parameters
    ----------
    path: string 
        Path to the stiffness matrix
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
    part: "real" or "imag"
        Whether to extract the real or imaginary parts

    Returns
    -------
    arr: tuple of arrays of shape (N, N, 6)
        N is the number of atoms along the edge of the configuration.
        If Ns statistics were requested, then the tuple contains the
        resulting Ns arrays. If no statistics were requested, then
        the tuple contains only the array for atom index atom_index.

    """
    stiff = np.load(path) * eva3_to_gpa
    assert stiff.ndim == 2 and stiff.shape[0] == stiff.shape[1]
    num_atoms_surface = int(stiff.shape[0] / 3)
    num_atoms_edge = int(np.sqrt(num_atoms_surface))
    reshape = Reshape(
        lambda x: np.reshape(x, (-1, num_atoms_edge)), lambda x: np.ravel(x)
    )
    if mask is not None:
        zeros = ma.zeros 
    else:
        zeros = np.zeros
    if statistics is not None:
        output = []
        for op in statistics:
            print(f"calculating {op.__name__} of data in {path}")
            stat = zeros((num_atoms_edge, num_atoms_edge, 6), dtype=float)
            for voigt_index in range(6):
                variables = zeros(
                    (num_atoms_edge, num_atoms_edge, num_atoms_surface), dtype=float,
                )
                if mask is not None:
                    variables.mask = zeros(variables.shape)
                for atom_index in range(num_atoms_surface):
                    variables[:, :, atom_index] = extract_local_stiffness(
                        stiff, atom_index, voigt_index, reshape, part=part
                    )
                    if mask is not None:
                        variables.mask[:, :, atom_index] = mask[atom_index]
                #if mask is not None:
                #    with np.printoptions(linewidth=220):
                #        print(variables[0, 0, :20].mask)
                stat[:, :, voigt_index] = op(variables, axis=2)
            output.append(stat)
        return tuple(output)
    else:
        print(f"extracting data for atom index {atom_index} from {path}")
        arr = zeros((num_atoms_edge, num_atoms_edge, 6), dtype=float)
        for voigt_index in range(6):
            arr[:, :, voigt_index] = extract_local_stiffness(
                stiff, atom_index, voigt_index, reshape, part=part
            )
        return (arr,)


def histogram_stiffness(path, voigt_index, part="real", num_bins=100, mask=None):
    """Generate histograms of stiffness components.

    Parameters
    ----------
    path: string 
        Path to the stiffness matrix
    voigt_index: int
        Index in voigt notation of the component of the 
        :math:`3\\times{}3` stiffness matrix 
    part: string 
        'real' for real part or 'imag' for imaginary part of the complex stiffness
    num_bins: int
        Number of histogram bins
    mask: array-like
        Masks values in the stiffness matrix

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
    stiff = np.load(path) * eva3_to_gpa
    assert stiff.ndim == 2 and stiff.shape[0] == stiff.shape[1]
    num_atoms_surface = int(stiff.shape[0] / 3)
    num_atoms_edge = int(np.sqrt(num_atoms_surface))
    reshape = Reshape(
        lambda x: np.reshape(x, (-1, num_atoms_edge)), lambda x: np.ravel(x)
    )

    if mask is not None:
        zeros = ma.zeros 
    else:
        zeros = np.zeros
    variables = zeros(
        (num_atoms_edge, num_atoms_edge, num_atoms_surface), dtype=float,
    )
    histograms = zeros(
        (num_atoms_edge, num_atoms_edge, num_bins), dtype=float
    )
    bin_edges = zeros(
        (num_atoms_edge, num_atoms_edge, num_bins+1), dtype=float
    )
    if mask is not None:
        variables.mask = zeros(variables.shape)

    for atom_index in range(num_atoms_surface):
        variables[:, :, atom_index] = extract_local_stiffness(
            stiff, atom_index, voigt_index, reshape, part=part
        )
        if mask is not None:
            variables.mask[:, :, atom_index] = mask[atom_index]
    for (i, j) in np.ndindex((num_atoms_edge, num_atoms_edge)):
        histograms[i, j, :], bin_edges[i, j, :] = np.histogram(variables[i, j, :], bins=num_bins, density=True)
    return histograms, bin_edges


class Reshape(object):
    """Convert array row to grid"""

    def __init__(self, vector_to_grid, grid_to_vector):
        self.vector_to_grid = vector_to_grid
        self.grid_to_vector = grid_to_vector
