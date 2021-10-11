#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Functions for working with matrices."""
import sys
from abc import ABC
import numpy as np
from numpy import ma
from pathlib import Path

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
    (0, 0): 0,
    (1, 1): 1,
    (2, 2): 2,
    (1, 2): 3,
    (2, 1): 3,
    (0, 2): 4,
    (2, 0): 4,
    (0, 1): 5,
    (1, 0): 5,
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
    assert (
        matrix.ndim == 2
        and matrix.shape[0] == matrix.shape[1]
        and matrix.shape[0] % block_size == 0
    )
    num_blocks = int(matrix.shape[0] / block_size)
    print(
        f"input matrix for FFT is partitioned into {num_blocks}x{num_blocks} blocks of size {block_size}x{block_size}"
    )
    for block_index in range(num_blocks):
        # sys.stdout.write(f'taking FFT of block column {block_index+1}\r')
        # sys.stdout.flush()
        # We make no assumption about symmetry of 3x3 blocks
        for i in range(block_size):
            for j in range(block_size):
                row = block_size * block_index + i
                values_on_grid = reshape.vector_to_grid(matrix[row, j::block_size])
                values_on_grid = np.roll(
                    values_on_grid, -(block_index // (int(np.sqrt(num_blocks)))), axis=0
                )
                values_on_grid = np.roll(values_on_grid, -block_index, axis=1)
                traffo_on_grid = np.fft.fftshift(np.fft.fft2(values_on_grid))
                matrix_fft[row, j::block_size] = reshape.grid_to_vector(traffo_on_grid)
    sys.stdout.write("\n")
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
            # sys.stdout.write(f'inverting block {i*num_blocks+j+1} out of {num_blocks*num_blocks}\r')
            # sys.stdout.flush()
            slice_i = slice(block_size * i, block_size * (i + 1), 1)
            slice_j = slice(block_size * j, block_size * (j + 1), 1)
            blockwise_inverse[slice_i, slice_j] = np.linalg.inv(
                matrix[slice_i, slice_j]
            )
    sys.stdout.write("\n")
    return blockwise_inverse


def extract_local_stiffness(
    stiff, atom_index, indices, reshape, part="real", indexing="voigt"
):
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


def load_atomistic_stiffness(
    stiff,
    reshape,
    statistics=None,
    atom_index=0,
    part="real",
    mask=None,
    indexing="voigt",
):
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
    tmp = extract_local_stiffness(stiff, 0, 0, reshape, part=part)
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
            # print(f"calculating {op.__name__}")
            stat = zeros(
                (reshape.grid_shape[0], reshape.grid_shape[1], num_indices),
                dtype=required_dtype,
            )
            for ii in range(num_indices):
                variables = zeros(
                    (reshape.grid_shape[0], reshape.grid_shape[1], num_atoms),
                    dtype=required_dtype,
                )
                if mask is not None:
                    variables.mask = zeros(variables.shape)
                for atom_index in range(num_atoms):
                    variables[:, :, atom_index] = extract_local_stiffness(
                        stiff,
                        atom_index,
                        indices[ii],
                        reshape,
                        part=part,
                        indexing=indexing,
                    )
                    if mask is not None:
                        variables.mask[:, :, atom_index] = mask[atom_index]
                stat[:, :, ii] = op(variables, axis=2)
            output.append(stat)
        return tuple(output)
    else:
        print(f"extracting data for atom index {atom_index}")
        arr = zeros(
            (reshape.grid_shape[0], reshape.grid_shape[1], num_indices),
            dtype=required_dtype,
        )
        for ii in range(num_indices):
            arr[:, :, ii] = extract_local_stiffness(
                stiff, atom_index, indices[ii], reshape, part=part, indexing=indexing
            )
        return (arr,)


def histogram_stiffness(
    stiff, reshape, indices, part="real", num_bins=100, mask=None, indexing="voigt"
):
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
        (reshape.grid_shape[0], reshape.grid_shape[1], num_atoms),
        dtype=float,
    )
    histograms = zeros(
        (reshape.grid_shape[0], reshape.grid_shape[1], num_bins), dtype=float
    )
    bin_edges = zeros(
        (reshape.grid_shape[0], reshape.grid_shape[1], num_bins + 1), dtype=float
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
            inverse[i, j, :] = invert(array[i, j, :])
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


def bootstrap_block_matrix(matrix, block_size=3, num_samples=100, rng=None, roll=False):
    """Resample a block matrix.

    This function is useful for performing a bootstrap.
    """
    if rng is None:
        rng = np.random.default_rng()
    num_blocks = matrix.shape[0] // block_size
    row_tiling = np.tile(np.arange(block_size, dtype=int), num_blocks)
    samples = np.zeros(
        (matrix.shape[0], matrix.shape[1], num_samples), dtype=matrix.dtype
    )
    original_blocks = np.arange(num_blocks)
    # may need to roll left/right so blocks ij with i==j occupy diagonal
    for i in range(num_samples):
        sampled_blocks = block_size * rng.integers(
            low=0, high=num_blocks, size=num_blocks
        )
        rows = np.repeat(sampled_blocks, 3, axis=0) + row_tiling
        samples[:, :, i] = matrix[rows, :]
        if roll:
            required_left_rolls = np.repeat(original_blocks - sampled_blocks, 3, axis=0)
            for j in range(samples.shape[0]):
                samples[j, :, i] = np.roll(
                    samples[j, :, i], shift=required_left_rolls[j]
                )
    return samples


class Reshape(ABC):
    """Reshape a vector into a grid."""

    def vector_to_grid(self):
        raise NotImplementedError

    def grid_to_vector(self):
        raise NotImplementedError


class OrderedVectorToRectangularGrid(Reshape):
    def __init__(self, edge_length_x, edge_length_y):
        """Transform between an ordered vector and a rectangular grid."""
        self.grid_shape = (edge_length_x, edge_length_y)

    def vector_to_grid(self, x):
        return np.reshape(x, (-1, self.grid_shape[1]))

    def grid_to_vector(self, x):
        return np.ravel(x)


class OrderedVectorToSquareGrid(OrderedVectorToRectangularGrid):
    def __init__(self, edge_length):
        """Transform between an ordered vector and a square grid."""
        super().__init__(edge_length, edge_length)


class BlockMatrixStatistics(object):

    """
    Class that has access to a collection of block matrices and
    can calculate statistics.

    Suppose we have ``M`` arrays of shape ``(N,N)``, where each array
    can be partitioned into ``Nb×Nb`` blocks of shape ``(Sb,Sb)``. In
    each of the ``M`` arrays, there are then ``Nb`` 'block rows' of
    shape ``(Sb, N)``. This class implements methods to calculate
    statistics of block rows without loading all arrays into memory.

    Say we want to calculate the mean of all block rows in all ``M`` arrays.
    A simple solution would be to stack the ``M*Nb`` block rows to obtain an
    array of shape ``(Sb, N, M*Nb)``. However, this would require a lot of
    memory. For example, if ``N=2883`` and ``M=500``, then we would need ca. 33 GB.

    This class divides each of the ``M`` array into slices along the
    second (column) dimension, and then calculates the statistics
    slice by slice. Let ``nc`` be the number of columns in a slice. The
    columns from the ``M`` arrays can be stacked to generate an array
    ``(Sb, nc, M*Nb)``. By calculating the statistic along the third
    dimension, we obtain the partial statistic of shape ``(Sb, nc)``. By
    doing this calculation for all slices and joining the resulting
    ``(Sb, nc)`` arrays, we obtain the full statistic of shape ``(Sb, N)``.

    For example, if ``N=2883``, ``Sb=3, and ``M=500``, and we do not want to
    use more than 1 GB or RAM, then ``M*(Nb*Sb)*nc*X<=1GB``, where ``X`` is
    the size of an array element (``np.zeros(1,dtype=A.dtype).nbytes``
    for input array ``A``), hence ``nc<= 1GB/X/M/Nb/Sb``.

    Consider another complication: suppose we are additionally given
    ``M`` masks ``K`` for the block rows in the corresponding arrays. Each
    mask is a bool array of shape ``(Nb,)``, where element ``K[i]`` tells
    whether block row ``i`` should be taken into account or not. Let
    ``Nk`` be the number of ``True`` values in some ``K``. Then we should
    first select the ``(Nk*Sb, nc)`` sub-array before proceeding.

    Parameters
    ----------
    paths: list
        List of paths to files from which to read. Each file should contain a
        numpy ndarray of shape ``(N, N)``. ``N`` must be the same for all files.
    block_size : int
        Size of the square blocks in each block matrix
    max_bytes :  int
        Partition the block matrices into column slices and work on these
        slices, so that the concatenation of slices from all input files
        does not consume more memory than ``max_bytes`` bytes
    block_mask_for_path : dict
        Dictionary with the files in paths as keys. The values are block masks,
        i.e. numpy arrays of shape ``(Nb,)``, whose members are either True or
        False. If ``block_mask_for_path[file[i]][j]==False``, then the i-th block
        row in the j-th file will not be included in the statistics.

    Attributes
    ----------
    paths: list
        List of paths to files from which to read. Each file should contain a
        numpy ndarray of shape ``(N, N)``. ``N`` must be the same for all files.
    block_size : int
        Size of the square blocks in each block matrix
    max_bytes :  int
        Partition the block matrices into column slices and work on these
        slices, so that the concatenation of slices from all input files
        does not consume more memory than ``max_bytes`` bytes
    block_mask_for_path : dict
        Dictionary with the files in paths as keys. The values are block masks,
        i.e. numpy arrays of shape ``(Nb,)``, whose members are either True or
        False. If ``block_mask_for_path[file[i]][j]==False``, then the i-th block
        row in the j-th file will not be included in the statistics.
    max_columns : int
        Maximum number of columns that can be extracted from each
        array so that the concatenation of data from all arrays
        does not consume more than ``max_bytes`` bytes of memory
    column_bin_edges : list
        Partitioning of the array along the second (column)
        dimension into work arrays. Work array ``i`` runs from
        ``column_bin_edges[i]:column_bin_edges[i+1]``.
    num_block_rows : int
        Number of block rows in each file; will be determined by reading
        the first file.
    bytes_per_var : int
        Number of bytes per array entry, should be 8 in case of standard
        numpy float arrays, and 16 in case of standard complex arrays.

    Examples
    --------

    Check that statistics are calculated correctly without masking.

    >>> import numpy as np
    >>> from tempfile import NamedTemporaryFile
    >>> from os import remove
    >>> np.set_printoptions(precision=2, linewidth=80)
    >>> block_size = 3
    >>> num_blocks = 33
    >>> A = np.random.rand(num_blocks * block_size, num_blocks * block_size)
    >>> tempfile = NamedTemporaryFile(delete=False)
    >>> np.save(tempfile, A)
    >>> tempfile.close()
    >>> M = 20
    >>> mock_path_list = [tempfile.name] * M
    >>> calculator = BlockMatrixStatistics(mock_path_list, block_size, max_bytes=1e8)
    >>> # Calculate the mean and the variance across all M arrays,
    >>> # which should be the same as the block row statistics
    >>> # of one array, since all arrays are the same.
    >>> results = calculator.calculate_statistics((np.mean, np.var))
    >>> x = np.load(mock_path_list[0])
    >>> x = np.split(x, x.shape[0]//block_size)
    >>> x = np.dstack(x)
    >>> mean_one = np.mean(x, axis=2)
    >>> var_one = np.var(x, axis=2)
    >>> assert(np.allclose(results[np.mean], mean_one))
    >>> assert(np.allclose(results[np.var], var_one))
    >>> remove(tempfile.name)

    Check that statistics are calculated correctly with masking.

    >>> import numpy as np
    >>> from tempfile import NamedTemporaryFile
    >>> from os import remove
    >>> np.set_printoptions(precision=2, linewidth=80)
    >>> block_size = 3
    >>> num_blocks = 33
    >>> A = np.random.rand(2 * num_blocks * block_size, num_blocks * block_size)
    >>> # Later we will calculate the statistics of every even or every odd block row of A
    >>> tempfile = NamedTemporaryFile(delete=False)
    >>> np.save(tempfile, A)
    >>> tempfile.close()
    >>> M = 20
    >>> mock_path_list = [tempfile.name] * M
    >>> block_row_indices = np.arange(2 * num_blocks, dtype=int)
    >>> odd_block_row = block_row_indices%2
    >>> even_block_row = np.logical_not(odd_block_row)
    >>> # Calculate mean and variance of even block rows
    >>> block_mask_for_path = {p: even_block_row for p in mock_path_list}
    >>> calculator = BlockMatrixStatistics(
    >>>     mock_path_list,
    >>>     block_size,
    >>>     max_bytes=1e8,
    >>>     block_mask_for_path=block_mask_for_path
    >>> )
    >>> results = calculator.calculate_statistics((np.mean, np.var))
    >>> x = np.load(mock_path_list[0])
    >>> x = np.split(x, x.shape[0]//block_size)
    >>> x = np.dstack(x)
    >>> x = x[:, :, even_block_row]
    >>> mean_one = np.mean(x, axis=2)
    >>> var_one = np.var(x, axis=2)
    >>> assert(np.allclose(results[np.mean], mean_one))
    >>> assert(np.allclose(results[np.var], var_one))
    >>> # Calculate mean and variance of odd block rows
    >>> block_mask_for_path = {p: odd_block_row for p in mock_path_list}
    >>> calculator = BlockMatrixStatistics(
    >>>     mock_path_list,
    >>>     block_size,
    >>>     max_bytes=1e8,
    >>>     block_mask_for_path=block_mask_for_path
    >>> )
    >>> results = calculator.calculate_statistics((np.mean, np.var))
    >>> x = np.load(mock_path_list[0])
    >>> x = np.split(x, x.shape[0]//block_size)
    >>> x = np.dstack(x)
    >>> x = x[:, :, odd_block_row.astype(bool)]
    >>> mean_one = np.mean(x, axis=2)
    >>> var_one = np.var(x, axis=2)
    >>> assert(np.allclose(results[np.mean], mean_one))
    >>> assert(np.allclose(results[np.var], var_one))
    >>> remove(tempfile.name)
    """

    def __init__(
        self,
        paths: list,
        block_size: int,
        max_bytes: int = int(1e9),
        block_mask_for_path: dict = {},
    ):
        self.paths = [Path(p) for p in paths]
        for path in self.paths:
            if not path.is_file():
                raise ValueError(f"{path} not a file")
        self.block_size = block_size
        self.max_bytes = max_bytes
        x = np.load(paths[0])
        if x.ndim != 2:
            raise ValueError("input arrays must have two dimensions")
        if x.shape[0] % block_size:
            raise ValueError(
                f"input array cannot be partitioned into blocks of "
                "size {block_size} along first dimension"
            )
        self.bytes_per_var = x[0, 0].nbytes
        self.num_rows = x.shape[0]
        self.num_block_rows = self.num_rows // self.block_size
        self.num_columns = x.shape[1]
        self._calculate_max_columns()
        self.row_selection_for_path = dict()
        for path in block_mask_for_path.keys():
            block_mask = block_mask_for_path[path]
            row_mask = np.repeat(block_mask, self.block_size)
            (self.row_selection_for_path[path],) = row_mask.nonzero()

    def calculate_statistics(self, statistics):
        """Calculate block row statistics.

        Calculate statistics of block rows of the block
        matrices in ``paths``.

        Parameters
        ----------
        statistics : list
            List of functions that calculate statistics, e.g. ``numpy.mean``

        Returns
        -------
        results : list
            List of statistics. Given ``M`` paths to block matrices of shape ``(Nb*Sb,Nb*Sb)``
            and ``Ns=len(statistics)``, the list will contain ``Ns`` arrays of shape
            ``(Nb, Nb*Sb)``.

        """
        results = {s: [] for s in statistics}
        for i in range(len(self.column_bin_edges) - 1):
            block_rows = []
            for path in self.paths:
                block_rows.extend(
                    self._load_columns(
                        path, self.column_bin_edges[i], self.column_bin_edges[i + 1]
                    )
                )
            block_rows = np.dstack(block_rows)
            for s in statistics:
                results[s].append(s(block_rows, axis=2))
        for stat, res in results.items():
            results[stat] = np.hstack(res)
        return results

    def _calculate_max_columns(self):
        """Calculate maximum number of block columns to load from each file."""
        max_num_variables = self.max_bytes // self.bytes_per_var
        max_variables_per_file = max_num_variables // len(self.paths)
        self.max_columns = max_variables_per_file // self.num_rows
        if self.max_columns == 0:
            raise ValueError(
                f"cannot perform calculation with max_bytes={self.max_bytes}"
            )
        imax = self.num_columns // self.max_columns
        if imax * self.max_columns == self.num_columns:
            # even partitioning
            self.column_bin_edges = np.arange(
                0, self.num_columns + 1, self.max_columns, dtype=int
            )
            self.column_bin_edges[-1] += 1  # to include last column
        else:
            self.column_bin_edges = np.r_[
                np.arange(0, self.num_columns, self.max_columns, dtype=int),
                self.num_columns + 1,
            ]

    def _load_columns(self, path, column_start, column_end):
        x_mmap = np.load(path, mmap_mode="r")
        x = x_mmap[:, column_start:column_end].copy()
        del x_mmap
        if x.shape[0] % self.block_size:
            raise ValueError(
                f"input array cannot be partitioned into blocks of "
                "size {self.block_size} along first dimension"
            )
        key = str(path)
        if key in self.row_selection_for_path.keys():
            x = np.take(x, self.row_selection_for_path[key], axis=0)
        return np.vsplit(x, x.shape[0] // self.block_size)
