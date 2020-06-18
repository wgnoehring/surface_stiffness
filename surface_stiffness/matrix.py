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


def extract_local_stiffness(stiff, atom_index, voigt_index, reshape, part="real"):
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


class Reshape(object):
    """Convert array row to grid"""

    def __init__(self, vector_to_grid, grid_to_vector):
        self.vector_to_grid = vector_to_grid
        self.grid_to_vector = grid_to_vector
