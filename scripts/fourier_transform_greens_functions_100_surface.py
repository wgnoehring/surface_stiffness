#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import argparse
import numpy as np

np.set_printoptions(precision=1)
from scipy.sparse import load_npz
from surface_stiffness import materials, configurations
from surface_stiffness.matrix import (
    fourier_transform_symmetric_square_block_matrix,
    calculate_blockwise_inverse,
)

__author__ = "Wolfram Georg Nöhring"
__copyright__ = "Copyright 2020, Uni Freiburg"
__license__ = "GNU General Public License"
__email__ = "wolfram.noehring@imtek.uni-freiburg.de"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "greens_functions", help="Numpy array containing the Green's functions"
    )
    parser.add_argument(
        "output",
        default="fourier_transformed_greens_functions.npy",
        help="Output array containig the Fourier transforms",
    )
    parser.add_argument(
        "-b",
        "--block_size",
        type=int,
        default=3,
        help="Size of blocks in the Green's function matrix",
    )
    args = parser.parse_args()
    greens_functions = np.load(args.greens_functions)
    # Green's functions calculated with PetSC inversion
    # may be padded with zeros along the first dimension
    num_cols = greens_functions.shape[1]
    greens_functions = greens_functions[:num_cols, :]
    # Determine the number of atoms along the edge of the configuration
    num_atoms_edge = int(greens_functions.shape[0] / args.block_size)
    # Create a dummy material to access the reshape method of
    # configurations.crystal. Height and side length of the
    # configuration do not matter here. The material does not matter.
    dummy_config = configurations.Configuration(
        None, configurations.FCCSurface001(num_atoms_edge, 1, 1.0),  #
    )
    ft_greens_functions = fourier_transform_symmetric_square_block_matrix(
        greens_functions, average_config.crystal.reshape
    )
    np.save(args.output, ft_greens_functions)


if __name__ == "__main__":
    main()
