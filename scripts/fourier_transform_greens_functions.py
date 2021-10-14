#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
from textwrap import dedent
import argparse
import logging
import numpy as np

np.set_printoptions(precision=1)
from scipy.sparse import load_npz
from surface_stiffness.matrix import (
    fourier_transform_symmetric_square_block_matrix,
    OrderedVectorToRectangularGrid
)

__author__ = "Wolfram Georg Nöhring"
__copyright__ = "Copyright 2020, Uni Freiburg"
__license__ = "GNU General Public License"
__email__ = "wolfram.noehring@imtek.uni-freiburg.de"

logger = logging.getLogger(
    "surface_stiffness.scripts.fourier_transform_greens_functions"
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "greens_functions",
        help="Numpy array containing the Green's functions. Alternatively: scipy sparse bsr matrix",
    )
    parser.add_argument(
        "-f",
        "--input_format",
        choices=("numpy", "sparse"),
        default="numpy",
        help="Input format: 'numpy' if the file should be loaded with numpy.load, 'sparse' if it should be loaded with scipy.sparse.load_npz",
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
    parser.add_argument(
        "-g",
        "--grid_shape",
        type=int,
        nargs=2,
        help=dedent(
            """\
            Number of sites along the x- and y-directions. 
            If this option is not set, it will be assumed 
            that the grid is square, and the dimensions 
            will be inferred from the shape of the greens
            functions array.
            """
        ),
    )
    args = parser.parse_args()
    if args.input_format == "numpy":
        greens_functions = np.load(args.greens_functions)
    elif args.input_format == "sparse":
        sparse = load_npz(args.greens_functions)
        greens_functions = sparse.todense()
    else:
        raise ValueError
    # Green's functions calculated with PetSC inversion
    # may be padded with zeros along the first dimension
    num_cols = greens_functions.shape[1]
    greens_functions = greens_functions[:num_cols, :]
    if not args.grid_shape:
        num_atoms_edge = int(np.sqrt(greens_functions.shape[0] // args.block_size))
        logger.info(f"Setting up reshape for grid size ({num_atoms_edge}, {num_atoms_edge})")
        reshape = OrderedVectorToRectangularGrid(num_atoms_edge, num_atoms_edge)
    else:
        logger.info(f"Setting up reshape for grid size {args.grid_shape}")
        reshape = OrderedVectorToRectangularGrid(*args.grid_shape)
    ft_greens_functions = fourier_transform_symmetric_square_block_matrix(
        greens_functions, reshape
    )
    np.save(args.output, ft_greens_functions)


if __name__ == "__main__":
    main()
