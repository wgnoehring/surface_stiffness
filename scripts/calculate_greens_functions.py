#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import numpy as np
from pathlib import Path
from scipy.sparse import load_npz
import time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path_to_hessian",
        type=pathlib.Path,
        help="Path to the file containing the Hessian matrix (block sparse row format).",
    )
    parser.add_argument(
        "num_atoms_per_plane", type=int, help="Number of atoms per plane"
    )
    parser.add_argument(
        "num_free_planes", type=int, help="Number of free planes"
    )
    args = parser.parse_args()
    sparse_hessian = load_npz(args.ath_to_hessian.as_posix())
    inverse, dense calculate_surface_greens_functions(sparse_hessian, num_atoms_per_plane, num_free_planes)
    path_inverse = Path(
        args.path_to_hessian.parent,
        f"block_minimized_hessian_matrix_inverse_dense_upper{num_atoms_per_plane}x3x{num_atoms_per_plane}x3_block.npy",
    )
    np.save(path_inverse.as_posix(), inverse)
    path_dense = Path(
        args.path_to_hessian.parent,
        f"block_minimized_hessian_matrix_dense_upper{num_atoms_per_plane}x3x{num_atoms_per_plane}x3_block.npy",
    )
    np.save(path_dense.as_posix(), dense)


def calculate_surface_greens_functions(sparse_hessian, num_atoms_per_plane, num_free_planes):
    """Calculate surface Green's functions by Hessian inversion.

    Parameters
    ----------
    sparse_hessian : scipy.sparse.bsr_matrix
        Hessian matrix
    num_atoms_per_plane : int
        Number of atoms per plane
    num_free_planes : int
        Number of free planes

    Returns
    -------
    inverse : numpy.ndarray
        :code:`3*num_atoms_per_plane` by :code:`3*num_atoms_per_plane` block
        of the inverse of the upper `3*num_atoms_per_plane*num_free_planes` by
        :code:`3*num_atoms_per_plane*num_free_planes` block of :code:`hessian`
    dense : 
        :code:`3*num_atoms_per_plane` by :code:`3*num_atoms_per_plane`
        :code:`block of hessian` as dense matrix
    """
    dense = sparse.todense()
    # Remove entries corresponding to frozen atoms
    imax = int(num_atoms_per_plane * num_free_planes * 3)
    dense = dense[:imax, :imax]
    # remove rows and columns corresponding to fixed atoms
    start = time.time()
    inverse = np.linalg.inv(dense)
    end = time.time()
    print(f"Time for matrix inversion: {end-start:.1f}")
    inverse = inverse[: num_atoms_per_plane * 3, : num_atoms_per_plane * 3]
    dense = dense[: num_atoms_per_plane * 3, : num_atoms_per_plane * 3]
    return inverse, dense


if __name__ == "__main__":
    main()
