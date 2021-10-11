#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import unittest
import numpy as np
from surface_stiffness import matrix

class TestMatrix(unittest.TestCase):

    @classmethod
    def create_test_block_matrix(cls, Nx, Ny, cutoff_radius=4, block_prefactors=np.ones((3, 3))):
        """Create a matrix for testing methods working on Hessian block matrices.

        Consider a `(Nx × Ny)` grid of sites  labeled as follows:
        ```
        | [ 4 3 2 1 0]
        | [ 9 8 7 6 5]
        | ...
        | [N-1 ...   ]
        | 
        | y
        | 🠕
        |  → x

        We asume that the grid is periodic and construct a block matrix by
        computing for each pair of sites I, J `A * np.cos(dx / cutoff_radius
        * 0.5 * pi) * np.cos(dx / cutoff_radius * 0.5 * pi)`, where dx and
        dx, respectively, are the site distances along x and y, and A is a
        `block_size × block_size` prefactor matrix.

        Parameters
        ==========
        Nx : int
            Number of sites along the x-direction
        Ny : int
            Number of sites along the y-direction
        cutoff_radius : int
            Interactions are cut off at this site distance. Must be less than half of 
            `Nx` and `Ny`.
        blocK_prefactors: numpy.ndarray
            Square 2D array with prefactors for the blocks.

        Returns
        =======
        h : numpy.ndarray
            Test matrix with dimensions `[Nx * Ny * block_prefactors.shape[0], Nx * Ny * block_prefactors.shape[0]]`
        """
        if block_prefactors.ndim != 2 or (block_prefactors.shape[0] != block_prefactors.shape[1]):
            raise ValueError("block prefactors must be a square 2D array")
        block_size = block_prefactors.shape[0]
        if (Nx < 2 * cutoff_radius) or (Ny < 2 * cutoff_radius):
            raise ValueError("grid size in x and y must be larger than twice the cutoff radius")
        num_sites = Nx * Ny
        n = block_size * num_sites
        hessian = np.zeros((n, n), float)
        for site_1 in range(num_sites):
            for site_2 in range(num_sites):
                row_1 = site_1 // Nx
                row_2 = site_2 // Nx
                col_1 = Nx - 1 - site_1%Nx
                col_2 = Nx - 1 - site_2%Nx
                dx = col_2 - col_1
                dy = row_2 - row_1
                # minimum-image conventions
                if (dx >   Nx // 2): dx = dx - Nx
                if (dx <= -Nx // 2): dx = dx + Nx
                if (dy >   Ny // 2): dy = dy - Ny
                if (dy <= -Ny // 2): dy = dy + Ny
                within_radius = (abs(dx) < cutoff_radius) & (abs(dy) < cutoff_radius)
                f = within_radius * (
                    np.cos(0.5 * np.pi * dx / cutoff_radius) *
                    np.cos(0.5 * np.pi * dy / cutoff_radius)
                )
                # find index of pair into hessian matrix
                for i in range(block_size):
                    for j in range(block_size):
                        hessian_row = site_1 * block_size + i
                        hessian_col = site_2 * block_size + j
                        hessian[hessian_row, hessian_col] = block_prefactors[i, j] * f
        return hessian

    @staticmethod
    def test_square_reshape():
        array_1D = np.arange(0, 16, dtype=int)
        array_2D_square = np.array(
            ((0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11), (12, 13, 14, 15)), int
        )
        array_2D_rect = np.array(
            ((0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15)), int
        )
        reshape = matrix.OrderedVectorToSquareGrid(4)
        x = reshape.vector_to_grid(array_1D)
        np.testing.assert_allclose(x, array_2D_square)

        reshape = matrix.OrderedVectorToRectangularGrid(2, 8)
        x = reshape.vector_to_grid(array_1D)
        np.testing.assert_allclose(x, array_2D_rect)

    @classmethod
    def test_fourier_transform(self):
        """Test that Fourier transforms of all block rows of test matrix are identical"""
        block_size = 3
        block_prefactors = np.ones((block_size, block_size))
        Nx = Ny = 11
        cutoff_radius = 5
        h = TestMatrix.create_test_block_matrix(Nx, Ny, cutoff_radius, block_prefactors)
        reshape = matrix.OrderedVectorToSquareGrid(Nx)
        ft = matrix.fourier_transform_symmetric_square_block_matrix(h, reshape, block_size)
        for i in range(block_size):
            for j in range(block_size):
                block_index = 0
                ft1 = ft[block_size * block_index + i, j::block_size]
                for block_index in range(2, Nx * Ny):
                    ft2 = ft[block_size * block_index + i, j::block_size]
                    diff = ft2 - ft1
                    np.testing.assert_allclose(diff, 0 + 0j)

        Nx = 15
        Ny = 10
        cutoff_radius = 5
        h = TestMatrix.create_test_block_matrix(Nx, Ny, cutoff_radius, block_prefactors)
        reshape = matrix.OrderedVectorToRectangularGrid(Nx, Ny)
        ft = matrix.fourier_transform_symmetric_square_block_matrix(h, reshape, block_size)
        for i in range(block_size):
            for j in range(block_size):
                block_index = 0
                ft1 = ft[block_size * block_index + i, j::block_size]
                for block_index in range(2, Nx * Ny):
                    ft2 = ft[block_size * block_index + i, j::block_size]
                    diff = ft2 - ft1
                    np.testing.assert_allclose(diff, 0 + 0j)

        Nx = 10
        Ny = 15
        cutoff_radius = 5
        h = TestMatrix.create_test_block_matrix(Nx, Ny, cutoff_radius, block_prefactors)
        reshape = matrix.OrderedVectorToRectangularGrid(Nx, Ny)
        ft = matrix.fourier_transform_symmetric_square_block_matrix(h, reshape, block_size)
        for i in range(block_size):
            for j in range(block_size):
                block_index = 0
                ft1 = ft[block_size * block_index + i, j::block_size]
                for block_index in range(2, Nx * Ny):
                    ft2 = ft[block_size * block_index + i, j::block_size]
                    diff = ft2 - ft1
                    np.testing.assert_allclose(diff, 0 + 0j)


if __name__ == "__main__":
    unittest.main()
