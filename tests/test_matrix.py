#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import unittest
import numpy as np
from surface_stiffness import matrix

class TestReshape(unittest.TestCase):

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


if __name__ == "__main__":
    unittest.main()
