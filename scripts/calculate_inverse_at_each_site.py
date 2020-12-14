#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
from textwrap import dedent
import numpy as np
np.set_printoptions(precision=1)
from surface_stiffness.matrix import calculate_blockwise_inverse

def main():
    args = parse_command_line()
    greens_functions = np.load(args.greens_functions)
    stiffness = calculate_blockwise_inverse(greens_functions)
    np.save(args.output_file, stiffness)


def parse_command_line():
    parser = argparse.ArgumentParser(
        description=dedent("""\
            Invert elastic greens functions to obtain stiffnesses.
        """
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "greens_functions",
        help=dedent("""\
            Elastic Greens functions stored as numpy array with shape
            ((N*N*3), (N*N*3)). We assume that the atoms are arranged
            in a two-dimensional simple cubic lattice, where N is the
            number of atoms along one edge. Each 3×3 subblock of the
            input array contains the Greens functions for one pair of
            atoms, or one point in the Brillouin zone. This script
            will invert every 3x3 Greens function matrix to obtain a
            stiffness matrix. Note that one could average the stiffness
            matrices over sites N, but this average would not be the
            correct average within stochastic homogenization. Rather,
            one should average the Greens functions and then invert
            them to find the correct average. The stiffnesses computed
            here are meant to check site-by-site fluctuations of the
            stiffness.
            """
        )
    )
    parser.add_argument("output_file", help="Output file (numpy .npy-file)")
    return parser.parse_args()


if __name__ == "__main__":
    main()
