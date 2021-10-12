#!/usr/bin/env python
# -*- coding: utf-8 -*-
from textwrap import dedent
import argparse
import logging
import numpy as np
from surface_stiffness.matrix import OrderedVectorToRectangularGrid
from surface_stiffness.stiffness import calculate_stiffness

logger = logging.getLogger(
    "surface_stiffness.scripts.calculate_stiffness_from_greens_function_average"
)


def main():
    args = parse_command_line()
    if args.confidence_interval != "stddev":
        num_stddev = 0
    else:
        num_stddev = args.num_stddev
    # The average stiffness will be calculated in any case.
    greens_functions = np.load(args.greens_functions)
    if not args.grid_shape:
        degrees_of_freedom = 3
        num_atoms_edge = int(np.sqrt(greens_functions.shape[0] // degrees_of_freedom))
        reshape = OrderedVectorToRectangularGrid(num_atoms_edge, num_atoms_edge)
    else:
        logger.info(f"Setting up reshape for grid size {args.grid_shape}")
        reshape = OrderedVectorToRectangularGrid(*args.grid_shape)
    mean_stiff, upper_stiff, lower_stiff = calculate_stiffness(
        greens_functions, reshape, num_stddev=num_stddev, mask=None
    )
    filename = f"./stiffness_from_average_of_greens_functions.npy"
    logger.info(f"...writing average stiffnesses to {filename}")
    np.save(filename, np.ma.filled(mean_stiff))

    if args.confidence_interval == "stddev":
        filename = f"./stiffness_from_average_of_greens_functions_confidence_interval_plus_{args.confidence_interval}_stddev.npy"
        logger.info(f"...writing upper confidence limit of stiffness to {filename}")
        np.save(filename, np.ma.filled(upper_stiff))
        filename = f"./stiffness_from_average_of_greens_functions_confidence_interval_minus_{args.confidence_interval}_stddev.npy"
        logger.info(f"...writing lower confidence limit of stiffness to {filename}")
        np.save(filename, np.ma.filled(lower_stiff))

    if args.element_wise is not None:
        mask_for_symbol, symbols = config.crystal.create_symbol_masks_for_surface(
            args.element_wise,
            return_symbols=True,
        )
        unique_symbols = np.unique(symbols)
        for symbol in unique_symbols:
            sutf = symbol.decode("UTF-8")
            logger.info(f"Calculating stiffness for sites with {sutf} atoms")
            mask = mask_for_symbol[symbol].ravel()
            num_masked = len(np.nonzero(mask)[0])
            logger.info(
                f"{num_masked} masked values --> {config.crystal.num_atoms_surface-num_masked} not masked"
            )
            mean_stiff, upper_stiff, lower_stiff = calculate_stiffness(
                greens_functions, reshape, num_stddev=num_stddev, mask=mask
            )
            filename = f"./stiffness_from_average_of_greens_functions_{sutf}_atoms.npy"
            logger.info(f"...writing average stiffnesses of {sutf} atoms to {filename}")
            np.save(filename, np.ma.filled(mean_stiff))
            if args.confidence_interval == "stddev":
                filename = f"./stiffness_from_average_of_greens_functions_{sutf}_atoms_confidence_interval_plus_{args.confidence_interval}_stddev.npy"
                logger.info(
                    f"...writing upper confidence limit of stiffnesses of {sutf} atoms to {filename}"
                )
                np.save(filename, np.ma.filled(upper_stiff))
                filename = f"./stiffness_from_average_of_greens_functions_{sutf}_atoms_confidence_interval_minus_{args.confidence_interval}_stddev.npy"
                logger.info(
                    f"...writing lower confidence limit of stiffnesses of {sutf} atoms to {filename}"
                )
                np.save(filename, np.ma.filled(lower_stiff))


def parse_command_line():
    parser = argparse.ArgumentParser(
        description=dedent(
            """\
            Invert elastic greens functions to obtain stiffnesses.
        """
        ),
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=dedent(
            """\
            The stiffness matrices are saved to *.npy-files. If the input array provided
            by 'greens_functions' has shape ((N*N*3), (N*N*3)), then the output array(s)
            will have shape ((N*3), (N*3)), because the average across sites is computed
            before inversion.
            """
        ),
    )
    parser.add_argument(
        "greens_functions",
        help=dedent(
            """\
            Elastic Greens functions stored as numpy array with shape
            ((N*N*3), (N*N*3)). We assume that the atoms are arranged
            in a two-dimensional simple cubic lattice, where N is the
            number of atoms along one edge. Each 3×3 subblock of the
            input array contains the Greens functions for one pair of
            atoms, or one point in the Brillouin zone.
            """
        ),
    )
    parser.add_argument(
        "-e",
        "--element-wise",
        metavar="XYZ_FILE",
        help=(
            dedent(
                """\
            If this argument is provided, the script calculates the
            average stiffness per element, meaning: for all elements
            'e', calculate the average over all sites occupied by 'e'.
            If parameter 'confidence-intervals' is set, confidence
            intervals will be calculated as well for each element. Site
            occupations are determined by reading the file %(metavar)s,
            which should be in the xyz format understood by ASE, see
            https://wiki.fysik.dtu.dk/ase/ase/io/io.html
            """
            )
        ),
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
    # Arguments for particular confidence interval types
    subparsers = parser.add_subparsers(
        dest="confidence_interval",
        help=(
            dedent(
                """\
            Method for calculating confidence intervals. If none is
            specified, then only the mean stiffness will be calculated.
            """
            )
        ),
    )
    none = subparsers.add_parser(
        "none",
        help=(
            dedent(
                """\
            Calculate no confidence intervals
            """
            )
        ),
    )
    stddev = subparsers.add_parser(
        "stddev",
        help=(
            dedent(
                """\
            Calculate confidence intervals by adding/substracting
            multiples of the standard deviations of the real and
            imaginary parts of the greens functions before inversion.
            """
            )
        ),
    )
    stddev.add_argument(
        "-n",
        "--num_stddev",
        type=int,
        default=0,
        help="Number of standard deviations around the mean",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
