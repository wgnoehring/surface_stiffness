#!/usr/bin/env python
# -*- coding: utf-8 -*-
from textwrap import dedent
import importlib
import argparse
import numpy as np
from surface_stiffness import configurations
from surface_stiffness.matrix import (
    fourier_transform_symmetric_square_block_matrix,
    calculate_blockwise_inverse,
    load_atomistic_stiffness,
    invert_grid_of_flattened_matrices,
)

def main():
    parser = argparse.ArgumentParser(
        description=dedent("""\
            Invert elastic greens functions to obtain stiffnesses.
        """
        ),
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=dedent("""\
            The stiffness matrices are saved to *.npy-files. If the input array provided
            by 'greens_functions' has shape ((N*N*3), (N*N*3)), then the output array(s)
            will have shape ((N*3), (N*3)), because the average across sites is computed
            before inversion.
            """
        )

    )
    parser.add_argument(
        "greens_functions",
        help=dedent("""\
            Elastic Greens functions stored as numpy array with shape
            ((N*N*3), (N*N*3)). We assume that the atoms are arranged
            in a two-dimensional simple cubic lattice, where N is the
            number of atoms along one edge. Each 3×3 subblock of the
            input array contains the Greens functions for one pair of
            atoms, or one point in the Brillouin zone.
            """
        )
    )
    parser.add_argument(
        "-e",
        "--element-wise",
        metavar="XYZ_FILE",
        help=(
            dedent("""\
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
        "-c",
        "--confidence-interval",
        type=int,
        default=0,
        help=(dedent("""\
            Calculate bounds by adding/substracting multiples of the
            standard deviations of the real and imaginary parts of the
            greens functions before inversion.
            """
            )
        ),
    )
    args = parser.parse_args()
    greens_functions = np.load(args.greens_functions)
    degrees_of_freedom = 3
    num_atoms_edge = int(np.sqrt(greens_functions.shape[0] // degrees_of_freedom))
    # The number of subsurface planes and the surface
    # width are irrelevant here and can be set to 1
    config = configurations.Configuration(
        material=None, crystal=configurations.FCCSurface001(num_atoms_edge, 1, 1.0),
    )
    mean_stiff, upper_stiff, lower_stiff = calculate_stiffness(
        greens_functions, config, num_stddev=args.confidence_interval, mask=None
    )
    filename = f"./stiffness_from_average_of_greens_functions.npy"
    print(f"...writing average stiffnesses to {filename}")
    np.save(filename, np.ma.filled(mean_stiff))
    if args.confidence_interval > 0:
        filename = f"./stiffness_from_average_of_greens_functions_confidence_interval_plus_{args.confidence_interval}_stddev.npy"
        print(f"...writing upper confidence limit of stiffness to {filename}")
        np.save(filename, np.ma.filled(upper_stiff))
        filename = f"./stiffness_from_average_of_greens_functions_confidence_interval_minus_{args.confidence_interval}_stddev.npy"
        print(f"...writing lower confidence limit of stiffness to {filename}")
        np.save(filename, np.ma.filled(lower_stiff))

    if args.element_wise is not None:
        mask_for_symbol, symbols = config.crystal.create_symbol_masks_for_surface(
            args.element_wise, return_symbols=True,
        )
        unique_symbols = np.unique(symbols)
        for symbol in unique_symbols:
            sutf = symbol.decode("UTF-8")
            print(f"Calculating stiffness for sites with {sutf} atoms")
            mask = mask_for_symbol[symbol].ravel()
            num_masked = len(np.nonzero(mask)[0])
            print(
                f"{num_masked} masked values --> {config.crystal.num_atoms_surface-num_masked} not masked"
            )
            mean_stiff, upper_stiff, lower_stiff = calculate_stiffness(
                greens_functions, config, num_stddev=args.confidence_interval, mask=mask
            )
            filename = f"./stiffness_from_average_of_greens_functions_{sutf}_atoms.npy"
            print(f"...writing average stiffnesses of {sutf} atoms to {filename}")
            np.save(filename, np.ma.filled(mean_stiff))
            if args.confidence_interval is not None:
                filename = f"./stiffness_from_average_of_greens_functions_{sutf}_atoms_confidence_interval_plus_{args.confidence_interval}_stddev.npy"
                print(f"...writing upper confidence limit of stiffnesses of {sutf} atoms to {filename}")
                np.save(filename, np.ma.filled(upper_stiff))
                filename = f"./stiffness_from_average_of_greens_functions_{sutf}_atoms_confidence_interval_minus_{args.confidence_interval}_stddev.npy"
                print(f"...writing lower confidence limit of stiffnesses of {sutf} atoms to {filename}")
                np.save(filename, np.ma.filled(lower_stiff))


def calculate_stiffness(greens_functions, config, num_stddev=0, mask=None):
    r"""Calculate stiffness by inversion of elastic Greens functions

    We assume that the atoms are arranged in a two-dimensional simple cubic
    lattice. Let :math:`N` be the number of atoms along the edge. The input
    array :code:`greens_functions` contains the elastic Greens functions.
    Each :math:`3\times{}3` block contains the Greens functions of one pair
    of sites and, corresponding to one point in the Brillouin zone. Holding
    one site fixed, we can obtain data for all points in the Brillouin zone
    by varying the other site. Varying the first site, we obtain :math:`N^2`
    :math:`3\times{}3` matrices for each Brillouin zone point. This function
    first calculates the average at each Brillouin zone point and then inverts
    the :math:`3\times{}3` matrices to obtain stiffness matrices.

    The :math:`N^2` :math:`3\times{}3` matrices at a given Brillouin zone
    point take the same values in case of a pure crytal. In the case of 
    an alloy (or some other disorder), the values differ. We can compute
    a confidence interval of the variation in stiffness by adding and 
    subtracting a multiple of the standard deviation of the Green's 
    functions before inversion. Since the Greens functions are complex-valued,
    we use the following approach. Noting that the variance of a complex
    number is :math:`Var[z] = Var[Re(z)] + Var[Im(z)]`, where :math:`Re`
    and :math:`Im` indicate the real and imaginary parts, respectively, 
    we compute :math:`z +- m(\sigma(Re(z)) + i * \sigma(Im(z)))`, where
    :math:`\sigma` is the standard deviation, and :math:`m` is the number
    of standard deviations to add.

    Parameters
    ----------
    greens_functions: array-like
        Array with shape :code:`((N*N*3), (N*N*3))`, where each block of shape
        :code:`(3,3)` contains the Greens functions for one atomic site.
    config: surface_stiffness.configurations.Configuration
        Configuration of crystal and material, which provides the information
        of how to reshape a row in greens_functions to obtain an array of shape
        :code:`(N,N)`, whose components are arranged according to the order of
        the corresponding sites in space, or the corresponding wavevectors in
        reciprocal space.
    num_stddev: int
        Number of standard deviations to add/subtract from the average Greens
        functions to obtain a confidence interval.
    mask: array-like
        Array of ones and zeros, or None. If None, then the average of the 
        Greens functions runs over all sites. If not None, then the average
        runs only over sites with a one in :code:`mask`.

    Returns
    -------
    mean_stiff: numpy.ndarray
        :code:`((N*3), (N*3))` array containing the inverses of the
        average Greens functions. Each block of shape :code:`(3,3)`
        the inverse of the corresponding block in the :code:`((N*3),(N*3))` 
        of average Greens functions.
    upper_stiff: numpy.ndarray
        Upper confidence limit, array with the same shape as :code:`mean_stiff`
    lower_stiff: numpy.ndarray
        Lower confidence limit, array with the same shape as :code:`mean_stiff`

    """
    if num_stddev < 0:
        raise ValueError("number of standard deviations for confidence interval must be greater or equal to zero")
    mean_both, = load_atomistic_stiffness(
        greens_functions,
        reshape=config.crystal.reshape,
        statistics=[np.average],
        part="both",
        indexing="matrix",
        mask=mask,
    )
    mean_stiff = invert_grid_of_flattened_matrices(mean_both)
    upper_stiff = lower_stiff = None
    if num_stddev > 0:
        std_both, = load_atomistic_stiffness(
            greens_functions,
            reshape=config.crystal.reshape,
            statistics=[np.std],
            part="both",
            indexing="matrix",
            mask=mask,
        )
        mean_real, std_real = load_atomistic_stiffness(
            greens_functions,
            reshape=config.crystal.reshape,
            statistics=[np.average, np.std],
            part="real",
            indexing="matrix",
            mask=mask,
        )
        mean_imag, std_imag = load_atomistic_stiffness(
            greens_functions,
            reshape=config.crystal.reshape,
            statistics=[np.average, np.std],
            part="imag",
            indexing="matrix",
            mask=mask,
        )
        fluctuation = std_real + 1.0j * std_imag
        upper = mean_both + fluctuation
        lower = mean_both - fluctuation
        upper_stiff = invert_grid_of_flattened_matrices(upper)
        lower_stiff = invert_grid_of_flattened_matrices(lower)
    return mean_stiff, upper_stiff, lower_stiff


if __name__ == "__main__":
    main()
