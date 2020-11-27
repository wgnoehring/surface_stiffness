#!/usr/bin/python3
import sys
import gzip
import argparse
import numpy as np
import time
from ase import io
import matscipy
from matscipy import calculators
from scipy.sparse import bsr_matrix, save_npz

__author__ = "Wolfram Georg Nöhring"
__copyright__ = "Copyright 2020, Uni Freiburg"
__license__ = "GNU General Public License"
__email__ = "wolfram.noehring@imtek.uni-freiburg.de"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "configuration", help="Atomic configuration in xyz format understood by ASE"
    )
    parser.add_argument(
        "potential", help="Embedded Atom Method interatomic potential table"
    )
    parser.add_argument(
        "output",
        default="block_minimized_hessian_matrix.npz",
        help="Output file containing the Hessian in BSR format",
    )
    atoms = io.read(args.configuration, format="xyz")
    calculator = calculators.eam.calculator.EAM(args.potential)
    atoms.set_calculator(calculator)
    atoms.set_pbc([True, True, False])
    start = time.time()
    hessian = calculator.calculate_hessian_matrix(atoms, divide_by_masses=False)
    end = time.time()
    print("Time for hessian matrix construction: {:f}".format(end - start))
    print("Number of atoms: {:d}".format(len(atoms)))
    print("Dimension of hessian matrix: {:d}x{:d}".format(*hessian.get_shape()))
    save_npz(args.output, hessian)


if __name__ == "__main__":
    main()