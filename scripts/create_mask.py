#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
from textwrap import dedent
import numpy as np

np.set_printoptions(precision=1)
from surface_stiffness.configurations import FCCSurface001


def main():
    args = parse_command_line()
    symbols = read_symbols_from_xyz(
        args.xyz_file, num_atoms=args.num_atoms, check_sorted=True
    )
    mask = np.array([s == args.symbol for s in symbols], dtype=bool)
    np.save(args.output_file, mask)


def parse_command_line():
    parser = argparse.ArgumentParser(
        description=dedent(
            """\
            Read the first N atoms from an xyz file and create
            an array of booleans, which indicates whether
            the atoms belong to a particular species or not.

            It is assumed that the xyz-file contains the coordinates
            of a configuration with a free surface. The surface atoms
            form a simple cubic 2D lattice (e.g. {100} surfaces in the
            face-centered cubic lattice), and the first N atoms in the
            file are the surface atoms. Thus, the boolean array tells
            which surface sites are occupied by the desired element.
        """
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("xyz_file", help="XYZ configuration file with atomic symbols")
    parser.add_argument(
        "num_atoms",
        type=int,
        help=(
            "Number of atoms that will be read from the file"
        ),
    )
    parser.add_argument("symbol", help="Atomic symbol to search for.")
    parser.add_argument(
        "output_file",
        help="Boolean array will be saved to this file (numpy .npy-format)",
    )
    return parser.parse_args()


def read_symbols_from_xyz(file, num_atoms, check_sorted=True):
    """Read the first num_atoms atomic symbols from an xyz_file.

    Parameters
    ==========
    file: str
        xyz-file
    num_atoms: int
    check_sorted: bool
        check that atoms are sorted according to increasing
        atom ID and that atom IDs are consecutive

    Returns
    =======
    symbols : list
        atomic symbols
    """
    symbols = []
    with open(file) as f:
        f.readline()
        f.readline()
        prev_index = -1
        for i in range(num_atoms):
            line = f.readline()
            words = line.strip().split()
            index = int(words[0]) - 1  # atom IDs start at 1
            symbols.append(words[1])
            if check_sorted:
                if index - prev_index != 1:
                    raise ValueError("atom IDs not increasing or consecutive")
            prev_index = index
        return symbols


if __name__ == "__main__":
    main()
