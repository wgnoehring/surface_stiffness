#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Change atom IDs according to position

Assign new IDs based on the positions of the atoms. Sort atoms by z, x, and
y-position (descending), and then assign new identifiers starting from one.
"""
import sys
import argparse
import numpy as np
import pandas as pd

# from pandas.compat import StringIO
from io import StringIO
import re

__author__ = "Wolfram Georg Nöhring"
__copyright__ = "Copyright 2019, Uni Freiburg"
__license__ = "GNU General Public License"
__email__ = "wolfram.noehring@imtek.uni-freiburg.de"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "infile", type=str, help="Input configuration (LAMMPS data file)"
    )
    parser.add_argument(
        "outfile",
        type=str,
        help="Output file name (Will be a LAMMPS data file. "
        + "Will be overwritten if it exists)",
    )
    args = parser.parse_args()

    # Get path of input file and seed for RNG
    data = to_list(args.infile)

    # Extract header and section information
    print("Reading input configuration")
    header_section_or_comment = re.compile(".*?[a-zA-Z](?!\+|-)")
    comment = re.compile("^\#")
    header_keywords = ["atoms", "atom types"]
    header_info = {}
    section_keywords = ["Masses", "Atoms"]
    section_info = {}
    for line_number, line in enumerate(data):
        if header_section_or_comment.match(line):
            # Strip comments
            words = line.rstrip().split()
            comment_pos = [i for i, word in enumerate(words) if comment.match(word)]
            if comment_pos:
                words = words[0 : comment_pos[0]]
            is_float = [read_as_float(w) for w in words]
            if all(is_float):
                continue
            # Divide into numbers and words
            first_string_pos = is_float.index(False)
            # Join strings to form the keyword
            keyword = " ".join(words[first_string_pos:])
            if keyword in header_keywords:
                header_info[keyword] = [line_number] + words[:first_string_pos]
            else:
                section_info[keyword] = [line_number] + words[:first_string_pos]

    natoms = int(header_info["atoms"][1])

    # Sort the atoms according to their z-, x-, and y-position
    line_offset = section_info["Atoms"][0] + 2
    header = "id type x y z ix iy iz\n"
    atoms_section = pd.read_csv(
        StringIO(header + "".join(data[line_offset : line_offset + natoms])), sep=" "
    )
    # Need to round positions to avoid bad sorting due to insignificant digits
    decimals = 10
    atoms_section.x = atoms_section.x.apply(lambda v: round(v, decimals))
    atoms_section.y = atoms_section.y.apply(lambda v: round(v, decimals))
    atoms_section.z = atoms_section.z.apply(lambda v: round(v, decimals))
    atoms_section.sort_values(
        by=["z", "y", "x"], ascending=[False, False, False], inplace=True
    )
    # atoms_section = numpy_based_lexsort_with_rounding(atoms_section)
    atoms_section.id = np.arange(1, atoms_section.shape[0] + 1, dtype=int)
    atoms_section = atoms_section.to_string(
        index=False, index_names=False, float_format=lambda x: f"{x:.16e}"
    )
    atoms_section = atoms_section.split("\n")
    del atoms_section[0]  # header
    data[line_offset : line_offset + natoms] = [x + "\n" for x in atoms_section]
    del data[line_offset + natoms :]

    # Write the outfile
    with open(args.outfile, "w") as file:
        file.writelines(data)


def numpy_based_lexsort_with_rounding(df):
    arr = df.values
    arr[:, 2:5] = np.round(arr[:, 2:5], 12)
    order = np.lexsort((arr[:, 2], arr[:, 3], arr[:, 4]))[-1::-1]
    return pd.DataFrame(arr[order], columns=df.columns)


def read_as_float(my_obj):
    """Check if an object can be converted to a number
    See also http://stackoverflow.com/q/354038
    """
    try:
        float(my_obj)
        return True
    except ValueError:
        return False


def to_list(infile):
    """ Reads a file and returns the contents as a list of lists."""
    with open(infile, "r") as file:
        list_of_lines = file.readlines()
    return list_of_lines


if __name__ == "__main__":
    main()
