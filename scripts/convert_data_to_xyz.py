#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert a Lammps data file to xyz-format

Convert the data file to an xyz file for later calculation
of the Hessian matrix. The Hessian will be calculated using
ASE. Since ASE cannot read fictitious atom types, the
name of a real element is required. Use Rn in this case.

Note: non-orthongonal cells are currently not supported.
"""
import sys
import argparse
import numpy as np
import re
import logging

__author__ = "Wolfram Georg Nöhring"
__copyright__ = "Copyright 2019, Uni Freiburg"
__license__ = "GNU General Public License"
__email__ = "wolfram.noehring@imtek.uni-freiburg.de"

logger = logging.getLogger("surface_stiffness.scripts.convert_data_to_xyz")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "infile", type=str, help="Input configuration (LAMMPS data file)"
    )
    parser.add_argument(
        "outfile",
        type=str,
        help="Output file name (XYZ format, " + "will be overwritten if it exists)",
    )
    parser.add_argument(
        "type_map",
        type=str,
        nargs="+",
        help="Mapping between numeric types and element names",
    )
    args = parser.parse_args()
    # Get path of input file and seed for RNG
    data = to_list(args.infile)
    # Extract header and section information
    logger.info("Reading input configuration")
    header_section_or_comment = re.compile(".*?[a-zA-Z](?!\+|-)")
    comment = re.compile("^\#")
    header_keywords = ["atoms", "atom types", "xlo xhi", "ylo yhi", "zlo zhi"]
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
    line_offset = section_info["Atoms"][0] + 2
    natoms = int(header_info["atoms"][1])
    ntypes = int(header_info["atom types"][1])
    del data[line_offset + natoms :]
    if len(args.type_map) != ntypes:
        raise ValueError("wrong number of elements in type map")
    element_name_for_type = {(i + 1): args.type_map[i] for i in range(ntypes)}
    dx = np.diff(np.array(header_info["xlo xhi"][1:], dtype=float))[0]
    dy = np.diff(np.array(header_info["ylo yhi"][1:], dtype=float))[0]
    dz = np.diff(np.array(header_info["zlo zhi"][1:], dtype=float))[0]
    header = f'Lattice="{dx:.16e} 0.0 0.0 0.0 {dy:.16e} 0.0 0.0 0.0 {dz:.16e}" Properties=id:I:1:species:S:1:pos:R:3\n'
    # Sort by atom identifier
    atoms_section = data[line_offset : line_offset + natoms]
    atoms_section.sort(key=lambda line: int(line.split()[0]))
    data[line_offset : line_offset + natoms] = atoms_section
    # Write the outfile
    with open(args.outfile, "w") as file:
        file.write(f"{natoms}\n")
        file.write(header)
        for line in data[line_offset:]:
            words = line.split()
            numeric_type = int(words[1])
            element_type = element_name_for_type[numeric_type]
            file.write(f"{words[0]} {element_type} {words[2]} {words[3]} {words[4]}\n")


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
