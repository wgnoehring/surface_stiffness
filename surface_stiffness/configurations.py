#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Configurations as a combination of site and material information"""
from abc import ABC
import logging
from dataclasses import dataclass, field
import numpy as np
from .materials import Material
from .matrix import OrderedVectorToRectangularGrid
from importlib import import_module

# TODO: method to generate Lammps data files

logger = logging.getLogger("surface_stiffness.configurations")


def _read_atoms_from_file(file, format="extxyz"):
    """Read atoms from file and sort by identifier

    Requires the Atomic Simulation Environment (ASE) [1]_.

    Parameters
    ----------
    file : str
        file in a format understood by ASE
    format : str
        file format

    Returns
    -------
    atoms : object

    References
    ----------
    .. [1] https://gitlab.com/ase/ase
    """
    ase_io = import_module("ase.io")
    atoms = ase_io.read(file, format=format)
    identifiers = atoms.get_array("id")
    identifiers = np.array(identifiers, int)
    order = np.argsort(identifiers)
    return atoms[order]


class Crystal(ABC):
    """Abstract base with common methods for crystals"""

    @staticmethod
    def generate_rotation_matrix(x, y, z):
        """Generate a rotation matrix.

        Generate the rotation matrix whose rows are the crystal
        directions that are parallel to the local coordinate system.

        Parameters
        ----------

        x: array-like
            Components of the x-axis in the rotated coordinate system
        y: array-like
            Components of the y-axis in the rotated coordinate system
        z: array-like
            Components of the z-axis in the rotated coordinate system

        Examples
        --------
        >>> x = np.array((1, 1, 0))
        >>> y = np.array((-1, 1, 0))
        >>> z = np.array((0, 0, 1))
        >>> generate_rotation_matrix(x, y, z)
        [[ 0.71  0.71  0.  ]
         [-0.71  0.71  0.  ]
         [ 0.    0.    1.  ]]

        """
        R = np.zeros((3, 3), dtype=float)
        R[0, :] = x / np.linalg.norm(x)
        R[1, :] = y / np.linalg.norm(y)
        R[2, :] = z / np.linalg.norm(z)
        assert np.isclose(np.linalg.det(R), 1.0)
        return R


@dataclass
class FCCSurface001(Crystal):
    """Represents a slab with a (001) surface.

    Stores geometric information about a face-centered cubic slab with
    orientation :math:`x=[110]`, :math:`y=[-110]`, and :math:`z=[001]`.
    It is assumed that the slab has the following properties. The
    surface in z-direction is free, while the x- and y-directions are
    periodic. The surface atoms form a simple square lattice. The
    atom identifiers are strictly ordered. Atom 1 has the maximum
    coordinate in x, y, and z-direction. The coordinates decrease with
    increasing identifier. The x-coordinate decreases fastest, then the
    y-coordinate, and finally the z-coordinate. The x- and y-coordinate
    wrap around at periodic boundaries at x=0 and y=0, respectively.

    Attributes
    ----------

    num_atoms_edge: int
        Number of atoms :math:`N_e` along an edge in x- and y-direction.
    num_atoms_surface: int
        Number of surface atoms
    num_subsurface_planes: int
        Number of planes below the surface, excluding substrate planes
    true_edge_length: float
        True length of the cell in x- and y-direction
    thickness: float
        Thickness of the configuration in z-direction,
        considering surface and subsurface planes
    """

    num_atoms_edge: int
    num_atoms_surface: int = field(init=False)
    num_subsurface_planes: int
    true_edge_length: float = None
    thickness: float = None

    x = np.array((1.0, 1.0, 0.0))
    y = np.array((-1.0, 1.0, 0.0))
    z = np.array((0.0, 0.0, 1.0))
    rotation_matrix = Crystal.generate_rotation_matrix(x, y, z)

    def __post_init__(self):
        self.num_atoms_surface = self.num_atoms_edge ** 2
        self.reshape = OrderedVectorToRectangularGrid(
            self.num_atoms_edge, self.num_atoms_edge
        )

    def calculate_area_per_atom(self):
        """Calculate the mean area per atom from the true edge length."""
        return self.true_edge_length ** 2 / float(self.num_atoms_surface)

    def calculate_unrelaxed_thickness(self, lattice_parameter):
        """Calculate the theoretical thickness in z-direction.

        Parameters
        ----------
        lattice_parameter : float
            FCC lattice parameter

        Returns
        -------
        thickness : float
        """
        num_planes_considered = 1.0 + self.num_subsurface_planes
        return num_planes_considered * lattice_parameter / 2.0

    def measure_thickness(self, file):
        """Calculate the thickness in z-direction.

        Load atomic coordinates from a file and determine the thickness
        in z-direction of the surface planes plus the subsurface planes.

        Parameters
        ----------
        file : str

        Returns
        -------
        thickness : float
        """
        atoms = _read_atoms_from_file(file)
        z = atoms.get_positions()[:, 2]
        num_planes_considered = 1.0 + self.num_subsurface_planes
        max_num_atoms = int(self.num_atoms_surface * num_planes_considered)
        z_surf = np.mean(z[0 : self.num_atoms_surface])
        z_last = np.mean(z[max_num_atoms - self.num_atoms_surface : max_num_atoms])
        thickness = z_surf - z_last
        return thickness

    def create_symbol_masks_for_surface(self, file, return_symbols=False):
        """Create array masks according to chemical symbol.

        Read atoms from file, extract surface atoms, and generate a
        mask for each chemical symbol, which masks sites not occupied
        by the given element in a :math:`N_e\timesN_e` array of sites.

        Parameters
        ----------
        file : str

        Returns
        -------
        mask_for_symbol : dict
            chemical symbols and masks as key-value pairs
        """

        atoms = _read_atoms_from_file(file)
        symbols = atoms.get_chemical_symbols()
        symbols = np.array(symbols, dtype="S2")
        symbols = symbols[: self.num_atoms_surface]
        symbols = self.reshape.vector_to_grid(symbols)
        unique_symbols = np.unique(symbols)
        mask_for_symbol = {}
        for s in unique_symbols:
            mask_for_symbol[s] = np.array(symbols != s, dtype=int)
        if return_symbols:
            return mask_for_symbol, symbols
        return mask_for_symbol

    def draw_surface_site_occupation(self, file):
        """Draw occupation of surface sites.

        For each element create a pixel art image which
        indicates the surface sites occupied by this element.

        Parameters
        ----------
        file : str
        """
        mask_for_symbol, symbols = self.create_symbol_masks_for_surface(
            file, return_symbols=True
        )
        symbol_lengths = [len(s) for s in mask_for_symbol.keys()]
        max_length = max(symbol_lengths)
        header = ("+" + "-" * max_length) * self.num_atoms_edge + "+"
        divider = "|" + "-" * (len(header) - 2) + "|"
        for s in mask_for_symbol.keys():
            print(f"{s.decode()} sites:")
            print(header)
            for i in range(self.num_atoms_edge):
                unmasked = np.logical_not(mask_for_symbol[s][i, :])
                symbol_row = symbols[i, :]
                strings = [
                    f"|{s.decode()}" if m else f"|{' '*max_length}"
                    for s, m in zip(symbol_row, unmasked)
                ]
                print("".join(strings) + "|")
            print(header)


@dataclass
class FCCSurface011(Crystal):
    """Represents a slab with a (001) surface.

    Stores geometric information about a face-centered cubic slab with
    orientation :math:`x=[100]`, :math:`y=[01-1]`, and :math:`z=[011]`.
    It is assumed that the slab has the following properties. The
    surface in z-direction is free, while the x- and y-directions are
    periodic. The surface atoms form a simple rectangular lattice. The
    atom identifiers are strictly ordered. Atom 1 has the maximum
    coordinate in x, y, and z-direction. The coordinates decrease with
    increasing identifier. The x-coordinate decreases fastest, then the
    y-coordinate, and finally the z-coordinate. The x- and y-coordinate
    wrap around at periodic boundaries at x=0 and y=0, respectively.

    Attributes
    ----------

    num_atoms_edge_x: int
        Number of atoms :math:`N_e` along an edge in the x-direction
    num_atoms_edge_y: int
        Number of atoms :math:`N_e` along an edge in the y-direction
    num_atoms_surface: int
        Number of surface atoms
    num_subsurface_planes: int
        Number of planes below the surface, excluding substrate planes
    true_edge_length_x: float
        True length of the cell in the x-direction
    true_edge_length_y: float
        True length of the cell in the y-direction
    thickness: float
        Thickness of the configuration in z-direction,
        considering surface and subsurface planes
    """

    num_atoms_edge_x: int
    num_atoms_edge_y: int
    num_atoms_surface: int = field(init=False)
    num_subsurface_planes: int
    true_edge_length_x: float = None
    true_edge_length_y: float = None
    thickness: float = None

    x = np.array((1.0, 0.0, 0.0))
    y = np.array((0.0, 1.0, -1.0))
    z = np.array((0.0, 1.0, 1.0))
    rotation_matrix = Crystal.generate_rotation_matrix(x, y, z)

    def __post_init__(self):
        self.num_atoms_surface = self.num_atoms_edge_x * self.num_atoms_edge_y
        self.reshape = OrderedVectorToRectangularGrid(
            self.num_atoms_edge_x, self.num_atoms_edge_y
        )

    def calculate_area_per_atom(self):
        """Calculate the mean area per atom from the true edge length."""
        return (
            self.true_edge_length_x
            * self.true_edge_length_y
            / float(self.num_atoms_surface)
        )

    def calculate_unrelaxed_thickness(self, lattice_parameter):
        """Calculate the theoretical thickness in z-direction.

        Parameters
        ----------
        lattice_parameter : float
            FCC lattice parameter

        Returns
        -------
        thickness : float
        """
        num_planes_considered = 1.0 + self.num_subsurface_planes
        return num_planes_considered * lattice_parameter / 2.0 / np.sqrt(2.0)

    def measure_thickness(self, file):
        """Calculate the thickness in z-direction.

        Load atomic coordinates from a file and determine the thickness
        in z-direction of the surface planes plus the subsurface planes.

        Parameters
        ----------
        file : str

        Returns
        -------
        thickness : float
        """
        atoms = _read_atoms_from_file(file)
        z = atoms.get_positions()[:, 2]
        num_planes_considered = 1.0 + self.num_subsurface_planes
        max_num_atoms = int(self.num_atoms_surface * num_planes_considered)
        z_surf = np.mean(z[0 : self.num_atoms_surface])
        z_last = np.mean(z[max_num_atoms - self.num_atoms_surface : max_num_atoms])
        thickness = z_surf - z_last
        return thickness

    def create_symbol_masks_for_surface(self, file, return_symbols=False):
        """Create array masks according to chemical symbol.

        Read atoms from file, extract surface atoms, and generate a
        mask for each chemical symbol, which masks sites not occupied
        by the given element in a :math:`N_e\timesN_e` array of sites.

        Parameters
        ----------
        file : str

        Returns
        -------
        mask_for_symbol : dict
            chemical symbols and masks as key-value pairs
        """

        atoms = _read_atoms_from_file(file)
        symbols = atoms.get_chemical_symbols()
        symbols = np.array(symbols, dtype="S2")
        symbols = symbols[: self.num_atoms_surface]
        symbols = self.reshape.vector_to_grid(symbols)
        unique_symbols = np.unique(symbols)
        mask_for_symbol = {}
        for s in unique_symbols:
            mask_for_symbol[s] = np.array(symbols != s, dtype=int)
        if return_symbols:
            return mask_for_symbol, symbols
        return mask_for_symbol


@dataclass
class Configuration:
    """Represents a crystal configuration."""

    material: Material
    crystal: Crystal
