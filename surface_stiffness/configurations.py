#!/usr/bin/env python
# -*- coding: utf-8 -*-
from .materials import Material

@dataclass
class Surface100:
    num_atoms_edge : int
    num_subsurface_planes : int
    true_edge_length : float 
    thickness_surface_to_last_subsurface : float 
    material : Material

    def calculate_number_of_surface_atoms(self) -> int:
        return self.num_atoms_edge ** 2

    def calculate_area_per_atom(self) -> float:
        return self.true_edge_length**2 / float(self.calculate_number_of_surface_atoms())

#TODO: method to generate Lammps data files
