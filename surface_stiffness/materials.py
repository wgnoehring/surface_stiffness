#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from dataclasses import dataclass

@dataclass
class Material:
    elements: list 
    concentrations: tuple
    lattice_parameter: float
    C11: float
    C12: float
    C44: float
    source: str = None

    def __post_init__(self):
        if not np.isclose(np.sum(self.concentrations), 1.0):
            raise ValueError("concentrations must sum to 1")

    def calculate_zener_ratio(self) -> float:
        return 2.0 * self.C44 / (self.C11 - self.C12)

#Average-alloy elastic constants, calculated with ELASTIC script
average_FeNiCr_stiffened = Material(
    ("X",), (1.0,), 
    3.52181864516065, 246.9299588857714, 147.0719468074990, 125.0305675876400,
)

random_FeNiCr_stiffened = Material(
    ("Fe", "Ni", "Cr"), (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0), 
    3.52136543384631, 242.92730469, 157.03243264, 134.89972062
)
