#!/usr/bin/env python
# -*- coding: utf-8 -*-
from dataclasses import dataclass

@dataclass
class Material:
    lattice_parameter: float
    C11: float
    C12: float
    C44: float

    def calculate_zener_ratio(self) -> float:
        return 2.0 * self.C44 / (self.C11 - self.C12)

average_FeNiCr_stiffened = Material(
    3.52181864516065, 246.9299588857714, 147.0719468074990, 125.0305675876400
)

random_FeNiCr_stiffened = Material(
    3.52136543384631, 242.92730469, 157.03243264, 134.89972062
)
