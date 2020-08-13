#!/usr/bin/env python
# -*- coding: utf-8 -*-
from abc import ABC
import numpy as np
from dataclasses import dataclass

@dataclass
class Material(ABC):
    """Describe a material with computed properties.

    Attributes
    ----------
    elements: list
        Chemical elements
    concentrations: tuple
        Concentrations of the elements, must sum to one
    source: str
        source of the data
    """

    elements: list 
    concentrations: tuple
    source: str

    def __post_init__(self):
        if not np.isclose(np.sum(self.concentrations), 1.0):
            raise ValueError("concentrations must sum to 1")

@dataclass
class FCCMaterial(Material):
    """Describe a face-centered cubic material

    Attributes
    ----------
    lattice_parameter: float
        Mean FCC lattice parameter
    C11: float
        Cubic elastic constant
    C12: float
        Cubic elastic constant
    C44: float
        Cubic elastic constant
    """

    lattice_parameter: float
    C11: float
    C12: float
    C44: float

    def calculate_zener_ratio(self) -> float:
        """Calculate the Zener anisotropy ratio.
        
        Returns
        -------
        zener_ratio: float
            The Zener ratio :math:`2C_{44}/(C_{11}-C_{12})`
        """
        return 2.0 * self.C44 / (self.C11 - self.C12)

average_FeNiCr_stiffened = FCCMaterial(
    elements=("X",), concentrations=(1.0,),  
    source="",
    lattice_parameter=3.52181864516065, 
    C11=246.9299588857714, 
    C12=147.0719468074990, 
    C44=125.0305675876400,
)

random_FeNiCr_stiffened = FCCMaterial(
    elements=("Fe", "Ni", "Cr"), 
    concentrations=(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0), 
    source="",
    lattice_parameter=3.52136543384631, 
    C11=242.92730469, 
    C12=157.03243264, 
    C44=134.89972062
)

average_FeNiCr_Bonny_2011 = FCCMaterial(
    elements=("A",), concentrations=(1.0,),  
    source="""\
---
description: EAM potential for alloys of Fe, Ni and Cr, with average-alloy 
  potential for the equicomposition alloy with 33% Fe, 33% Ni, and 33% Cr.
reference: "G. Bonny, D. Terentyev, R.C. Pasianot, S. Poncé, and A. Bakaev (2011), 
  'Interatomic potential to study plasticity in stainless steels: the FeNiCr 
  model alloy', Modelling and Simulation in Materials Science and Engineering, 
  19(8), 085008. DOI: 10.1088/0965-0393/19/8/085008"
url: "https://www.ctcms.nist.gov/potentials/Download/2011--Bonny-G-Terentyev-D-Pasianot-R-C-et-al--Fe-Ni-Cr/1/FeNiCr.eam.alloy"
""",
    lattice_parameter=3.5218186155137734, 
    C11=246.610145262896367, 
    C12=158.121884686608013, 
    C44=138.525095411635675,
)

random_FeNiCr_Bonny_2011 = FCCMaterial(
    elements=("Fe", "Ni", "Cr"), concentrations=(1.0/3.0, 1.0/3.0, 1.0/3.0),  
    source="""\
---
description: EAM potential for alloys of Fe, Ni and Cr. The data here is for the equicomposition random alloy with 33% Fe, 33% Ni, and 33% Cr.
reference: "G. Bonny, D. Terentyev, R.C. Pasianot, S. Poncé, and A. Bakaev (2011), 
  'Interatomic potential to study plasticity in stainless steels: the FeNiCr 
  model alloy', Modelling and Simulation in Materials Science and Engineering, 
  19(8), 085008. DOI: 10.1088/0965-0393/19/8/085008"
url: "https://www.ctcms.nist.gov/potentials/Download/2011--Bonny-G-Terentyev-D-Pasianot-R-C-et-al--Fe-Ni-Cr/1/FeNiCr.eam.alloy". 
""",
    lattice_parameter=3.52137032484813,
    C11=243.39483263182854,
    C12=157.44603732988062,
    C44=134.9885847097457,
)
