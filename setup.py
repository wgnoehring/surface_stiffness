from setuptools import setup

setup(
   name='surface_stiffness',
   version='0.1.0',
   author='Wolfram Nöhring',
   author_email='wolfram.noehring@imtek.uni-freiburg.de',
   packages=['surface_stiffness'],
   scripts=[ 
    'scripts/calculate_greens_functions.py',
    'scripts/calculate_inverse_at_each_site.py',
    'scripts/calculate_stiffness_from_greens_function_average.py',
    'scripts/change_atom_identifiers.py',
    'scripts/construct_hessian.py',
    'scripts/convert_data_to_xyz.py',
    'scripts/create_mask.py',
    'scripts/fourier_transform_greens_functions_100_surface.py'
   ],
   url='https://github.com/wgnoehring/surface_stiffness',
   license_files = ('LICENSE',),
   description='Python code for the manuscript https://arxiv.org/abs/2101.12519',
   long_description=open('README.md').read(),
   install_requires=[
        "numpy", "pandas", "scipy"
   ],
)
