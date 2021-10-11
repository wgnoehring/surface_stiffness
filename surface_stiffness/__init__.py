import logging
from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions

logger = logging.getLogger('surface_stiffness')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('surface_stiffness.log')
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info(f"surface_stiffness version {__version__}")
