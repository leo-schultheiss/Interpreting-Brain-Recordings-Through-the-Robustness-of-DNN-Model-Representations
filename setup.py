import logging
import re

from setuptools import find_packages, setup

# get version from voxelwise_tutorials/__init__.py
with open('compare_variance_residual/__init__.py') as f:
    infos = f.readlines()
__version__ = ''
for line in infos:
    if "__version__" in line:
        match = re.search(r"__version__ = '([^']*)'", line)
        __version__ = match.groups()[0]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

requirements = [
    "requests~=2.32.3",
    "tqdm~=4.66.5",
    'transformers~=4.45.2',
    "torch~=2.5.0",
    "pandas~=2.2.3",
    "numpy~=2.1.2",
    "tables~=3.10.1",
    "scipy~=1.14.1",
    "h5py~=3.12.1",
    "ridge-utils~=0.2.0",
    "gitpython",
    "himalaya~=0.4.6",
    "scikit-learn~=1.5.2",
]

extras_require = {
    "plotting": [
        "matplotlib~=3.9.2",
        "voxelwise-tutorials~=0.1.7"
    ]
}

if __name__ == "__main__":
    setup(
        name='compare_variance_residual',
        maintainer="Leo Schultheiß",
        maintainer_email="leo.schultheiss@tum.de",
        description="Tools for robustness testing of fMRI models using variance partitioning and residual analysis",
        # license='BSD (3-clause)',
        version=__version__,
        packages=find_packages(),
        install_requires=requirements,
        extras_require=extras_require,
        # long_description=long_description,
        long_description_content_type='text/x-rst',
    )