# mito_utils

Utilities for MI-TO analyses.
This package is still under active development. To use functions and modules perform the following steps:

1. Install mamba package manager (https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html)
2. Choose one of the following environment recipes:

```bash
envs
├── environment.yml # minimal, OSX
├── environment_latest_OSX.yml # Latest, OSX
├── environment_mito_utils.yml # Verified, Linux
└── environment_preprocessing.yml # Minimal, only for preprocessing
```

3. Reproduce the environment with:

```bash
mamba env create -f <chosen environment name>.yml -n <your conda env name>
```

4. activate the new environment (with successfull installation of all packages) and cd to the main mito_utils repo path.
Make the repo path visible to python import system.

```path
cd <path to mito_utils clone>
mamba develop .
```

5. Fire a python terminal and verify successfull installation.

```python
from mito_utils import *
```
