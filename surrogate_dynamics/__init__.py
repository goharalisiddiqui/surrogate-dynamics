"""
Collective Encoder: A machine learning framework for molecular dynamics.

This package provides autoencoder-based architectures for creating surrogate models
that predict dynamics of molecular systems as time series data.
"""

import os

__version__ = "0.1.0"
LIBRARY_ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(LIBRARY_ROOT, 'configs')
EXECUTION_DIR = os.getcwd()


from . import (
    datamodules,
    datareaders,
    datasets,
    propagators,
)
