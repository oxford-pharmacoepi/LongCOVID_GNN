"""
Data processing package.
Modular data loading, mapping, filtering, and storage for OpenTargets datasets.
"""

from .loaders import OpenTargetsLoader
from .mappers import IdMapper, NodeIndexMapper
from .filters import MoleculeFilter, AssociationFilter
from .storage import DataStorage

__all__ = [
    'OpenTargetsLoader',
    'IdMapper',
    'NodeIndexMapper',
    'MoleculeFilter',
    'AssociationFilter',
    'DataStorage',
]
