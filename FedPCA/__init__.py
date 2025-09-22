"""Backward-compatible shim for the historical FedPCA namespace."""
from importlib import import_module as _import_module

_module = _import_module('federated_harmony.fedpca')

global horizontal_pca_power_iteration, shared_functions

horizontal_pca_power_iteration = _import_module('federated_harmony.fedpca.horizontal_pca_power_iteration')
shared_functions = _import_module('federated_harmony.fedpca.shared_functions')

__all__ = []
__all__.extend(getattr(horizontal_pca_power_iteration, '__all__', []))
__all__.extend(getattr(shared_functions, '__all__', []))

for _name in dir(horizontal_pca_power_iteration):
    if _name.startswith('_'):
        continue
    globals()[_name] = getattr(horizontal_pca_power_iteration, _name)

for _name in dir(shared_functions):
    if _name.startswith('_'):
        continue
    globals()[_name] = getattr(shared_functions, _name)

