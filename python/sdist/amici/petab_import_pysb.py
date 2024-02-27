"""
PEtab import for PySB models

.. deprecated:: 0.21.0
    Use :mod:`amici.petab.pysb_import` instead.
"""
import warnings

from .petab.pysb_import import *  # noqa: F401, F403

# DEPRECATED - DON'T ADD ANYTHING NEW HERE

warnings.warn(
    "Importing amici.petab_import_pysb is deprecated. Use `amici.petab.pysb_import` instead.",
    DeprecationWarning,
)

__all__ = [
    "import_model_pysb",
]
