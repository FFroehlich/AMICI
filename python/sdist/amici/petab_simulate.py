"""
Simulate a PEtab problem

.. deprecated:: 0.21.0
    Use :mod:`amici.petab.simulator` instead.
"""
# THIS FILE IS TO BE REMOVED - DON'T ADD ANYTHING HERE!

import warnings

warnings.warn(
    f"Importing {__name__} is deprecated. Use `amici.petab.simulator` instead.",
    DeprecationWarning,
)

from .petab.simulator import PetabSimulator  # noqa: F401

__all__ = [
    "PetabSimulator",
]
