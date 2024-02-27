"""
Evaluate a PEtab objective function.

.. deprecated:: 0.21.0
    Use :mod:`amici.petab.simulations` instead.
"""

# THIS FILE IS TO BE REMOVED - DON'T ADD ANYTHING HERE!

import warnings

warnings.warn(
    f"Importing {__name__} is deprecated. Use `amici.petab.simulations` instead.",
    DeprecationWarning,
)

from .petab.conditions import fill_in_parameters  # noqa: F401
from .petab.parameter_mapping import create_parameter_mapping  # noqa: F401
from .petab.simulations import (  # noqa: F401
    EDATAS,
    FIM,
    LLH,
    RDATAS,
    RES,
    S2LLH,
    SLLH,
    SRES,
    aggregate_sllh,
    create_edatas,
    rdatas_to_measurement_df,
    rdatas_to_simulation_df,
    rescale_sensitivity,
    simulate_petab,
)

__all__ = [
    "EDATAS",
    "FIM",
    "LLH",
    "RDATAS",
    "RES",
    "S2LLH",
    "SLLH",
    "SRES",
    "aggregate_sllh",
    "create_edatas",
    "fill_in_parameters",
    "create_parameter_mapping",
    "rdatas_to_measurement_df",
    "rdatas_to_simulation_df",
    "rescale_sensitivity",
    "simulate_petab",
]
