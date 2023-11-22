# THIS FILE IS TO BE REMOVED - DON'T ADD ANYTHING HERE!

import warnings

warnings.warn(
    f"Importing {__name__} is deprecated. Use `amici.petab.simulations` instead.",
    DeprecationWarning,
)

from .petab.simulations import (  # noqa: F401
    aggregate_sllh,
    rdatas_to_measurement_df,
    rdatas_to_simulation_df,
    rescale_sensitivity,
    simulate_petab,
)
