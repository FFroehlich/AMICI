"""Various helper functions for working with PEtab problems."""

# THIS FILE IS TO BE REMOVED - DON'T ADD ANYTHING HERE!

import warnings

from .petab import PREEQ_INDICATOR_ID  # noqa: F401
from .petab.util import get_states_in_condition_table  # noqa: F401

warnings.warn(
    f"Importing {__name__} is deprecated. Use `amici.petab.util` instead.",
    DeprecationWarning,
)
