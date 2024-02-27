"""Parameter mapping between AMICI and PEtab.

.. deprecated:: 0.21.0
    Use :mod:`amici.petab.parameter_mapping` instead.
"""

# some extra imports for backward-compatibility
import warnings

from .petab.conditions import (  # noqa # pylint: disable=unused-import
    fill_in_parameters,
    fill_in_parameters_for_condition,
)
from .petab.parameter_mapping import (  # noqa # pylint: disable=unused-import
    ParameterMapping,
    ParameterMappingForCondition,
    SingleParameterMapping,
    SingleScaleMapping,
    amici_to_petab_scale,
    petab_to_amici_scale,
    scale_parameter,
    scale_parameters_dict,
    unscale_parameter,
    unscale_parameters_dict,
)

warnings.warn(
    "Importing amici.parameter_mapping is deprecated. Use `amici.petab.parameter_mapping` instead.",
    DeprecationWarning,
)

__all__ = [
    "fill_in_parameters",
    "fill_in_parameters_for_condition",
    "ParameterMapping",
    "ParameterMappingForCondition",
    "SingleParameterMapping",
    "SingleScaleMapping",
    "amici_to_petab_scale",
    "petab_to_amici_scale",
    "scale_parameter",
    "scale_parameters_dict",
    "unscale_parameter",
    "unscale_parameters_dict",
]
