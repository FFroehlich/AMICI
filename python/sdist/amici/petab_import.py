"""
PEtab Import
------------
Import a model in the :mod:`petab` (https://github.com/PEtab-dev/PEtab) format
into AMICI.

.. deprecated:: 0.21.0
    Use :mod:`amici.petab` instead.
"""
import warnings

warnings.warn(
    "Importing amici.petab_import is deprecated. Use `amici.petab` instead.",
    DeprecationWarning,
)

from .petab.import_helpers import (  # noqa # pylint: disable=unused-import
    get_observation_model,
    petab_noise_distributions_to_amici,
    petab_scale_to_amici_scale,
)

# DEPRECATED - DON'T ADD ANYTHING NEW HERE
from .petab.petab_import import (  # noqa # pylint: disable=unused-import
    check_model,
    import_model,
    import_model_sbml,
    import_petab_problem,
)
from .petab.sbml_import import (  # noqa
    _get_fixed_parameters_sbml as get_fixed_parameters,
)
from .petab.sbml_import import species_to_parameters  # noqa

__all__ = [
    "get_observation_model",
    "petab_noise_distributions_to_amici",
    "petab_scale_to_amici_scale",
    "check_model",
    "import_model",
    "import_model_sbml",
    "import_petab_problem",
    "get_fixed_parameters",
    "species_to_parameters",
]
