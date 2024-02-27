"""
PEtab Simulator
---------------
Functionality related to the use of AMICI for simulation with :class:`petab.Simulator`.

Use cases:

- generate data for use with PEtab's plotting methods
- generate synthetic data
"""

import inspect
import sys
from typing import Callable

import pandas as pd
import petab
from amici import AmiciModel, SensitivityMethod_none

from .petab_import import import_petab_problem
from .simulations import RDATAS, rdatas_to_measurement_df, simulate_petab

AMICI_MODEL = "amici_model"
AMICI_SOLVER = "solver"
MODEL_NAME = "model_name"
MODEL_OUTPUT_DIR = "model_output_dir"

PETAB_PROBLEM = "petab_problem"


class PetabSimulator(petab.simulate.Simulator):
    """Implementation of the PEtab `Simulator` class that uses AMICI."""

    def __init__(self, *args, amici_model: AmiciModel = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.amici_model = amici_model

    def simulate_without_noise(self, **kwargs) -> pd.DataFrame:
        """
        See :py:func:`petab.simulate.Simulator.simulate()` docstring.

        Additional keyword arguments can be supplied to specify arguments for
        the AMICI PEtab import, simulate, and export methods. See the
        docstrings for the respective methods for argument options:
        - :py:func:`amici.petab_import.import_petab_problem`, and
        - :py:func:`amici.petab_objective.simulate_petab`.

        Note that some arguments are expected to have already been specified
        in the Simulator constructor (including the PEtab problem).
        """
        if AMICI_MODEL in {*kwargs, *dir(self)} and (
            any(
                k in kwargs
                for k in inspect.signature(import_petab_problem).parameters
            )
        ):
            print(
                "Arguments related to the PEtab import are unused if "
                f"`{AMICI_MODEL}` is specified, or the "
                "`PetabSimulator.simulate()` method was previously called."
            )

        kwargs[PETAB_PROBLEM] = self.petab_problem

        # The AMICI model instance for the PEtab problem is saved in the state,
        # such that it need not be supplied with each request for simulated
        # data. Any user-supplied AMICI model will overwrite the model saved
        # in the state.
        if AMICI_MODEL not in kwargs:
            if self.amici_model is None:
                if MODEL_NAME not in kwargs:
                    kwargs[MODEL_NAME] = AMICI_MODEL
                    # If the model name is the name of a module that is already
                    # cached, it can cause issues during import.
                    while kwargs[MODEL_NAME] in sys.modules:
                        kwargs[MODEL_NAME] += str(self.rng.integers(10))
                if MODEL_OUTPUT_DIR not in kwargs:
                    kwargs[MODEL_OUTPUT_DIR] = self.working_dir
                self.amici_model = _subset_call(import_petab_problem, kwargs)
            kwargs[AMICI_MODEL] = self.amici_model
        self.amici_model = kwargs[AMICI_MODEL]

        if AMICI_SOLVER not in kwargs:
            kwargs[AMICI_SOLVER] = self.amici_model.getSolver()
            kwargs[AMICI_SOLVER].setSensitivityMethod(SensitivityMethod_none)

        result = _subset_call(simulate_petab, kwargs)
        return rdatas_to_measurement_df(
            result[RDATAS], self.amici_model, self.petab_problem.measurement_df
        )


def _subset_call(method: Callable, kwargs: dict):
    """
    Helper function to call a method with the intersection of arguments in the
    method signature and the supplied arguments.

    :param method:
        The method to be called.
    :param kwargs:
        The argument superset as a dictionary, similar to ``**kwargs`` in
        method signatures.
    :return:
        The output of ``method``, called with the applicable arguments in
        ``kwargs``.
    """
    method_args = inspect.signature(method).parameters
    subset_kwargs = {k: v for k, v in kwargs.items() if k in method_args}
    return method(**subset_kwargs)
