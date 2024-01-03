"""PEtab-problem based simulations."""
import copy
from typing import Optional, Sequence, Union

import amici
import pandas as pd
import petab
from petab.C import (
    DATASET_ID,
    NOISE_PARAMETERS,
    OBSERVABLE_ID,
    PREEQUILIBRATION_CONDITION_ID,
    SIMULATION,
    SIMULATION_CONDITION_ID,
    TIME,
)

from .conditions import create_edatas, fill_in_parameters
from .parameter_mapping import create_parameter_mapping


class AmiciPetabProblem:
    """Manage experimental conditions based on a PEtab problem definition.

    Create :class:`ExpData` objects from a PEtab problem definition, handle
    parameter scales and parameter mapping.

    :param petab_problem: PEtab problem definition.
    :param amici_model: AMICI model
    :param amici_solver: AMICI solver (Solver with default options will be
        used if not provided).
    :param problem_parameters: Problem parameters to use for simulation
        (default: PEtab nominal values and model values).
    :param scaled_parameters: Whether the provided parameters are on PEtab
        `parameterScale` or not.
    :param simulation_conditions: Simulation conditions to use for simulation.
        It Can be used to subset the conditions in the PEtab problem.
        All subsequent operations will only be performed based on that subset.
        By default, all conditions are used.
    :param store_edatas: Whether to create and store all ExpData objects for
        all conditions upfront. If set to False, ExpData objects will be
        created and disposed of on the fly during simulation. This can save
        memory if many conditions are simulated.
    """

    def __init__(
        self,
        petab_problem: petab.Problem,
        amici_model: amici.Model,
        problem_parameters: Optional[dict[str, float]] = None,
        # move to a separate AmiciPetabProblemSimulator class?
        amici_solver: Optional[amici.Solver] = None,
        scaled_parameters: Optional[bool] = False,
        simulation_conditions: Union[pd.DataFrame, list[dict]] = None,
        store_edatas: bool = True,
    ):
        self._petab_problem = petab_problem
        self._amici_model = amici_model
        self._amici_solver = amici_solver
        self._scaled_parameters = scaled_parameters

        self._simulation_conditions = simulation_conditions or (
            petab_problem.get_simulation_conditions_from_measurement_df()
        )
        if not isinstance(self._simulation_conditions, pd.DataFrame):
            self._simulation_conditions = pd.DataFrame(
                self._simulation_conditions
            )
        if (
            preeq_id := PREEQUILIBRATION_CONDITION_ID
            in self._simulation_conditions
        ):
            self._simulation_conditions[
                preeq_id
            ] = self._simulation_conditions[preeq_id].fillna("")

        if problem_parameters is None:
            # Use PEtab nominal values as default
            self._problem_parameters = self._default_parameters()
            if scaled_parameters is True:
                raise NotImplementedError(
                    "scaled_parameters=True in combination with default "
                    "parameters is not implemented yet."
                )
            scaled_parameters = False
        else:
            self._problem_parameters = problem_parameters
        self._scaled_parameters = scaled_parameters

        if store_edatas:
            self._parameter_mapping = create_parameter_mapping(
                petab_problem=self._petab_problem,
                simulation_conditions=self._simulation_conditions,
                scaled_parameters=self._scaled_parameters,
                amici_model=self._amici_model,
            )

            self._create_edatas()
        else:
            self._parameter_mapping = None
            self._edatas = None

    def set_parameters(
        self,
        problem_parameters: dict[str, float],
        scaled_parameters: bool = False,
    ):
        """Set problem parameters.

        :param problem_parameters: Problem parameters to use for simulation.
        :param scaled_parameters: Whether the provided parameters are on PEtab
            `parameterScale` or not.
        """
        if scaled_parameters != self._scaled_parameters:
            # redo parameter mapping if scale changed
            self._parameter_mapping = create_parameter_mapping(
                petab_problem=self._petab_problem,
                simulation_conditions=self._simulation_conditions,
                scaled_parameters=scaled_parameters,
                amici_model=self._amici_model,
            )

        self._problem_parameters = problem_parameters
        self._scaled_parameters = scaled_parameters

        if self._edatas:
            fill_in_parameters(
                edatas=self._edatas,
                problem_parameters=self._problem_parameters,
                scaled_parameters=self._scaled_parameters,
                parameter_mapping=self._parameter_mapping,
                amici_model=self._amici_model,
            )

    def get_edata(
        self, condition_id: str, preequilibration_condition_id: str
    ) -> amici.ExpData:
        """Get ExpData object for a given condition.

        NOTE: If `store_edatas=True` was passed to the constructor and the
        returned object is modified, the changes will be reflected in the
        internal ExpData objects. Also, if parameter values of
        AmiciPetabProblem are changed, all ExpData objects will be updated.
        Create a deep copy if you want to avoid this.

        :param condition_id: PEtab Condition ID
        :param preequilibration_condition_id: PEtab Preequilibration condition ID
        :return: ExpData object
        """
        # exists or has to be created?
        if self._edatas:
            edata_id = condition_id
            if preequilibration_condition_id:
                edata_id += "+" + preequilibration_condition_id

            for edata in self._edatas:
                if edata.id == edata_id:
                    return edata

        return self._create_edata(condition_id, preequilibration_condition_id)

    def get_edatas(self):
        """Get all ExpData objects.

        NOTE: If `store_edatas=True` was passed to the constructor and the
        returned objects are modified, the changes will be reflected in the
        internal ExpData objects. Also, if parameter values of
        AmiciPetabProblem are changed, all ExpData objects will be updated.
        Create a deep copy if you want to avoid this.

        :return: List of ExpData objects
        """
        if self._edatas:
            # shallow copy
            return self._edatas.copy()

        # not storing edatas - create and return
        self._create_edatas()
        result = self._edatas
        self._edatas = []
        return result

    def _create_edata(
        self, condition_id: str, preequilibration_condition_id: str
    ) -> amici.ExpData:
        """Create ExpData object for a given condition.

        :param condition_id: PEtab Condition ID
        :param preequilibration_condition_id: PEtab Preequilibration condition ID
        :return: ExpData object
        """
        simulation_condition = pd.DataFrame(
            [
                {
                    SIMULATION_CONDITION_ID: condition_id,
                    PREEQUILIBRATION_CONDITION_ID: preequilibration_condition_id
                    or "",
                }
            ]
        )
        edatas = create_edatas(
            amici_model=self._amici_model,
            petab_problem=self._petab_problem,
            simulation_conditions=simulation_condition,
        )
        parameter_mapping = create_parameter_mapping(
            petab_problem=self._petab_problem,
            simulation_conditions=simulation_condition,
            scaled_parameters=self._scaled_parameters,
            amici_model=self._amici_model,
        )

        # Fill parameters in ExpDatas (in-place)
        fill_in_parameters(
            edatas=edatas,
            problem_parameters=self._problem_parameters,
            scaled_parameters=self._scaled_parameters,
            parameter_mapping=parameter_mapping,
            amici_model=self._amici_model,
        )

        if len(edatas) != 1:
            raise AssertionError("Expected exactly one ExpData object.")
        return edatas[0]

    @property
    def solver(self):
        """Get the solver."""
        return self._amici_solver or self._amici_model.getSolver()

    def _create_edatas(
        self,
    ):
        """Create ExpData objects from PEtab problem definition."""
        self._edatas = create_edatas(
            amici_model=self._amici_model,
            petab_problem=self._petab_problem,
            simulation_conditions=self._simulation_conditions,
        )

        fill_in_parameters(
            edatas=self._edatas,
            problem_parameters=self._problem_parameters,
            scaled_parameters=self._scaled_parameters,
            parameter_mapping=self._parameter_mapping,
            amici_model=self._amici_model,
        )

    def _default_parameters(self) -> dict[str, float]:
        return {
            t.Index: getattr(t, petab.NOMINAL_VALUE)
            for t in self._petab_problem.parameter_df.itertuples()
        }
