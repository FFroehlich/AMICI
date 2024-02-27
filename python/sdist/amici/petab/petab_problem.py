"""PEtab-problem based simulations."""
import copy
from typing import Optional, Union

import amici
import pandas as pd
import petab
from petab.C import PREEQUILIBRATION_CONDITION_ID, SIMULATION_CONDITION_ID

from .conditions import create_edatas, fill_in_parameters
from .parameter_mapping import create_parameter_mapping


class PetabProblem:
    """Manage experimental conditions based on a PEtab problem definition.

    Create :class:`ExpData` objects from a PEtab problem definition, and handle
    parameter scales and parameter mapping.

    :param petab_problem: PEtab problem definition.
    :param amici_model: AMICI model
    :param problem_parameters: Problem parameters to use for simulation
        (default: PEtab nominal values and model values).
    :param scaled_parameters: Whether the provided parameters are on PEtab
        `parameterScale` or not.
    :param simulation_conditions: Simulation conditions to use for simulation.
        It can be used to subset the conditions in the PEtab problem.
        All subsequent operations will only be performed on that subset.
        By default, all conditions are used.
    :param store_edatas: Whether to create and store all `ExpData` objects for
        all conditions upfront. If set to ``False``, `ExpData` objects will be
        created and disposed of on the fly during simulation. The latter saves
        memory if the given PEtab problem comprises many simulation conditions.
    """

    def __init__(
        self,
        petab_problem: petab.Problem,
        amici_model: Optional[amici.Model] = None,
        problem_parameters: Optional[dict[str, float]] = None,
        scaled_parameters: bool = False,
        simulation_conditions: Union[pd.DataFrame, list[dict]] = None,
        store_edatas: bool = True,
    ):
        self._petab_problem = copy.deepcopy(petab_problem)

        if amici_model is not None:
            self._amici_model = amici_model
        else:
            from .petab_import import import_petab_problem

            self._amici_model = import_petab_problem(petab_problem)

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
        ) in self._simulation_conditions:
            self._simulation_conditions[
                preeq_id
            ] = self._simulation_conditions[preeq_id].fillna("")

        if problem_parameters is None:
            # Use PEtab nominal values as default
            self._problem_parameters = self._default_parameters()
            if scaled_parameters:
                raise NotImplementedError(
                    "scaled_parameters=True in combination with default "
                    "parameters is not implemented yet."
                )
        else:
            self._problem_parameters = problem_parameters

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
            This may be a subset of all parameters.
        :param scaled_parameters: Whether the provided parameters are on PEtab
            `parameterScale` or not.
        """
        if scaled_parameters != self._scaled_parameters and self._parameter_mapping is not None:
            # redo parameter mapping if scale changed
            self._parameter_mapping = create_parameter_mapping(
                petab_problem=self._petab_problem,
                simulation_conditions=self._simulation_conditions,
                scaled_parameters=scaled_parameters,
                amici_model=self._amici_model,
            )

        if set(self._problem_parameters) - set(problem_parameters):
            # not all parameters are provided - update
            # bring previously set parameters to the same scale if necessary
            if scaled_parameters and not self._scaled_parameters:
                self._problem_parameters = (
                    self._petab_problem.scale_parameters(
                        self._problem_parameters,
                    )
                )
            elif not scaled_parameters and self._scaled_parameters:
                self._problem_parameters = (
                    self._petab_problem.unscale_parameters(
                        self._problem_parameters,
                    )
                )
            self._problem_parameters |= problem_parameters
        else:
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
        self, condition_id: str, preequilibration_condition_id: str = None
    ) -> amici.ExpData:
        """Get ExpData object for a given condition.

        NOTE: If ``store_edatas=True`` was passed to the constructor and the
        returned object is modified, the changes will be reflected in the
        internal `ExpData` objects. Also, if parameter values of
        `PetabProblem` are changed, all `ExpData` objects will be updated.
        Create a deep copy if you want to avoid this.

        :param condition_id: PEtab condition ID
        :param preequilibration_condition_id: PEtab preequilibration condition ID
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

        NOTE: If ``store_edatas=True`` was passed to the constructor and the
        returned objects are modified, the changes will be reflected in the
        internal `ExpData` objects. Also, if parameter values of
        `PetabProblem` are changed, all `ExpData` objects will be updated.
        Create a deep copy if you want to avoid this.

        :return: List of ExpData objects
        """
        if self._edatas:
            # shallow copy
            return self._edatas.copy()

        # not storing edatas - create and return
        self._parameter_mapping = create_parameter_mapping(
            petab_problem=self._petab_problem,
            simulation_conditions=self._simulation_conditions,
            scaled_parameters=self._scaled_parameters,
            amici_model=self._amici_model,
        )
        self._create_edatas()
        result = self._edatas
        self._edatas = []
        return result

    def _create_edata(
        self, condition_id: str, preequilibration_condition_id: str
    ) -> amici.ExpData:
        """Create ExpData object for a given condition.

        :param condition_id: PEtab condition ID
        :param preequilibration_condition_id: PEtab preequilibration condition ID
        :return: ExpData object
        """
        simulation_condition = pd.DataFrame(
            [
                {
                    SIMULATION_CONDITION_ID: condition_id,
                    PREEQUILIBRATION_CONDITION_ID: preequilibration_condition_id
                    or None,
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
            problem_parameters={
                p: self._problem_parameters[p]
                for p in parameter_mapping.free_symbols
                if p in self._problem_parameters
            },
            scaled_parameters=self._scaled_parameters,
            parameter_mapping=parameter_mapping,
            amici_model=self._amici_model,
        )

        if len(edatas) != 1:
            raise AssertionError("Expected exactly one ExpData object.")
        return edatas[0]

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
        """Get unscaled default parameters."""
        return {
            t.Index: getattr(t, petab.NOMINAL_VALUE)
            for t in self._petab_problem.parameter_df[
                self._petab_problem.parameter_df[petab.ESTIMATE] == 1
            ].itertuples()
        }

    @property
    def model(self) -> amici.Model:
        """AMICI model."""
        return self._amici_model
