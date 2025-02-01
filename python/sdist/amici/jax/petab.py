"""PEtab wrappers for JAX models.""" ""

import copy
import shutil
from numbers import Number
from collections.abc import Sized, Iterable
from pathlib import Path
from collections.abc import Callable


import diffrax
import equinox as eqx
import jaxtyping as jt
import jax.lax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import petab.v1 as petab

from amici import _module_from_path
from amici.petab.parameter_mapping import (
    ParameterMappingForCondition,
    create_parameter_mapping,
)
from amici.jax.model import JAXModel, ReturnValue

DEFAULT_CONTROLLER_SETTINGS = {
    "atol": 1e-8,
    "rtol": 1e-8,
    "pcoeff": 0.4,
    "icoeff": 0.3,
    "dcoeff": 0.0,
}

SCALE_TO_INT = {
    petab.LIN: 0,
    petab.LOG: 1,
    petab.LOG10: 2,
}


def jax_unscale(
    parameter: jnp.float_,
    scale_str: str,
) -> jnp.float_:
    """Unscale parameter according to ``scale_str``.

    Arguments:
        parameter:
            Parameter to be unscaled.
        scale_str:
            One of ``petab.LIN``, ``petab.LOG``, ``petab.LOG10``.

    Returns:
        The unscaled parameter.
    """
    if scale_str == petab.LIN or not scale_str:
        return parameter
    if scale_str == petab.LOG:
        return jnp.exp(parameter)
    if scale_str == petab.LOG10:
        return jnp.power(10, parameter)
    raise ValueError(f"Invalid parameter scaling: {scale_str}")


class JAXProblem(eqx.Module):
    """
    PEtab problem wrapper for JAX models.

    :ivar parameters:
        Values for the model parameters. Do not change dimensions, values may be changed during, e.g. model training.
    :ivar model:
        JAXModel instance to use for simulation.
    :ivar _parameter_mappings:
        :class:`ParameterMappingForCondition` instances for each simulation condition.
    :ivar _measurements:
        Preprocessed arrays for each simulation condition.
    :ivar _petab_problem:
        PEtab problem to simulate.
    """

    parameters: jnp.ndarray
    model: JAXModel
    simulation_conditions: tuple[tuple[str, ...], ...]
    _parameter_mappings: dict[str, ParameterMappingForCondition]
    _ts_dyn: np.ndarray
    _ts_posteq: np.ndarray
    _my: np.ndarray
    _iys: np.ndarray
    _iy_trafos: np.ndarray
    _ts_masks: np.ndarray
    _op_numeric: np.ndarray
    _op_mask: np.ndarray
    _op_indices: np.ndarray
    _np_numeric: np.ndarray
    _np_mask: np.ndarray
    _np_indices: np.ndarray
    _petab_measurement_indices: np.ndarray
    _petab_problem: petab.Problem

    def __init__(self, model: JAXModel, petab_problem: petab.Problem):
        """
        Initialize a JAXProblem instance with a model and a PEtab problem.

        :param model:
            JAXModel instance to use for simulation.
        :param petab_problem:
            PEtab problem to simulate.
        """
        self.model = model
        scs = petab_problem.get_simulation_conditions_from_measurement_df()
        self.simulation_conditions = tuple(tuple(sc) for sc in scs.values)
        self._petab_problem = petab_problem
        self._parameter_mappings = self._get_parameter_mappings(scs)
        (
            self._ts_dyn,
            self._ts_posteq,
            self._my,
            self._iys,
            self._iy_trafos,
            self._ts_masks,
            self._petab_measurement_indices,
            self._op_numeric,
            self._op_mask,
            self._op_indices,
            self._np_numeric,
            self._np_mask,
            self._np_indices,
        ) = self._get_measurements(scs)

        self.parameters = self._get_nominal_parameter_values()

    def save(self, directory: Path):
        """
        Save the problem to a directory.

        :param directory:
            Directory to save the problem to.
        """
        self._petab_problem.to_files(
            prefix_path=directory,
            model_file="model",
            condition_file="conditions.tsv",
            measurement_file="measurements.tsv",
            parameter_file="parameters.tsv",
            observable_file="observables.tsv",
            yaml_file="problem.yaml",
        )
        shutil.copy(self.model.jax_py_file, directory / "jax_py_file.py")
        with open(directory / "parameters.pkl", "wb") as f:
            eqx.tree_serialise_leaves(f, self)

    @classmethod
    def load(cls, directory: Path):
        """
        Load a problem from a directory.

        :param directory:
            Directory to load the problem from.

        :return:
            Loaded problem instance.
        """
        petab_problem = petab.Problem.from_yaml(
            directory / "problem.yaml",
        )
        model = _module_from_path("jax", directory / "jax_py_file.py").Model()
        problem = cls(model, petab_problem)
        with open(directory / "parameters.pkl", "rb") as f:
            return eqx.tree_deserialise_leaves(f, problem)

    def _get_parameter_mappings(
        self, simulation_conditions: pd.DataFrame
    ) -> dict[str, ParameterMappingForCondition]:
        """
        Create parameter mappings for the provided simulation conditions.

        :param simulation_conditions:
            Simulation conditions to create parameter mappings for. Same format as returned by
            :meth:`petab.Problem.get_simulation_conditions_from_measurement_df`.
        :return:
            Dictionary mapping simulation conditions to parameter mappings.
        """
        scs = list(set(simulation_conditions.values.flatten()))
        petab_problem = copy.deepcopy(self._petab_problem)
        # remove observable and noise parameters from measurement dataframe as we are mapping them elsewhere
        petab_problem.measurement_df.drop(
            columns=[petab.OBSERVABLE_PARAMETERS, petab.NOISE_PARAMETERS],
            inplace=True,
            errors="ignore",
        )
        mappings = create_parameter_mapping(
            petab_problem=petab_problem,
            simulation_conditions=[
                {petab.SIMULATION_CONDITION_ID: sc} for sc in scs
            ],
            scaled_parameters=False,
            allow_timepoint_specific_numeric_noise_parameters=True,
        )
        # fill in dummy variables
        for mapping in mappings:
            for sim_var, value in mapping.map_sim_var.items():
                if isinstance(value, Number) and not np.isfinite(value):
                    mapping.map_sim_var[sim_var] = 1.0
        return dict(zip(scs, mappings, strict=True))

    def _get_measurements(
        self, simulation_conditions: pd.DataFrame
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """
        Get measurements for the model based on the provided simulation conditions.

        :param simulation_conditions:
            Simulation conditions to create parameter mappings for. Same format as returned by
            :meth:`petab.Problem.get_simulation_conditions_from_measurement_df`.
        :return:
            tuple of padded
             - dynamic time points
             - post-equilibrium time points
             - measurements
             - observable indices
             - observable transformations indices
             - measurement masks
             - data indices (index in petab measurement dataframe).
             - numeric values for observable parameter overrides
             - non-numeric mask for observable parameter overrides
             - parameter indices (problem parameters) for observable parameter overrides
             - numeric values for noise parameter overrides
             - non-numeric mask for noise parameter overrides
             - parameter indices (problem parameters) for noise parameter overrides
        """
        measurements = dict()
        petab_indices = dict()

        n_pars = dict()
        for col in [petab.OBSERVABLE_PARAMETERS, petab.NOISE_PARAMETERS]:
            n_pars[col] = 0
            if col in self._petab_problem.measurement_df:
                if np.issubdtype(
                    self._petab_problem.measurement_df[col].dtype, np.number
                ):
                    n_pars[col] = 1 - int(
                        self._petab_problem.measurement_df[col].isna().all()
                    )
                else:
                    n_pars[col] = (
                        self._petab_problem.measurement_df[col]
                        .str.split(";")
                        .apply(
                            lambda x: len(x)
                            if isinstance(x, Sized)
                            else 1 - int(pd.isna(x))
                        )
                        .max()
                    )

        for _, simulation_condition in simulation_conditions.iterrows():
            query = " & ".join(
                [f"{k} == '{v}'" for k, v in simulation_condition.items()]
            )
            m = self._petab_problem.measurement_df.query(query).sort_values(
                by=petab.TIME
            )

            ts = m[petab.TIME]
            ts_dyn = ts[np.isfinite(ts)]
            ts_posteq = ts[np.logical_not(np.isfinite(ts))]
            index = pd.concat([ts_dyn, ts_posteq]).index
            ts_dyn = ts_dyn.values
            ts_posteq = ts_posteq.values
            my = m[petab.MEASUREMENT].values
            iys = np.array(
                [
                    self.model.observable_ids.index(oid)
                    for oid in m[petab.OBSERVABLE_ID].values
                ]
            )
            if (
                petab.OBSERVABLE_TRANSFORMATION
                in self._petab_problem.observable_df
            ):
                iy_trafos = np.array(
                    [
                        SCALE_TO_INT[
                            self._petab_problem.observable_df.loc[
                                oid, petab.OBSERVABLE_TRANSFORMATION
                            ]
                        ]
                        for oid in m[petab.OBSERVABLE_ID].values
                    ]
                )
            else:
                iy_trafos = np.zeros_like(iys)

            parameter_overrides_par_indices = dict()
            parameter_overrides_numeric_vals = dict()
            parameter_overrides_mask = dict()

            def get_parameter_override(x):
                if (
                    x in self._petab_problem.parameter_df.index
                    and not self._petab_problem.parameter_df.loc[
                        x, petab.ESTIMATE
                    ]
                ):
                    return self._petab_problem.parameter_df.loc[
                        x, petab.NOMINAL_VALUE
                    ]
                return x

            for col in [petab.OBSERVABLE_PARAMETERS, petab.NOISE_PARAMETERS]:
                if col not in m or m[col].isna().all():
                    mat_numeric = jnp.ones((len(m), n_pars[col]))
                    par_mask = np.zeros_like(mat_numeric, dtype=bool)
                    par_index = np.zeros_like(mat_numeric, dtype=int)
                elif np.issubdtype(m[col].dtype, np.number):
                    mat_numeric = np.expand_dims(m[col].values, axis=1)
                    par_mask = np.zeros_like(mat_numeric, dtype=bool)
                    par_index = np.zeros_like(mat_numeric, dtype=int)
                else:
                    split_vals = m[col].str.split(";")
                    list_vals = split_vals.apply(
                        lambda x: [get_parameter_override(y) for y in x]
                        if isinstance(x, list)
                        else []
                        if pd.isna(x)
                        else [
                            x
                        ]  # every string gets transformed to lists, so this is already a float
                    )
                    vals = list_vals.apply(
                        lambda x: np.pad(
                            x,
                            (0, n_pars[col] - len(x)),
                            mode="constant",
                            constant_values=1.0,
                        )
                    )
                    mat = np.stack(vals)
                    # deconstruct such that we can reconstruct mapped parameter overrides via vectorized operations
                    # mat = np.where(par_mask, map(lambda ip: p.at[ip], par_index), mat_numeric)
                    par_index = np.vectorize(
                        lambda x: self.parameter_ids.index(x)
                        if x in self.parameter_ids
                        else -1
                    )(mat)
                    # map out numeric values
                    par_mask = par_index != -1
                    # remove non-numeric values
                    mat[par_mask] = 0.0
                    mat_numeric = mat.astype(float)
                    # replace dummy index with some valid index
                    par_index[~par_mask] = 0

                parameter_overrides_numeric_vals[col] = mat_numeric
                parameter_overrides_mask[col] = par_mask
                parameter_overrides_par_indices[col] = par_index

            measurements[tuple(simulation_condition)] = (
                ts_dyn,  # 0
                ts_posteq,  # 1
                my,  # 2
                iys,  # 3
                iy_trafos,  # 4
                parameter_overrides_numeric_vals[
                    petab.OBSERVABLE_PARAMETERS
                ],  # 5
                parameter_overrides_mask[petab.OBSERVABLE_PARAMETERS],  # 6
                parameter_overrides_par_indices[
                    petab.OBSERVABLE_PARAMETERS
                ],  # 7
                parameter_overrides_numeric_vals[petab.NOISE_PARAMETERS],  # 8
                parameter_overrides_mask[petab.NOISE_PARAMETERS],  # 9
                parameter_overrides_par_indices[petab.NOISE_PARAMETERS],  # 10
            )
            petab_indices[tuple(simulation_condition)] = tuple(index.tolist())

        # compute maximum lengths
        n_ts_dyn = max(len(mv[0]) for mv in measurements.values())
        n_ts_posteq = max(len(mv[1]) for mv in measurements.values())

        # pad with last value and stack
        ts_dyn = np.stack(
            [
                np.pad(mv[0], (0, n_ts_dyn - len(mv[0])), mode="edge")
                for mv in measurements.values()
            ]
        )
        ts_posteq = np.stack(
            [
                np.pad(mv[1], (0, n_ts_posteq - len(mv[1])), mode="edge")
                for mv in measurements.values()
            ]
        )

        def pad_measurement(x_dyn, x_peq):
            # only pad first axis
            pad_width_dyn = tuple(
                [(0, n_ts_dyn - len(x_dyn))] + [(0, 0)] * (x_dyn.ndim - 1)
            )
            pad_width_peq = tuple(
                [(0, n_ts_posteq - len(x_peq))] + [(0, 0)] * (x_peq.ndim - 1)
            )
            return np.concatenate(
                (
                    np.pad(x_dyn, pad_width_dyn, mode="edge"),
                    np.pad(x_peq, pad_width_peq, mode="edge"),
                )
            )

        def pad_and_stack(output_index: int):
            return np.stack(
                [
                    pad_measurement(
                        mv[output_index][: len(mv[0])],
                        mv[output_index][len(mv[0]) :],
                    )
                    for mv in measurements.values()
                ]
            )

        my = pad_and_stack(2)
        iys = pad_and_stack(3)
        iy_trafos = pad_and_stack(4)
        op_numeric = pad_and_stack(5)
        op_mask = pad_and_stack(6)
        op_indices = pad_and_stack(7)
        np_numeric = pad_and_stack(8)
        np_mask = pad_and_stack(9)
        np_indices = pad_and_stack(10)
        ts_masks = np.stack(
            [
                np.concatenate(
                    (
                        np.pad(
                            np.ones_like(mv[0]), (0, n_ts_dyn - len(mv[0]))
                        ),
                        np.pad(
                            np.ones_like(mv[1]), (0, n_ts_posteq - len(mv[1]))
                        ),
                    )
                )
                for mv in measurements.values()
            ]
        ).astype(bool)
        petab_indices = np.stack(
            [
                pad_measurement(
                    np.array(idx[: len(mv[0])]),
                    np.array(idx[len(mv[0]) :]),
                )
                for mv, idx in zip(
                    measurements.values(), petab_indices.values()
                )
            ]
        )

        return (
            ts_dyn,
            ts_posteq,
            my,
            iys,
            iy_trafos,
            ts_masks,
            petab_indices,
            op_numeric,
            op_mask,
            op_indices,
            np_numeric,
            np_mask,
            np_indices,
        )

    def get_all_simulation_conditions(self) -> tuple[tuple[str, ...], ...]:
        simulation_conditions = (
            self._petab_problem.get_simulation_conditions_from_measurement_df()
        )
        return tuple(tuple(row) for _, row in simulation_conditions.iterrows())

    def _get_nominal_parameter_values(self) -> jt.Float[jt.Array, "np"]:
        """
        Get the nominal parameter values for the model based on the nominal values in the PEtab problem.

        :return:
            jax array with nominal parameter values
        """
        return jnp.array(
            [
                petab.scale(
                    self._petab_problem.parameter_df.loc[
                        pval, petab.NOMINAL_VALUE
                    ],
                    self._petab_problem.parameter_df.loc[
                        pval, petab.PARAMETER_SCALE
                    ],
                )
                for pval in self.parameter_ids
            ]
        )

    @property
    def parameter_ids(self) -> list[str]:
        """
        Parameter ids that are estimated in the PEtab problem. Same ordering as values in :attr:`parameters`.

        :return:
            PEtab parameter ids
        """
        return self._petab_problem.parameter_df[
            self._petab_problem.parameter_df[petab.ESTIMATE] == 1
        ].index.tolist()

    def get_petab_parameter_by_id(self, name: str) -> jnp.float_:
        """
        Get the value of a PEtab parameter by name.

        :param name:
            PEtab parameter id, as returned by :attr:`parameter_ids`.
        :return:
            Value of the parameter
        """
        return self.parameters[self.parameter_ids.index(name)]

    def _unscale(
        self, p: jt.Float[jt.Array, "np"], scales: tuple[str, ...]
    ) -> jt.Float[jt.Array, "np"]:
        """
        Unscaling of parameters.

        :param p:
            Parameter values
        :param scales:
            Parameter scalings
        :return:
            Unscaled parameter values
        """
        return jnp.array(
            [jax_unscale(pval, scale) for pval, scale in zip(p, scales)]
        )

    def load_parameters(
        self, simulation_condition: str
    ) -> jt.Float[jt.Array, "np"]:
        """
        Load parameters for a simulation condition.

        :param simulation_condition:
            Simulation condition to load parameters for.
        :return:
            Parameters for the simulation condition.
        """
        mapping = self._parameter_mappings[simulation_condition]
        p = jnp.array(
            [
                pval
                if isinstance(pval := mapping.map_sim_var[pname], Number)
                else self.get_petab_parameter_by_id(pval)
                for pname in self.model.parameter_ids
            ]
        )
        pscale = tuple(
            [
                mapping.scale_map_sim_var[pname]
                for pname in self.model.parameter_ids
            ]
        )
        return self._unscale(p, pscale)

    def _state_needs_reinitialisation(
        self,
        simulation_condition: str,
        state_id: str,
    ) -> bool:
        """
        Check if a state needs reinitialisation for a simulation condition.

        :param simulation_condition:
            simulation condition to check reinitialisation for
        :param state_id:
            state id to check reinitialisation for
        :return:
            True if state needs reinitialisation, False otherwise
        """
        if state_id not in self._petab_problem.condition_df:
            return False
        xval = self._petab_problem.condition_df.loc[
            simulation_condition, state_id
        ]
        if isinstance(xval, Number) and np.isnan(xval):
            return False
        return True

    def _state_reinitialisation_value(
        self,
        simulation_condition: str,
        state_id: str,
        p: jt.Float[jt.Array, "np"],
    ) -> jt.Float[jt.Scalar, ""] | float:  # noqa: F722
        """
        Get the reinitialisation value for a state.

        :param simulation_condition:
            simulation condition to get reinitialisation value for
        :param state_id:
            state id to get reinitialisation value for
        :param p:
            parameters for the simulation condition
        :return:
            reinitialisation value for the state
        """
        if state_id not in self._petab_problem.condition_df:
            # no reinitialisation, return dummy value
            return 0.0
        xval = self._petab_problem.condition_df.loc[
            simulation_condition, state_id
        ]
        if isinstance(xval, Number) and np.isnan(xval):
            # no reinitialisation, return dummy value
            return 0.0
        if isinstance(xval, Number):
            # numerical value, return as is
            return xval
        if xval in self.model.parameter_ids:
            # model parameter, return value
            return p[self.model.parameter_ids.index(xval)]
        if xval in self.parameter_ids:
            # estimated PEtab parameter, return unscaled value
            return jax_unscale(
                self.get_petab_parameter_by_id(xval),
                self._petab_problem.parameter_df.loc[
                    xval, petab.PARAMETER_SCALE
                ],
            )
        # only remaining option is nominal value for PEtab parameter
        # that is not estimated, return nominal value
        return self._petab_problem.parameter_df.loc[xval, petab.NOMINAL_VALUE]

    def load_reinitialisation(
        self,
        simulation_condition: str,
        p: jt.Float[jt.Array, "np"],
    ) -> tuple[jt.Bool[jt.Array, "nx"], jt.Float[jt.Array, "nx"]]:  # noqa: F821
        """
        Load reinitialisation values and mask for the state vector for a simulation condition.

        :param simulation_condition:
            Simulation condition to load reinitialisation for.
        :param p:
            Parameters for the simulation condition.
        :return:
            Tuple of reinitialisation masm and value for states.
        """
        if not any(
            x_id in self._petab_problem.condition_df
            for x_id in self.model.state_ids
        ):
            return jnp.array([]), jnp.array([])

        mask = jnp.array(
            [
                self._state_needs_reinitialisation(simulation_condition, x_id)
                for x_id in self.model.state_ids
            ]
        )
        reinit_x = jnp.array(
            [
                self._state_reinitialisation_value(
                    simulation_condition, x_id, p
                )
                for x_id in self.model.state_ids
            ]
        )
        return mask, reinit_x

    def update_parameters(self, p: jt.Float[jt.Array, "np"]) -> "JAXProblem":
        """
        Update parameters for the model.

        :param p:
            New problem instance with updated parameters.
        """
        return eqx.tree_at(lambda p: p.parameters, self, p)

    def _prepare_conditions(
        self,
        conditions: list[str],
        op_numeric: np.ndarray | None = None,
        op_mask: np.ndarray | None = None,
        op_indices: np.ndarray | None = None,
        np_numeric: np.ndarray | None = None,
        np_mask: np.ndarray | None = None,
        np_indices: np.ndarray | None = None,
    ) -> tuple[
        jt.Float[jt.Array, "nc np"],  # noqa: F821, F722
        jt.Bool[jt.Array, "nx"],  # noqa: F821
        jt.Float[jt.Array, "nx"],  # noqa: F821
        jt.Float[jt.Array, "nc nt nop"],  # noqa: F821, F722
        jt.Float[jt.Array, "nc nt nnp"],  # noqa: F821, F722
    ]:
        """
        Prepare conditions for simulation.

        :param conditions:
            Simulation conditions to prepare.
        :param op_numeric:
            Numeric values for observable parameter overrides. If None, no overrides are used.
        :param op_mask:
            Mask for observable parameter overrides. True for free parameter overrides, False for numeric values.
        :param op_indices:
            Free parameter indices (wrt. `self.parameters`) for observable parameter overrides.
        :param np_numeric:
            Numeric values for noise parameter overrides. If None, no overrides are used.
        :param np_mask:
            Mask for noise parameter overrides. True for free parameter overrides, False for numeric values.
        :param np_indices:
            Free parameter indices (wrt. `self.parameters`) for noise parameter overrides.
        :return:
            Tuple of parameter arrays, reinitialisation masks and reinitialisation values, observable parameters and
            noise parameters.
        """
        p_array = jnp.stack([self.load_parameters(sc) for sc in conditions])
        unscaled_parameters = jnp.stack(
            [
                jax_unscale(
                    self.parameters[ip],
                    self._petab_problem.parameter_df.loc[
                        p_id, petab.PARAMETER_SCALE
                    ],
                )
                for ip, p_id in enumerate(self.parameter_ids)
            ]
        )

        if op_numeric is not None and op_numeric.size:
            op_array = jnp.where(
                op_mask,
                jax.vmap(
                    jax.vmap(jax.vmap(lambda ip: unscaled_parameters[ip]))
                )(op_indices),
                op_numeric,
            )
        else:
            op_array = jnp.zeros((*self._ts_masks.shape[:2], 0))

        if np_numeric is not None and np_numeric.size:
            np_array = jnp.where(
                np_mask,
                jax.vmap(
                    jax.vmap(jax.vmap(lambda ip: unscaled_parameters[ip]))
                )(np_indices),
                np_numeric,
            )
        else:
            np_array = jnp.zeros((*self._ts_masks.shape[:2], 0))

        mask_reinit_array = jnp.stack(
            [
                self.load_reinitialisation(sc, p)[0]
                for sc, p in zip(conditions, p_array)
            ]
        )
        x_reinit_array = jnp.stack(
            [
                self.load_reinitialisation(sc, p)[1]
                for sc, p in zip(conditions, p_array)
            ]
        )
        return p_array, mask_reinit_array, x_reinit_array, op_array, np_array

    @eqx.filter_vmap(
        in_axes={
            "max_steps": None,
            "self": None,
        },  # only list arguments here where eqx.is_array(0) is not the right thing
    )
    def run_simulation(
        self,
        p: jt.Float[jt.Array, "np"],  # noqa: F821, F722
        ts_dyn: np.ndarray,
        ts_posteq: np.ndarray,
        my: np.ndarray,
        iys: np.ndarray,
        iy_trafos: np.ndarray,
        ops: jt.Float[jt.Array, "nt *nop"],  # noqa: F821, F722
        nps: jt.Float[jt.Array, "nt *nnp"],  # noqa: F821, F722
        mask_reinit: jt.Bool[jt.Array, "nx"],  # noqa: F821, F722
        x_reinit: jt.Float[jt.Array, "nx"],  # noqa: F821, F722
        solver: diffrax.AbstractSolver,
        controller: diffrax.AbstractStepSizeController,
        steady_state_event: Callable[
            ..., diffrax._custom_types.BoolScalarLike
        ],
        max_steps: jnp.int_,
        x_preeq: jt.Float[jt.Array, "*nx"] = jnp.array([]),  # noqa: F821, F722
        ts_mask: np.ndarray = np.array([]),
        ret: ReturnValue = ReturnValue.llh,
    ) -> tuple[jnp.float_, dict]:
        """
        Run a simulation for a given simulation condition.

        :param p:
            Parameters for the simulation condition
        :param ts_dyn:
            (Padded) dynamic time points
        :param ts_posteq:
            (Padded) post-equilibrium time points
        :param my:
            (Padded) measurements
        :param iys:
            (Padded) observable indices
        :param iy_trafos:
            (Padded) observable transformations indices
        :param ops:
            (Padded) observable parameters
        :param nps:
            (Padded) noise parameters
        :param mask_reinit:
            Mask for states that need reinitialisation
        :param x_reinit:
            Reinitialisation values for states
        :param solver:
            ODE solver to use for simulation
        :param controller:
            Step size controller to use for simulation
        :param steady_state_event:
            Steady state event function to use for post-equilibration. Allows customisation of the steady state
            condition, see :func:`diffrax.steady_state_event` for details.
        :param max_steps:
            Maximum number of steps to take during simulation
        :param x_preeq:
            Pre-equilibration state. Can be empty if no pre-equilibration is available, in which case the states will
            be initialised to the model default values.
        :param ts_mask:
            padding mask, see :meth:`JAXModel.simulate_condition` for details.
        :param ret:
            which output to return. See :class:`ReturnValue` for available options.
        :return:
            Tuple of output value and simulation statistics
        """
        return self.model.simulate_condition(
            p=p,
            ts_dyn=jax.lax.stop_gradient(jnp.array(ts_dyn)),
            ts_posteq=jax.lax.stop_gradient(jnp.array(ts_posteq)),
            my=jax.lax.stop_gradient(jnp.array(my)),
            iys=jax.lax.stop_gradient(jnp.array(iys)),
            iy_trafos=jax.lax.stop_gradient(jnp.array(iy_trafos)),
            nps=nps,
            ops=ops,
            x_preeq=x_preeq,
            mask_reinit=jax.lax.stop_gradient(mask_reinit),
            x_reinit=x_reinit,
            ts_mask=jax.lax.stop_gradient(jnp.array(ts_mask)),
            solver=solver,
            controller=controller,
            max_steps=max_steps,
            steady_state_event=steady_state_event,
            adjoint=diffrax.RecursiveCheckpointAdjoint()
            if ret in (ReturnValue.llh, ReturnValue.chi2)
            else diffrax.DirectAdjoint(),
            ret=ret,
        )

    def run_simulations(
        self,
        simulation_conditions: list[str],
        preeq_array: jt.Float[jt.Array, "ncond *nx"],  # noqa: F821, F722
        solver: diffrax.AbstractSolver,
        controller: diffrax.AbstractStepSizeController,
        steady_state_event: Callable[
            ..., diffrax._custom_types.BoolScalarLike
        ],
        max_steps: jnp.int_,
        ret: ReturnValue = ReturnValue.llh,
    ):
        """
        Run simulations for a list of simulation conditions.

        :param simulation_conditions:
            List of simulation conditions to run simulations for.
        :param preeq_array:
            Matrix of pre-equilibrated states for the simulation conditions. Ordering must match the simulation
            conditions. If no pre-equilibration is available for a condition, the corresponding row must be empty.
        :param solver:
            ODE solver to use for simulation.
        :param controller:
            Step size controller to use for simulation.
        :param steady_state_event:
            Steady state event function to use for post-equilibration. Allows customisation of the steady state
            condition, see :func:`diffrax.steady_state_event` for details.
        :param max_steps:
            Maximum number of steps to take during simulation.
        :param ret:
            which output to return. See :class:`ReturnValue` for available options.
        :return:
            Output value and condition specific results and statistics. Results and statistics are returned as a dict
            with arrays with the leading dimension corresponding to the simulation conditions.
        """
        p_array, mask_reinit_array, x_reinit_array, op_array, np_array = (
            self._prepare_conditions(
                simulation_conditions,
                self._op_numeric,
                self._op_mask,
                self._op_indices,
                self._np_numeric,
                self._np_mask,
                self._np_indices,
            )
        )
        return self.run_simulation(
            p_array,
            self._ts_dyn,
            self._ts_posteq,
            self._my,
            self._iys,
            self._iy_trafos,
            op_array,
            np_array,
            mask_reinit_array,
            x_reinit_array,
            solver,
            controller,
            steady_state_event,
            max_steps,
            preeq_array,
            self._ts_masks,
            ret,
        )

    @eqx.filter_vmap(
        in_axes={
            "max_steps": None,
            "self": None,
        },  # only list arguments here where eqx.is_array(0) is not the right thing
    )
    def run_preequilibration(
        self,
        p: jt.Float[jt.Array, "np"],  # noqa: F821, F722
        mask_reinit: jt.Bool[jt.Array, "nx"],  # noqa: F821, F722
        x_reinit: jt.Float[jt.Array, "nx"],  # noqa: F821, F722
        solver: diffrax.AbstractSolver,
        controller: diffrax.AbstractStepSizeController,
        steady_state_event: Callable[
            ..., diffrax._custom_types.BoolScalarLike
        ],
        max_steps: jnp.int_,
    ) -> tuple[jt.Float[jt.Array, "nx"], dict]:  # noqa: F821
        """
        Run a pre-equilibration simulation for a given simulation condition.

        :param p:
            Parameters for the simulation condition
        :param mask_reinit:
            Mask for states that need reinitialisation
        :param x_reinit:
            Reinitialisation values for states
        :param solver:
            ODE solver to use for simulation
        :param controller:
            Step size controller to use for simulation
        :param steady_state_event:
            Steady state event function to use for pre-equilibration. Allows customisation of the steady state
            condition, see :func:`diffrax.steady_state_event` for details.
        :param max_steps:
            Maximum number of steps to take during simulation
        :return:
            Pre-equilibration state
        """
        return self.model.preequilibrate_condition(
            p=p,
            mask_reinit=mask_reinit,
            x_reinit=x_reinit,
            solver=solver,
            controller=controller,
            max_steps=max_steps,
            steady_state_event=steady_state_event,
        )

    def run_preequilibrations(
        self,
        simulation_conditions: list[str],
        solver: diffrax.AbstractSolver,
        controller: diffrax.AbstractStepSizeController,
        steady_state_event: Callable[
            ..., diffrax._custom_types.BoolScalarLike
        ],
        max_steps: jnp.int_,
    ):
        p_array, mask_reinit_array, x_reinit_array, _, _ = (
            self._prepare_conditions(simulation_conditions, None, None)
        )
        return self.run_preequilibration(
            p_array,
            mask_reinit_array,
            x_reinit_array,
            solver,
            controller,
            steady_state_event,
            max_steps,
        )


def run_simulations(
    problem: JAXProblem,
    simulation_conditions: Iterable[tuple[str, ...]] | None = None,
    solver: diffrax.AbstractSolver = diffrax.Kvaerno5(),
    controller: diffrax.AbstractStepSizeController = diffrax.PIDController(
        **DEFAULT_CONTROLLER_SETTINGS
    ),
    steady_state_event: Callable[
        ..., diffrax._custom_types.BoolScalarLike
    ] = diffrax.steady_state_event(),
    max_steps: int = 2**10,
    ret: ReturnValue | str = ReturnValue.llh,
):
    """
    Run simulations for a problem.

    :param problem:
        Problem to run simulations for.
    :param simulation_conditions:
        Simulation conditions to run simulations for. This is a series of tuples, where each tuple contains the
        simulation condition or the pre-equilibration condition followed by the simulation condition. Default is to run
        simulations for all conditions.
    :param solver:
        ODE solver to use for simulation.
    :param controller:
        Step size controller to use for simulation.
    :param steady_state_event:
        Steady state event function to use for pre-/post-equilibration. Allows customisation of the steady state
        condition, see :func:`diffrax.steady_state_event` for details.
    :param max_steps:
        Maximum number of steps to take during simulation.
    :param ret:
        which output to return. See :class:`ReturnValue` for available options.
    :return:
        Overall output value and condition specific results and statistics.
    """
    if isinstance(ret, str):
        ret = ReturnValue[ret]

    if simulation_conditions is None:
        simulation_conditions = problem.get_all_simulation_conditions()

    dynamic_conditions = [sc[0] for sc in simulation_conditions]
    preequilibration_conditions = list(
        {sc[1] for sc in simulation_conditions if len(sc) > 1}
    )

    conditions = {
        "dynamic_conditions": dynamic_conditions,
        "preequilibration_conditions": preequilibration_conditions,
        "simulation_conditions": simulation_conditions,
    }

    if preequilibration_conditions:
        preeqs, preresults = problem.run_preequilibrations(
            preequilibration_conditions,
            solver,
            controller,
            steady_state_event,
            max_steps,
        )
    else:
        preresults = {
            "stats_preeq": None,
        }

    if dynamic_conditions:
        preeq_array = jnp.stack(
            [
                preeqs[preequilibration_conditions.index(sc[1]), :]
                if len(sc) > 1
                else jnp.array([])
                for sc in simulation_conditions
            ]
        )
        output, results = problem.run_simulations(
            dynamic_conditions,
            preeq_array,
            solver,
            controller,
            steady_state_event,
            max_steps,
            ret,
        )
    else:
        output = jnp.array(0.0)
        results = {
            "llh": jnp.array([]),
            "stats_dyn": None,
            "stats_posteq": None,
            "ts": jnp.array([]),
            "x": jnp.array([]),
        }

    if ret in (ReturnValue.llh, ReturnValue.chi2):
        output = jnp.sum(output)

    return output, results | preresults | conditions


def petab_simulate(
    problem: JAXProblem,
    solver: diffrax.AbstractSolver = diffrax.Kvaerno5(),
    controller: diffrax.AbstractStepSizeController = diffrax.PIDController(
        **DEFAULT_CONTROLLER_SETTINGS
    ),
    steady_state_event: Callable[
        ..., diffrax._custom_types.BoolScalarLike
    ] = diffrax.steady_state_event(),
    max_steps: int = 2**10,
):
    """
    Run simulations for a problem and return the results as a petab simulation dataframe.

    :param problem:
        Problem to run simulations for.
    :param solver:
        ODE solver to use for simulation.
    :param controller:
        Step size controller to use for simulation.
    :param max_steps:
        Maximum number of steps to take during simulation.
    :param steady_state_event:
        Steady state event function to use for pre-/post-equilibration. Allows customisation of the steady state
        condition, see :func:`diffrax.steady_state_event` for details.
    :return:
        petab simulation dataframe.
    """
    y, r = run_simulations(
        problem,
        solver=solver,
        controller=controller,
        steady_state_event=steady_state_event,
        max_steps=max_steps,
        ret=ReturnValue.y,
    )
    dfs = []
    for ic, sc in enumerate(r["dynamic_conditions"]):
        obs = [
            problem.model.observable_ids[io]
            for io in problem._iys[ic, problem._ts_masks[ic, :]]
        ]
        t = jnp.concat(
            (
                problem._ts_dyn[ic, :],
                problem._ts_posteq[ic, :],
            )
        )
        df_sc = pd.DataFrame(
            {
                petab.SIMULATION: y[ic, problem._ts_masks[ic, :]],
                petab.TIME: t[problem._ts_masks[ic, :]],
                petab.OBSERVABLE_ID: obs,
                petab.SIMULATION_CONDITION_ID: [sc] * len(t),
            },
            index=problem._petab_measurement_indices[ic, :],
        )
        if (
            petab.OBSERVABLE_PARAMETERS
            in problem._petab_problem.measurement_df
        ):
            df_sc[petab.OBSERVABLE_PARAMETERS] = (
                problem._petab_problem.measurement_df.query(
                    f"{petab.SIMULATION_CONDITION_ID} == '{sc}'"
                )[petab.OBSERVABLE_PARAMETERS]
            )
        if petab.NOISE_PARAMETERS in problem._petab_problem.measurement_df:
            df_sc[petab.NOISE_PARAMETERS] = (
                problem._petab_problem.measurement_df.query(
                    f"{petab.SIMULATION_CONDITION_ID} == '{sc}'"
                )[petab.NOISE_PARAMETERS]
            )
        if (
            petab.PREEQUILIBRATION_CONDITION_ID
            in problem._petab_problem.measurement_df
        ):
            df_sc[petab.PREEQUILIBRATION_CONDITION_ID] = (
                problem._petab_problem.measurement_df.query(
                    f"{petab.SIMULATION_CONDITION_ID} == '{sc}'"
                )[petab.PREEQUILIBRATION_CONDITION_ID]
            )
        dfs.append(df_sc)
    return pd.concat(dfs).sort_index()
