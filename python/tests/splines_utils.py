"""
Utilities for creating test SBML models containing splines,
for running them and for comparing them to a symbolically
computed ground truth.
"""

import math
import os
import uuid
from tempfile import mkdtemp
from typing import Any, Optional, Union
from collections.abc import Sequence

import amici
import numpy as np
import pandas as pd
import petab
import sympy as sp
from amici.gradient_check import _check_results
from amici.petab.petab_import import import_petab_problem
from amici.petab.simulations import EDATAS, LLH, RDATAS, SLLH, simulate_petab
from amici.sbml_utils import (
    add_compartment,
    add_inflow,
    add_parameter,
    add_rate_rule,
    add_species,
    amici_time_symbol,
    create_sbml_model,
)
from amici.splines import AbstractSpline, CubicHermiteSpline, UniformGrid
from amici.testing import TemporaryDirectoryWinSafe as TemporaryDirectory
from petab.models.sbml_model import SbmlModel


def evaluate_spline(
    spline: AbstractSpline, params: dict, tt: Sequence[float], **kwargs
):
    """
    Evaluate the `AbstractSpline` `spline` at timepoints `tt`
    for the parameters given in the dictionary `params`.
    """
    return np.asarray([spline.evaluate(t).subs(params) for t in tt], **kwargs)


def integrate_spline(
    spline: AbstractSpline,
    params: Union[dict, None],
    tt: Sequence[float],
    initial_value: float = 0,
):
    """
    Integrate the `AbstractSpline` `spline` at timepoints `tt`
    for the parameters given in the dictionary `params`.
    """
    ispline = [initial_value + spline.integrate(0, t) for t in tt]
    if params is not None:
        ispline = [x.subs(params) for x in ispline]
    return ispline


def create_condition_table() -> pd.DataFrame:
    """Create a PEtab condition table."""
    condition_df = pd.DataFrame({"conditionId": ["condition1"]})
    condition_df.set_index(["conditionId"], inplace=True)
    return condition_df


def create_parameter_table(**columns) -> pd.DataFrame:
    """Create a PEtab parameter table."""
    if isinstance(columns["parameterId"], str):
        columns["parameterId"] = [columns["parameterId"]]
    columns.setdefault("parameterScale", "lin")
    columns.setdefault("estimate", 1)
    parameter_df = pd.DataFrame(columns)
    parameter_df.set_index(["parameterId"], inplace=True)
    return parameter_df


def create_observable_table(**columns) -> pd.DataFrame:
    """Create a PEtab observable table."""
    if isinstance(columns["observableId"], str):
        columns["observableId"] = [columns["observableId"]]
    columns.setdefault("observableTransformation", "lin")
    columns.setdefault("noiseDistribution", "normal")
    observable_df = pd.DataFrame(columns)
    observable_df.set_index(["observableId"], inplace=True)
    return observable_df


def create_measurement_table(**columns) -> pd.DataFrame:
    """Create a PEtab measurement table."""
    if isinstance(columns["observableId"], str):
        columns["observableId"] = [columns["observableId"]]
    columns.setdefault("simulationConditionId", "condition1")
    return pd.DataFrame(columns)


def species(i) -> str:
    """Name to use for the `i`-th species."""
    return f"z{i}"


def observable(i) -> str:
    """
    Name to use for the `i`-th observable,
    i.e., the observable associated to the
    `i`-th species.
    """
    return f"{species(i)}_obs"


def species_to_index(name) -> int:
    """Get the species index from a species name."""
    assert name[0] == "z"
    return int(name[1:])


def create_petab_problem(
    splines: list[AbstractSpline],
    params_true: dict,
    initial_values: Optional[np.ndarray] = None,
    use_reactions: bool = False,
    measure_upsample: int = 6,
    sigma: float = 1.0,
    t_extrapolate: float = 0.25,
    folder: Optional[str] = None,
    model_name: str = "test_splines",
):
    """
    Given a list of `AbstractSplines`, create a PEtab problem for the system of
    ODEs given by `z_i(t)' = spline[i](t)`.

    :param params_true:
        parameter values used to compute the analytical solution of the ODE
        system in order to fill the PEtab measurement table

    :param initial_values:
        initial values of the state variables

    :param use_reactions:
        whether the ODEs are encoded in the SBML model as reactions (inflows)
        or rate rules

    :param measure_upsample:
        controls the number of time points at which synthetic measurements are
        taken. The interval between subsequent time points is equal to the
        smallest interval between subsequent spline nodes divided by
        `measure_upsample`

    :param sigma:
        standard deviation for additive Normal noise used to corrupt synthetic
        measurements

    :param t_extrapolate:
        factor controlling how long after the final spline node the simulation
        should continue in order to test extrapolation methods.

    :param folder:
        if not `None`, save the PEtab problem to this folder

    :param model_name:
        name of the SBML model to be created
    """

    for spline in splines:
        if spline.evaluate_at != amici_time_symbol:
            raise ValueError(
                "the given splines must be evaluated at the simulation time"
            )

    if initial_values is None:
        initial_values = np.zeros(len(splines))

    # Create SBML document
    doc, model = create_sbml_model(model_name)
    add_compartment(model, "compartment")
    for i, spline in enumerate(splines):
        spline.add_to_sbml_model(model)
        add_species(model, species(i), initial_amount=initial_values[i])
        if use_reactions:
            add_inflow(model, species(i), splines[i].sbml_id)
        else:
            add_rate_rule(model, species(i), splines[i].sbml_id)
    for parId, value in params_true.items():
        add_parameter(model, parId, value=value, constant=True)
    for spline in splines:
        add_parameter(model, spline.sbml_id, constant=False)

    # Compute simulation time
    # Must cover all the intervals of definition for the splines,
    # plus something extra for extrapolated or periodic splines
    T = 0
    for spline in splines:
        if spline.extrapolate[0] is None and spline.nodes[0] > 0:
            raise ValueError(
                "if no left-extrapolation is defined for a spline, "
                "its interval of definition should contain zero"
            )
        if spline.extrapolate[1] is not None:
            f = (
                t_extrapolate
                if spline.extrapolate[1] != "periodic"
                else 1 + t_extrapolate
            )
            DT = f * (spline.nodes[-1] - spline.nodes[0])
        else:
            DT = 0
        T = max(T, spline.nodes[-1] + DT)

    # Compute synthetic measurements
    dt = min(np.diff(spline.nodes).min() for spline in splines)
    dt /= measure_upsample
    n_obs = math.ceil(T / dt) + 1
    tt_obs = np.linspace(0, float(T), n_obs)
    zz_true = np.array(
        [
            integrate_spline(spline, params_true, tt_obs, iv)
            for (spline, iv) in zip(splines, initial_values)
        ],
        dtype=float,
    )
    zz_obs = [zz + sigma * np.random.randn(len(zz)) for zz in zz_true]

    # Create PEtab tables
    condition_df = create_condition_table()
    # ensure that same parameter order is used for all columns
    _params = list(params_true.items())
    parameter_df = create_parameter_table(
        parameterId=[p.name for (p, v) in _params],
        lowerBound=min(v for (p, v) in _params) if _params else [],
        upperBound=max(v for (p, v) in _params) if _params else [],
        nominalValue=[v for (p, v) in _params],
        estimate=1,
    )
    observable_df = create_observable_table(
        observableId=[observable(i) for i in range(len(splines))],
        observableFormula=[species(i) for i in range(len(splines))],
        noiseFormula=sigma if sigma > 0 else 1.0,
    )
    measurement_df = create_measurement_table(
        observableId=np.concatenate(
            [len(tt_obs) * [observable(i)] for i in range(len(splines))]
        ),
        time=len(splines) * list(tt_obs),
        measurement=np.concatenate(zz_obs),
    )

    # Create and validate PEtab problem
    problem = petab.Problem(
        model=SbmlModel(
            sbml_document=doc,
            sbml_model=model,
        ),
        condition_df=condition_df,
        measurement_df=measurement_df,
        parameter_df=parameter_df,
        observable_df=observable_df,
    )
    if petab.lint_problem(problem):
        raise RuntimeError("PEtab lint failed")

    # Write PEtab problem to disk
    if folder is None:
        return problem, initial_values, T
    folder = os.path.abspath(folder)
    os.makedirs(folder, exist_ok=True)
    problem.to_files(
        model_file=os.path.join(folder, f"{model_name}_model.xml"),
        condition_file=os.path.join(folder, f"{model_name}_conditions.tsv"),
        measurement_file=os.path.join(
            folder, f"{model_name}_measurements.tsv"
        ),
        parameter_file=os.path.join(folder, f"{model_name}_parameters.tsv"),
        observable_file=os.path.join(folder, f"{model_name}_observables.tsv"),
        yaml_file=os.path.join(folder, f"{model_name}.yaml"),
    )
    return os.path.join(folder, f"{model_name}.yaml"), initial_values, T


def simulate_splines(
    splines,
    params_true,
    initial_values=None,
    *,
    folder: Optional[str] = None,
    keep_temporary: bool = False,
    benchmark: Union[bool, int] = False,
    rtol: float = 1e-12,
    atol: float = 1e-12,
    maxsteps: int = 500_000,
    discard_annotations: bool = False,
    use_adjoint: bool = False,
    skip_sensitivity: bool = False,
    petab_problem=None,
    amici_model=None,
    **kwargs,
):
    """
    Create a PEtab problem using `create_petab_problem` and simulate it with
    AMICI.

    :param splines:
        passed to `create_petab_problem`

    :param params_true:
        passed to `create_petab_problem`

    :param initial_values:
        passed to `create_petab_problem`

    :param folder:
        working directory (a temporary one is used if not specified)

    :param keep_temporary:
        whether to keep or delete temporary working directories on exit

    :param benchmark:
        instead of returning the simulation data, run the simulation
        `benchmark` times (defaults to `50` if `benchmark` is `True`)
        and return execution times

    :param rtol:
        relative tolerance for AMICI solver

    :param atol:
        absolute tolerance for AMICI solver

    :param maxsteps:
        maximum number of steps for AMICI solver

    :param discard_annotations:
        whether to discard spline annotations,
        forcing AMICI to read the spline as a piecewise assignment rule

    :param use_adjoint:
        whether to use adjoint sensitivity computation

    :param skip_sensitivity:
        whether to skip sensitivity computation

    :param petab_problem:
        PEtab problem (if already created)

    :param amici_model:
        AMICI model (if already created)

    :param kwargs:
        passed to `create_petab_problem`
    """
    # If no working directory is given, create a temporary one
    if folder is None:
        if keep_temporary:
            folder = mkdtemp()
            print(f"temporary folder is {folder}")
        else:
            with TemporaryDirectory() as folder:
                return simulate_splines(
                    splines,
                    params_true,
                    initial_values,
                    folder=folder,
                    benchmark=benchmark,
                    rtol=rtol,
                    atol=atol,
                    maxsteps=maxsteps,
                    discard_annotations=discard_annotations,
                    use_adjoint=use_adjoint,
                    skip_sensitivity=skip_sensitivity,
                    petab_problem=petab_problem,
                    amici_model=amici_model,
                    **kwargs,
                )

    if petab_problem is None and amici_model is not None:
        raise ValueError(
            "if amici_model is given, petab_problem must be given too"
        )

    if petab_problem is not None and initial_values is None:
        raise ValueError(
            "if petab_problem is given, initial_values must be given too"
        )

    if petab_problem is None:
        # Create PEtab problem
        path, initial_values, T = create_petab_problem(
            splines,
            params_true,
            initial_values,
            sigma=0.0,
            folder=folder,
            **kwargs,
        )
        petab_problem = petab.Problem.from_yaml(path)

    if amici_model is None:
        # Create and compile AMICI model
        model_id = uuid.uuid4().hex[-5:]  # to prevent folder/module collisions
        amici_model = import_petab_problem(
            petab_problem,
            discard_sbml_annotations=discard_annotations,
            model_output_dir=os.path.join(folder, f"amici_models_{model_id}"),
            model_name=f"splinetest_{model_id}",
        )

    # Set solver options
    solver = amici_model.getSolver()
    solver.setRelativeTolerance(rtol)
    solver.setAbsoluteTolerance(atol)
    solver.setMaxSteps(maxsteps)
    if not skip_sensitivity:
        solver.setSensitivityOrder(amici.SensitivityOrder.first)
        if use_adjoint:
            solver.setSensitivityMethod(amici.SensitivityMethod.adjoint)
        else:
            solver.setSensitivityMethod(amici.SensitivityMethod.forward)

    # Compute and set timepoints
    # NB not working, will always be equal to the observation times
    # n = max(len(spline.nodes) for spline in splines) * simulate_upsample
    # tt = np.linspace(0, float(T), n)
    # model.setTimepoints(tt)

    # Create dictionary for parameter values
    params_str = {p.name: v for (p, v) in params_true.items()}

    if benchmark is False:
        # Simulate PEtab problem
        res = simulate_petab(petab_problem, amici_model, solver, params_str)
        llh, rdatas, edatas = res[LLH], res[RDATAS], res[EDATAS]
        assert len(rdatas) == 1
        llh = float(llh)
        rdata = rdatas[0]
        assert SLLH in rdata.keys()
        sllh = rdata[SLLH]
        assert len(edatas) == 1
        edata = edatas[0]

        # Return state/parameter ordering
        state_ids = amici_model.getStateIds()
        param_ids = amici_model.getParameterIds()

        return (
            initial_values,
            petab_problem,
            amici_model,
            solver,
            llh,
            sllh,
            rdata,
            edata,
            state_ids,
            param_ids,
        )

    if benchmark is True:
        benchmark = 50
    import time

    runtimes = []
    for _ in range(int(benchmark)):
        t0 = time.perf_counter()
        simulate_petab(petab_problem, amici_model, solver, params_str)
        t_elapsed = time.perf_counter() - t0
        runtimes.append(t_elapsed)

    return dict(
        runtimes=runtimes,
        mean=np.mean(runtimes),
        median=np.median(runtimes),
        min=min(runtimes),
        max=max(runtimes),
    )


def compute_ground_truth(
    splines, initial_values, times, params_true, params_sorted
):
    x_true_sym = sp.Matrix(
        [
            integrate_spline(spline, None, times, iv)
            for (spline, iv) in zip(splines, initial_values)
        ]
    ).transpose()
    groundtruth = {
        "x_true": np.asarray(x_true_sym.subs(params_true), dtype=float)
    }
    sx_by_state = [
        x_true_sym[:, i].jacobian(params_sorted).subs(params_true)
        for i in range(x_true_sym.shape[1])
    ]
    sx_by_state = [np.asarray(sx, dtype=float) for sx in sx_by_state]
    groundtruth["sx_true"] = np.concatenate(
        [sx[:, :, np.newaxis] for sx in sx_by_state], axis=2
    )
    return groundtruth


def check_splines(
    splines,
    params_true,
    initial_values=None,
    *,
    discard_annotations: bool = False,
    use_adjoint: bool = False,
    skip_sensitivity: bool = False,
    debug: Union[bool, str] = False,
    parameter_lists: Optional[Sequence[Sequence[int]]] = None,
    llh_rtol: float = 1e-8,
    sllh_atol: float = 1e-8,
    x_rtol: float = 1e-11,
    x_atol: float = 1e-11,
    w_rtol: float = 1e-11,
    w_atol: float = 1e-11,
    sx_rtol: float = 1e-10,
    sx_atol: float = 1e-10,
    groundtruth: Optional[Union[str, dict[str, Any]]] = None,
    **kwargs,
):
    """
    Create a PEtab problem using `create_petab_problem`,
    simulate it with `simulate_splines`
    and check it against the analytical solution.

    :param splines:
        passed to `simulate_splines`

    :param params_true:
        passed to `simulate_splines`

    :param initial_values:
        passed to `simulate_splines`

    :param discard_annotations:
        whether to discard spline annotations,
        forcing AMICI to read the spline as a piecewise assignment rule

    :param use_adjoint:
        whether to use adjoint sensitivity computation

    :param skip_sensitivity:
        whether to skip sensitivity computation

    :param debug:
        if not `False`, do not check and return results and ground truth
        instead.
        If equal to `'print'`, in addition to the above print error values.

    :param parameter_lists:
        Set AMICI parameter list to these values,
        in order to check that partial sensitivity computation works.

    :param kwargs:
        passed to `simulate_splines`
    """
    if isinstance(splines, AbstractSpline):
        splines = [splines]

    # Simulate PEtab problem
    (
        initial_values,
        petab_problem,
        amici_model,
        amici_solver,
        llh,
        sllh,
        rdata,
        edata,
        state_ids,
        param_ids,
    ) = simulate_splines(
        splines,
        params_true,
        initial_values,
        discard_annotations=discard_annotations,
        skip_sensitivity=skip_sensitivity,
        use_adjoint=use_adjoint,
        **kwargs,
    )

    tt = rdata["ts"]

    # Sort splines/ics/parameters as in the AMICI model
    splines = [splines[species_to_index(name)] for name in state_ids]
    initial_values = [
        initial_values[species_to_index(name)] for name in state_ids
    ]

    def param_by_name(id):
        for p in params_true.keys():
            if p.name == id:
                return p
        assert False

    params_sorted = [param_by_name(id) for id in param_ids]

    # Check states
    if groundtruth == "compute":
        groundtruth = compute_ground_truth(
            splines, initial_values, tt, params_true, params_sorted
        )
    if groundtruth is None:
        x_true_sym = sp.Matrix(
            [
                integrate_spline(spline, None, tt, iv)
                for (spline, iv) in zip(splines, initial_values)
            ]
        ).transpose()
        x_true = np.asarray(x_true_sym.subs(params_true), dtype=float)
    else:
        x_true = groundtruth["x_true"]
    if not debug:
        assert rdata.x.shape == x_true.shape
        _check_results(rdata, "x", x_true, atol=x_atol, rtol=x_rtol)
    elif debug == "print":
        x_err_abs = abs(rdata["x"] - x_true)
        x_err_rel = np.where(x_err_abs == 0, 0, x_err_abs / abs(x_true))
        print(f"x_atol={x_atol} x_rtol={x_rtol}")
        print("x_err_abs:")
        print(np.squeeze(x_err_abs))
        print("x_err_abs (maximum):")
        print(x_err_abs.max())
        print("x_err_rel:")
        print(np.squeeze(x_err_rel))
        print("x_err_rel (maximum):")
        print(x_err_rel.max())

    # Check spline evaluations
    # TODO can we know how the splines are ordered inside w?
    if False and discard_annotations and len(splines) == 1:
        assert rdata["w"].shape[1] == 1
        w_true = np.column_stack(
            [
                evaluate_spline(spline, params_true, tt, dtype=float)
                for spline in splines
            ]
        )
        if not debug:
            _check_results(
                rdata,
                "w",
                w_true,
                atol=w_atol,
                rtol=w_rtol,
            )
        elif debug == "print":
            w_err_abs = abs(rdata["w"] - w_true)
            w_err_rel = np.where(w_err_abs == 0, 0, w_err_abs / abs(w_true))
            print(f"w_atol={w_atol} w_rtol={w_rtol}")
            print("w_err_abs:")
            print(np.squeeze(w_err_abs))
            print("w_err_abs (maximum):")
            print(w_err_abs.max())
            print("w_err_rel:")
            print(np.squeeze(w_err_rel))
            print("w_err_rel (maximum):")
            print(w_err_rel.max())
    else:
        w_true = None

    # Check sensitivities
    if params_sorted and not use_adjoint:
        if skip_sensitivity:
            pass
        if groundtruth is None:
            sx_by_state = [
                x_true_sym[:, i].jacobian(params_sorted).subs(params_true)
                for i in range(x_true_sym.shape[1])
            ]
            sx_by_state = [np.asarray(sx, dtype=float) for sx in sx_by_state]
            sx_true = np.concatenate(
                [sx[:, :, np.newaxis] for sx in sx_by_state], axis=2
            )
        else:
            sx_true = groundtruth["sx_true"]
        if not debug:
            assert rdata.sx.shape == sx_true.shape
            _check_results(
                rdata,
                "sx",
                sx_true,
                atol=sx_atol,
                rtol=sx_rtol,
            )
        elif debug == "print":
            sx_err_abs = abs(rdata["sx"] - sx_true)
            sx_err_rel = np.where(
                sx_err_abs == 0, 0, sx_err_abs / abs(sx_true)
            )
            print(f"sx_atol={sx_atol} sx_rtol={sx_rtol}")
            print("sx_err_abs:")
            print(np.squeeze(sx_err_abs))
            print("sx_err_abs (maximum):")
            print(sx_err_abs.max())
            print("sx_err_rel:")
            print(np.squeeze(sx_err_rel))
            print("sx_err_rel (maximum):")
            print(sx_err_rel.max())
    else:
        assert rdata["sx"] is None

    # Check log-likelihood
    llh_true = -0.5 * rdata["y"].size * np.log(2 * np.pi)
    llh_error_rel = abs(llh - llh_true) / abs(llh_true)
    if (llh_error_rel > llh_rtol and debug is not True) or debug == "print":
        print(f"{llh_rtol=}")
        print(f"{llh_error_rel=}")
    if not debug:
        assert llh_error_rel <= llh_rtol

    # Check log-likelihood sensitivities
    # (should be all zero, since we simulated with the true parameters)
    if params_sorted:
        if not skip_sensitivity:
            if sllh_atol is None:
                sllh_atol = np.finfo(float).eps
            sllh_err_abs = abs(sllh).max()
            if (
                sllh_err_abs > sllh_atol and debug is not True
            ) or debug == "print":
                print(f"sllh_atol={sllh_atol}")
                print(f"sllh_err_abs = {sllh_err_abs}")
            if not debug:
                assert sllh_err_abs <= sllh_atol
    else:
        assert sllh is None

    # Try different parameter lists
    if (
        not skip_sensitivity
        and (not use_adjoint)
        and parameter_lists is not None
    ):
        for plist in parameter_lists:
            amici_model.setParameterList(plist)
            amici_model.setTimepoints(rdata.t)
            rdata_partial = amici.runAmiciSimulation(amici_model, amici_solver)
            assert rdata.sx[:, plist, :].shape == rdata_partial.sx.shape
            assert np.allclose(rdata.sx[:, plist, :], rdata_partial.sx)

    if debug:
        return dict(
            splines=splines,
            initial_values=initial_values,
            petab_problem=petab_problem,
            amici_model=amici_model,
            groundtruth=groundtruth,
            rdata=rdata,
            params_true=params_true,
            params_sorted=params_sorted,
            x_true=x_true,
            w_true=w_true,
            sx_true=sx_true,
            llh_true=llh_true,
        )
    else:
        return dict(
            initial_values=initial_values,
            petab_problem=petab_problem,
            amici_model=amici_model,
            groundtruth=groundtruth,
        )


def check_splines_full(
    splines,
    params,
    tols,
    *args,
    check_piecewise: bool = True,
    check_forward: bool = True,
    check_adjoint: bool = True,
    folder: Optional[str] = None,
    groundtruth: Optional[Union[dict, str]] = "compute",
    return_groundtruth: bool = False,
    **kwargs,
):
    """
    Check example PEtab problem with `check_splines`
    both using adjoint and forward sensitivities
    and also in the case in which the splines are read as piecewise functions.
    """
    if folder is None:
        with TemporaryDirectory() as folder:
            return check_splines_full(
                splines,
                params,
                tols,
                *args,
                check_piecewise=check_piecewise,
                check_forward=check_forward,
                check_adjoint=check_adjoint,
                folder=folder,
                groundtruth=groundtruth,
                return_groundtruth=return_groundtruth,
                **kwargs,
            )

    if isinstance(tols, dict):
        tols1 = tols2 = tols3 = tols
    else:
        tols1, tols2, tols3 = tols

    if isinstance(splines, AbstractSpline):
        splines = [splines]

    contains_periodic = any(
        spline.extrapolate == ("periodic", "periodic") for spline in splines
    )

    # Amortize creation of PEtab and AMICI objects
    results = None
    initial_values = None
    petab_problem = None
    amici_model = None

    if check_piecewise and not contains_periodic:
        results = check_splines(
            splines,
            params,
            *args,
            **kwargs,
            **tols1,
            folder=folder,
            discard_annotations=True,
            use_adjoint=False,
            groundtruth=groundtruth,
        )
        initial_values = results["initial_values"]
        petab_problem = results["petab_problem"]
        groundtruth = results["groundtruth"]

    if check_forward:
        results = check_splines(
            splines,
            params,
            *args,
            **kwargs,
            **tols2,
            initial_values=initial_values,
            folder=folder,
            petab_problem=petab_problem,
            use_adjoint=False,
            groundtruth=groundtruth,
        )
        initial_values = results["initial_values"]
        petab_problem = results["petab_problem"]
        amici_model = results["amici_model"]
        groundtruth = results["groundtruth"]

    if check_adjoint:
        results = check_splines(
            splines,
            params,
            *args,
            **kwargs,
            **tols3,
            initial_values=initial_values,
            folder=folder,
            petab_problem=petab_problem,
            amici_model=amici_model,
            use_adjoint=True,
            groundtruth=(
                None if groundtruth == "compute" else groundtruth
            ),  # do not compute sensitivities if possible
        )

    if return_groundtruth:
        if groundtruth is not None and not isinstance(groundtruth, str):
            return groundtruth
        elif results is None:
            return None
        else:
            return results["groundtruth"]


def example_spline_1(
    idx: int = 0,
    offset: float = 0,
    scale: float = 1,
    num_nodes: int = 9,
    fixed_values=None,  # a list of indices or 'all'
    extrapolate=None,
):
    """A simple spline with no extrapolation."""

    yy_true = np.asarray(
        [0.0, 2.0, 5.0, 6.0, 5.0, 4.0, 2.0, 3.0, 4.0, 6.0, 7.0, 7.5, 6.5, 4.0]
    )
    if num_nodes is not None:
        assert 1 < num_nodes <= len(yy_true)
        yy_true = yy_true[:num_nodes]
    yy_true = scale * yy_true + offset
    xx = UniformGrid(0, 25, number_of_nodes=len(yy_true))
    yy = list(sp.symbols(f"y{idx}_0:{len(yy_true)}"))

    if fixed_values is None:
        params = dict(zip(yy, yy_true))
    elif fixed_values == "all":
        params = {}
        for i in range(len(yy_true)):
            yy[i] = yy_true[i]
    else:
        params = {}
        for i in range(len(yy_true)):
            if i in fixed_values:
                yy[i] = yy_true[i]
            else:
                params[yy[i]] = yy_true[i]

    spline = CubicHermiteSpline(
        f"y{idx}",
        nodes=xx,
        values_at_nodes=yy,
        bc=None,
        extrapolate=extrapolate,
    )

    if os.name == "nt":
        tols = (
            dict(llh_rtol=1e-15, x_rtol=1e-8, x_atol=1e-7),
            dict(llh_rtol=1e-15, x_rtol=1e-8, x_atol=1e-7),
            dict(llh_rtol=1e-15, sllh_atol=5e-8, x_rtol=1e-8, x_atol=1e-7),
        )
    else:
        tols = (
            dict(llh_rtol=1e-15),
            dict(llh_rtol=1e-15),
            dict(llh_rtol=1e-15, sllh_atol=5e-8),
        )

    return spline, params, tols


def example_spline_2(idx: int = 0):
    """A simple periodic spline."""
    yy_true = [0.0, 2.0, 3.0, 4.0, 1.0, -0.5, -1, -1.5, 0.5, 0.0]
    xx = UniformGrid(0, 25, number_of_nodes=len(yy_true))
    yy = list(sp.symbols(f"y{idx}_0:{len(yy_true) - 1}"))
    yy.append(yy[0])
    params = dict(zip(yy, yy_true))
    spline = CubicHermiteSpline(
        f"y{idx}",
        nodes=xx,
        values_at_nodes=yy,
        bc="periodic",
        extrapolate="periodic",
    )
    tols = (
        dict(llh_rtol=1e-15),
        dict(llh_rtol=1e-15),
        dict(llh_rtol=1e-15, sllh_atol=5e-8, x_rtol=1e-10, x_atol=5e-10),
    )
    return spline, params, tols


def example_spline_3(idx: int = 0):
    """A simple spline with extrapolation on the right side."""
    yy_true = [0.0, 2.0, 5.0, 6.0, 5.0, 4.0, 2.0, 3.0, 4.0, 6.0]
    xx = UniformGrid(0, 25, number_of_nodes=len(yy_true))
    yy = list(sp.symbols(f"y{idx}_0:{len(yy_true)}"))
    params = dict(zip(yy, yy_true))
    spline = CubicHermiteSpline(
        f"y{idx}",
        nodes=xx,
        values_at_nodes=yy,
        bc=(None, "zeroderivative"),
        extrapolate=(None, "constant"),
    )
    tols = {}
    return spline, params, tols
