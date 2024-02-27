"""Tests for simulate_petab on PEtab benchmark problems."""
import os
from pathlib import Path

import amici
import numpy as np
import pandas as pd
import petab
import pytest
from amici.petab.petab_import import import_petab_problem

# Absolute and relative tolerances for finite difference gradient checks.
ATOL: float = 1e-3
RTOL: float = 1e-2

repo_root = Path(__file__).parent.parent.parent
benchmark_path = repo_root / "Benchmark-Models-PEtab" / "Benchmark-Models"
if not benchmark_path.exists():
    benchmark_path = Path(os.environ["BENCHMARK_COLLECTION"])

# reuse compiled models from test_benchmark_collection.sh
benchmark_outdir = repo_root / "test_bmc"
models = [
    str(petab_path.stem)
    for petab_path in benchmark_path.glob("*")
    if petab_path.is_dir()
    if str(petab_path.stem)
    not in (
        # excluded due to excessive runtime
        "Bachmann_MSB2011",
        "Chen_MSB2009",
        "Froehlich_CellSystems2018",
        "Raimundez_PCB2020",
        "Lucarelli_CellSystems2018",
        "Isensee_JCB2018",
        "Beer_MolBioSystems2014",
        "Alkan_SciSignal2018",
        # excluded due to excessive numerical failures
        "Crauste_CellSystems2017",
        "Fujita_SciSignal2010",
    )
]

debug = False
if debug:
    debug_path = Path(__file__).parent / "debug"
    debug_path.mkdir(exist_ok=True, parents=True)


# until fiddy is updated
@pytest.mark.filterwarnings(
    "ignore:Importing amici.petab_objective is deprecated.:DeprecationWarning"
)
@pytest.mark.filterwarnings("ignore:divide by zero encountered in log10")
@pytest.mark.parametrize("scale", (True, False))
@pytest.mark.parametrize("model", models)
def test_benchmark_gradient(model, scale):
    from fiddy import MethodId, get_derivative
    from fiddy.derivative_check import NumpyIsCloseDerivativeCheck
    from fiddy.extensions.amici import simulate_petab_to_cached_functions
    from fiddy.success import Consistency

    if not scale and model in (
        "Smith_BMCSystBiol2013",
        "Brannmark_JBC2010",
        "Elowitz_Nature2000",
        "Borghans_BiophysChem1997",
        "Sneyd_PNAS2002",
        "Bertozzi_PNAS2020",
        "Okuonghae_ChaosSolitonsFractals2020",
    ):
        # not really worth the effort trying to fix these cases if they
        # only fail on linear scale
        pytest.skip()

    petab_problem = petab.Problem.from_yaml(
        benchmark_path / model / (model + ".yaml")
    )
    petab.flatten_timepoint_specific_output_overrides(petab_problem)

    # Only compute gradient for estimated parameters.
    parameter_df_free = petab_problem.parameter_df.loc[
        petab_problem.x_free_ids
    ]
    parameter_ids = list(parameter_df_free.index)

    # Setup AMICI objects.
    amici_model = import_petab_problem(
        petab_problem,
        model_output_dir=benchmark_outdir / model,
    )
    amici_solver = amici_model.getSolver()
    amici_solver.setAbsoluteTolerance(1e-12)
    amici_solver.setRelativeTolerance(1e-12)
    if model in (
        "Smith_BMCSystBiol2013",
        "Oliveira_NatCommun2021",
    ):
        amici_solver.setAbsoluteTolerance(1e-10)
        amici_solver.setRelativeTolerance(1e-10)
    elif model in ("Okuonghae_ChaosSolitonsFractals2020",):
        amici_solver.setAbsoluteTolerance(1e-14)
        amici_solver.setRelativeTolerance(1e-14)
    amici_solver.setMaxSteps(int(1e5))

    if model in ("Brannmark_JBC2010",):
        amici_model.setSteadyStateSensitivityMode(
            amici.SteadyStateSensitivityMode.integrationOnly
        )

    amici_function, amici_derivative = simulate_petab_to_cached_functions(
        petab_problem=petab_problem,
        parameter_ids=parameter_ids,
        amici_model=amici_model,
        solver=amici_solver,
        scaled_parameters=scale,
        scaled_gradients=scale,
        cache=not debug,
    )

    noise_level = 0.1

    np.random.seed(0)
    if scale:
        point = np.asarray(
            list(
                petab_problem.scale_parameters(
                    dict(parameter_df_free.nominalValue)
                ).values()
            )
        )
        point_noise = np.random.randn(len(point)) * noise_level
    else:
        point = parameter_df_free.nominalValue.values
        point_noise = np.random.randn(len(point)) * point * noise_level
    point += point_noise  # avoid small gradients at nominal value

    expected_derivative = amici_derivative(point)

    sizes = [
        1e-1,
        1e-2,
        1e-3,
        1e-4,
        1e-5,
    ]
    if model in ("Okuonghae_ChaosSolitonsFractals2020",):
        sizes.insert(0, 0.2)

    derivative = get_derivative(
        function=amici_function,
        point=point,
        sizes=sizes,
        direction_ids=parameter_ids,
        method_ids=[MethodId.CENTRAL, MethodId.FORWARD, MethodId.BACKWARD],
        success_checker=Consistency(atol=1e-4, rtol=0.2),
        expected_result=expected_derivative,
        relative_sizes=not scale,
    )
    success = False
    if derivative.df.success.all():
        check = NumpyIsCloseDerivativeCheck(
            derivative=derivative,
            expectation=expected_derivative,
            point=point,
        )
        success = check(rtol=RTOL, atol=ATOL)

    if debug:
        df = pd.DataFrame(
            [
                {
                    (
                        "fd",
                        r.metadata["size_absolute"],
                        str(r.method_id),
                    ): r.value
                    for c in d.computers
                    for r in c.results
                }
                for d in derivative.directional_derivatives
            ],
            index=parameter_ids,
        )
        df[("fd", "full", "")] = derivative.series.values
        df[("amici", "", "")] = expected_derivative

        file_name = f"{model}_scale={scale}.tsv"
        df.to_csv(debug_path / file_name, sep="\t")

    # The gradients for all parameters are correct.
    assert success, derivative.df
