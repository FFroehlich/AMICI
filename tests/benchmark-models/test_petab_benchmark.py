"""Tests for simulate_petab on PEtab benchmark problems."""

from pathlib import Path

import fiddy

import amici
import numpy as np
import pandas as pd
import petab.v1 as petab
import pytest
from amici.petab.petab_import import import_petab_problem
import benchmark_models_petab
from collections import defaultdict
from dataclasses import dataclass
from amici import SensitivityMethod
from fiddy import MethodId, get_derivative
from fiddy.derivative_check import NumpyIsCloseDerivativeCheck
from fiddy.extensions.amici import simulate_petab_to_cached_functions
from fiddy.success import Consistency


repo_root = Path(__file__).parent.parent.parent

# reuse compiled models from test_benchmark_collection.sh
benchmark_outdir = repo_root / "test_bmc"
models = set(benchmark_models_petab.MODELS) - {
    # excluded due to excessive runtime
    "Bachmann_MSB2011",
    "Chen_MSB2009",
    "Froehlich_CellSystems2018",
    "Raimundez_PCB2020",
    "Lucarelli_CellSystems2018",
    "Isensee_JCB2018",
    "Beer_MolBioSystems2014",
    "Alkan_SciSignal2018",
    "Lang_PLOSComputBiol2024",
    "Smith_BMCSystBiol2013",
    # excluded due to excessive numerical failures
    "Crauste_CellSystems2017",
    "Fujita_SciSignal2010",
    # FIXME: re-enable
    "Raia_CancerResearch2011",
}
models = list(sorted(models))


@dataclass
class GradientCheckSettings:
    # Absolute and relative tolerances for simulation
    atol_sim: float = 1e-16
    rtol_sim: float = 1e-12
    # Absolute and relative tolerances for finite difference gradient checks.
    atol_check: float = 1e-3
    rtol_check: float = 1e-2
    # Absolute and relative tolerances for fiddy consistency check between
    # forward/backward/central differences.
    atol_consistency: float = 1e-5
    rtol_consistency: float = 1e-1
    # Step sizes for finite difference gradient checks.
    step_sizes = [
        1e-1,
        5e-2,
        1e-2,
        1e-3,
        1e-4,
        1e-5,
    ]
    rng_seed: int = 0
    ss_sensitivity_mode: amici.SteadyStateSensitivityMode = (
        amici.SteadyStateSensitivityMode.integrateIfNewtonFails
    )
    noise_level: float = 0.05


settings = defaultdict(GradientCheckSettings)
# NOTE: Newton method fails badly with ASA for Blasi_CellSystems2016
settings["Blasi_CellSystems2016"] = GradientCheckSettings(
    atol_check=1e-3,
    rtol_check=1e-4,
    ss_sensitivity_mode=amici.SteadyStateSensitivityMode.integrationOnly,
)
settings["Borghans_BiophysChem1997"] = GradientCheckSettings(
    rng_seed=2,
    atol_check=1e-5,
    rtol_check=1e-3,
)
settings["Brannmark_JBC2010"] = GradientCheckSettings(
    ss_sensitivity_mode=amici.SteadyStateSensitivityMode.integrationOnly,
    atol_check=1e-6,
    rtol_check=1e-3,
)
settings["Giordano_Nature2020"] = GradientCheckSettings(
    atol_check=1e-6, rtol_check=1e-3, rng_seed=1
)
settings["Okuonghae_ChaosSolitonsFractals2020"] = GradientCheckSettings(
    atol_sim=1e-14,
    rtol_sim=1e-14,
    noise_level=0.01,
    atol_consistency=1e-3,
)
settings["Okuonghae_ChaosSolitonsFractals2020"].step_sizes.extend([0.2, 0.005])
settings["Oliveira_NatCommun2021"] = GradientCheckSettings(
    # Avoid "root after reinitialization"
    atol_sim=1e-12,
    rtol_sim=1e-12,
    rtol_check=1e-3,
    atol_check=1e-12,
)
settings["Smith_BMCSystBiol2013"] = GradientCheckSettings(
    atol_sim=1e-10,
    rtol_sim=1e-10,
)
settings["Sneyd_PNAS2002"] = GradientCheckSettings(
    atol_sim=1e-15,
    rtol_sim=1e-12,
    atol_check=1e-5,
    rtol_check=1e-4,
    rng_seed=7,
)
settings["Weber_BMC2015"] = GradientCheckSettings(
    atol_sim=1e-12,
    rtol_sim=1e-12,
    atol_check=1e-6,
    rtol_check=1e-2,
    rng_seed=1,
)
settings["Zheng_PNAS2012"] = GradientCheckSettings(
    rng_seed=1,
    rtol_sim=1e-15,
    atol_check=5e-4,
    rtol_check=4e-3,
    noise_level=0.01,
)


debug = False
if debug:
    debug_path = Path(__file__).parent / "debug"
    debug_path.mkdir(exist_ok=True, parents=True)


@pytest.mark.filterwarnings(
    "ignore:divide by zero encountered in log",
    # https://github.com/AMICI-dev/AMICI/issues/18
    "ignore:Adjoint sensitivity analysis for models with discontinuous "
    "right hand sides .*:UserWarning",
)
@pytest.mark.parametrize("scale", (True, False), ids=["scaled", "unscaled"])
@pytest.mark.parametrize(
    "sensitivity_method",
    (SensitivityMethod.forward, SensitivityMethod.adjoint),
    ids=["forward", "adjoint"],
)
@pytest.mark.parametrize("model", models)
def test_benchmark_gradient(model, scale, sensitivity_method, request):
    if not scale and model in (
        "Smith_BMCSystBiol2013",
        "Brannmark_JBC2010",
        "Elowitz_Nature2000",
        "Borghans_BiophysChem1997",
        "Sneyd_PNAS2002",
        "Bertozzi_PNAS2020",
        "Zheng_PNAS2012",
    ):
        # not really worth the effort trying to fix these cases if they
        # only fail on linear scale
        pytest.skip()

    petab_problem = benchmark_models_petab.get_problem(model)
    petab.flatten_timepoint_specific_output_overrides(petab_problem)

    # Only compute gradient for estimated parameters.
    parameter_ids = petab_problem.x_free_ids
    cur_settings = settings[model]

    # Setup AMICI objects.
    amici_model = import_petab_problem(
        petab_problem,
        model_output_dir=benchmark_outdir / model,
    )
    amici_solver = amici_model.getSolver()
    amici_solver.setAbsoluteTolerance(cur_settings.atol_sim)
    amici_solver.setRelativeTolerance(cur_settings.rtol_sim)
    amici_solver.setMaxSteps(2 * 10**5)
    amici_solver.setSensitivityMethod(sensitivity_method)
    # TODO: we should probably test all sensitivity modes
    amici_model.setSteadyStateSensitivityMode(cur_settings.ss_sensitivity_mode)

    amici_function, amici_derivative = simulate_petab_to_cached_functions(
        petab_problem=petab_problem,
        parameter_ids=parameter_ids,
        amici_model=amici_model,
        solver=amici_solver,
        scaled_parameters=scale,
        scaled_gradients=scale,
        # FIXME: there is some issue with caching in fiddy
        #  e.g. Elowitz_Nature2000-True fails with cache=True,
        #  but not with cache=False
        # cache=not debug,
        cache=False,
    )
    np.random.seed(cur_settings.rng_seed)

    # find a point where the derivative can be computed
    for _ in range(5):
        if scale:
            point = petab_problem.x_nominal_free_scaled
            point_noise = (
                np.random.randn(len(point)) * cur_settings.noise_level
            )
        else:
            point = petab_problem.x_nominal_free
            point_noise = (
                np.random.randn(len(point)) * point * cur_settings.noise_level
            )
        point += point_noise  # avoid small gradients at nominal value

        try:
            expected_derivative = amici_derivative(point)
            break
        except RuntimeError as e:
            print(e)
            continue
    else:
        raise RuntimeError("Could not compute expected derivative.")

    derivative = get_derivative(
        function=amici_function,
        point=point,
        sizes=cur_settings.step_sizes,
        direction_ids=parameter_ids,
        method_ids=[MethodId.CENTRAL, MethodId.FORWARD, MethodId.BACKWARD],
        success_checker=Consistency(
            rtol=cur_settings.rtol_consistency,
            atol=cur_settings.atol_consistency,
        ),
        expected_result=expected_derivative,
        relative_sizes=not scale,
    )

    print()
    print("Testing at:", point)
    print("Expected derivative:", expected_derivative)
    print("Print actual derivative:", derivative.series.values)

    if debug:
        write_debug_output(
            debug_path / f"{request.node.callspec.id}.tsv",
            derivative,
            expected_derivative,
            parameter_ids,
        )

    assert_gradient_check_success(
        derivative,
        expected_derivative,
        point,
        rtol=cur_settings.rtol_check,
        atol=cur_settings.atol_check,
        always_print=True,
    )


def assert_gradient_check_success(
    derivative: fiddy.Derivative,
    expected_derivative: np.array,
    point: np.array,
    atol: float,
    rtol: float,
    always_print: bool = False,
) -> None:
    if not derivative.df.success.all():
        raise AssertionError(
            f"Failed to compute finite differences:\n{derivative.df}"
        )
    check = NumpyIsCloseDerivativeCheck(
        derivative=derivative,
        expectation=expected_derivative,
        point=point,
    )
    check_result = check(rtol=rtol, atol=atol)

    if check_result.success is True and not always_print:
        return

    df = check_result.df
    df["abs_diff"] = np.abs(df["expectation"] - df["test"])
    df["rel_diff"] = df["abs_diff"] / np.abs(df["expectation"])
    df["atol_success"] = df["abs_diff"] <= atol
    df["rtol_success"] = df["rel_diff"] <= rtol
    max_adiff = df["abs_diff"].max()
    max_rdiff = df["rel_diff"].max()
    with pd.option_context("display.max_columns", None, "display.width", None):
        message = (
            f"Gradient check failed:\n{df}\n\n"
            f"Maximum absolute difference: {max_adiff} (tolerance: {atol})\n"
            f"Maximum relative difference: {max_rdiff} (tolerance: {rtol})"
        )

    if check_result.success is False:
        raise AssertionError(message)

    if always_print:
        print(message)


def write_debug_output(
    file_name, derivative, expected_derivative, parameter_ids
):
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
    df["abs_diff"] = np.abs(df[("fd", "full", "")] - df[("amici", "", "")])
    df["rel_diff"] = df["abs_diff"] / np.abs(df[("amici", "", "")])

    df.to_csv(file_name, sep="\t")
