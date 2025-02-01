"""Tests based on the PEtab benchmark problems.

Tests simulate_petab, correctness of the log-likelihood computation at nominal
parameters, correctness of the gradient computation, and simulation times
for a subset of the benchmark problems.
"""

import copy
from functools import partial
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
from dataclasses import dataclass, field
from amici import SensitivityMethod
from petab.v1.lint import measurement_table_has_timepoint_specific_mappings
from fiddy import MethodId, get_derivative
from fiddy.derivative_check import NumpyIsCloseDerivativeCheck
from fiddy.extensions.amici import simulate_petab_to_cached_functions
from fiddy.success import Consistency
import contextlib
import logging
import yaml
from amici.logging import get_logger
from amici.petab.simulations import (
    LLH,
    SLLH,
    RDATAS,
    rdatas_to_measurement_df,
    simulate_petab,
)

from petab.v1.visualize import plot_problem


# Enable various debug output
debug = False

logger = get_logger(
    f"amici.{__name__}", logging.DEBUG if debug else logging.INFO
)

script_dir = Path(__file__).parent.absolute()
repo_root = script_dir.parent.parent
benchmark_outdir = repo_root / "test_bmc"
debug_path = script_dir / "debug"
if debug:
    debug_path.mkdir(exist_ok=True, parents=True)

# reference values for simulation times and log-likelihoods
references_yaml = script_dir / "benchmark_models.yaml"
with open(references_yaml) as f:
    reference_values = yaml.full_load(f)

# problem IDs for which to check the gradient
# TODO: extend
problems_for_gradient_check = set(benchmark_models_petab.MODELS) - {
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
}
problems_for_gradient_check = list(sorted(problems_for_gradient_check))


# Problems for checking the log-likelihood computation at nominal parameters
#
# not merged:
# Becker_Science2010 (multiple models)             https://github.com/Benchmarking-Initiative/Benchmark-Models-PEtab/tree/model_Becker_Science2010
# Hass_PONE2017 (???)                              https://github.com/Benchmarking-Initiative/Benchmark-Models-PEtab/tree/model_Hass_PONE2017
# Korkut_eLIFE2015 (???)                           https://github.com/Benchmarking-Initiative/Benchmark-Models-PEtab/tree/model_Korkut_eLIFE2015
# Casaletto_PNAS2019 (yaml missing)                https://github.com/Benchmarking-Initiative/Benchmark-Models-PEtab/tree/model_Casaletto_PNAS2019
# Merkle_PCB2016 (model missing)                   https://github.com/Benchmarking-Initiative/Benchmark-Models-PEtab/tree/model_Merkle_PCB2016
# Parmar_PCB2019 (SBML extensions)                 https://github.com/Benchmarking-Initiative/Benchmark-Models-PEtab/tree/model_Parmar_PCB2019
# Swameye_PNAS2003 (splines)                       https://github.com/Benchmarking-Initiative/Benchmark-Models-PEtab/tree/model_Swameye_PNAS2003
# Sobotta_Frontiers2017 (???)                      https://github.com/Benchmarking-Initiative/Benchmark-Models-PEtab/tree/model_Sobotta_Frontiers2017
# Raia_CancerResearch2011 (state dependent sigmas) https://github.com/Benchmarking-Initiative/Benchmark-Models-PEtab/tree/model_Raia_CancerResearch2011
#
# no reference value:
# Alkan_SciSignal2018 (d2d: Alkan_DDRP_SciSignal2018)
# Bertozzi_PNAS2020 (gh: vanako, doi: https://doi.org/10.1073/pnas.2006520117, code/data: https://github.com/gomohler/pnas2020 (R))
# Blasi_CellSystems2016 (gh: Leonard Schmiester, doi: https://doi.org/10.1016/j.cels.2016.01.002, code/data: not available)
# Giordano_Nature2020 (gh: Paul Jonas Jost, doi: https://doi.org/10.1038/s41591-020-0883-7, code/data: http://users.dimi.uniud.it/~giulia.giordano/docs/papers/SIDARTHEcode.zip (MATLAB))
# Laske_PLOSComputBiol2019 (gh: Clemens Peiter, doi: https://doi.org/10.1128/JVI.00080-12 (?), code/data: ???)
# Okuonghae_ChaosSolitonsFractals2020 (gh: Paul Jonas Jost, doi: https://doi.org/10.1016/j.chaos.2020.110032, code/data: ???)
# Oliveira_NatCommun2021 (gh: lorenapohl, doi: https://doi.org/10.1038/s41467-020-19798-3, code: https://github.com/cidacslab/Mathematical-and-Statistical-Modeling-of-COVID19-in-Brazil (python) data: https://infovis.sei.ba.gov.br/covid19/ )
# Perelson_Science1996 (gh: Philipp Staedter, doi: https://doi.org/10.1126/science.271.5255.1582, code/data: ???)
# Rahman_MBS2016 (gh: Yannik Schaelte, doi: https://doi.org/10.1016/j.mbs.2016.07.009, code: not available, data: table in paper ...)
# Raimundez_PCB2020 (gh: Elba Raimundez, doi: https://doi.org/10.1371/journal.pcbi.1007147, code/data: https://zenodo.org/record/2908234#.Y5hUUS8w3yw (d2d))
# SalazarCavazos_MBoC2020 (gh: Dilan Pathirana, doi: https://doi.org/10.1091/mbc.E19-09-0548, code/data: supplement (BNGL))
# Zhao_QuantBiol2020 (gh: Iva Ewert, doi: https://doi.org/10.1007/s40484-020-0199-0, code: not available, data: table in supp)
#
# covered by performance test:
# Froehlich_CellSystems2018
#
# Unknown reasons:
# Chen_MSB2009
#
# Confirmed to be working:
# TODO: extend
problems_for_llh_check = [
    "Bachmann_MSB2011",
    "Beer_MolBioSystems2014",
    "Boehm_JProteomeRes2014",
    "Borghans_BiophysChem1997",
    "Brannmark_JBC2010",
    "Bruno_JExpBot2016",
    "Crauste_CellSystems2017",
    "Elowitz_Nature2000",
    "Fiedler_BMCSystBiol2016",
    "Fujita_SciSignal2010",
    # Excluded until https://github.com/Benchmarking-Initiative/Benchmark-Models-PEtab/pull/253
    #  is sorted out
    # "Isensee_JCB2018",
    "Lucarelli_CellSystems2018",
    "Schwen_PONE2014",
    "Smith_BMCSystBiol2013",
    "Sneyd_PNAS2002",
    "Weber_BMC2015",
    "Zheng_PNAS2012",
]

# all PEtab problems we need to import
problems = list(
    sorted(set(problems_for_gradient_check + problems_for_llh_check))
)


@dataclass
class GradientCheckSettings:
    """Problem-specific settings for gradient checks."""

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
    step_sizes: list[float] = field(
        default_factory=lambda: [
            2e-1,
            1e-1,
            5e-2,
            1e-2,
            5e-1,
            1e-3,
            1e-4,
            1e-5,
        ]
    )
    rng_seed: int = 0
    ss_sensitivity_mode: amici.SteadyStateSensitivityMode = (
        amici.SteadyStateSensitivityMode.integrateIfNewtonFails
    )
    noise_level: float = 0.05


settings = defaultdict(GradientCheckSettings)
# NOTE: Newton method fails badly with ASA for Blasi_CellSystems2016
settings["Blasi_CellSystems2016"] = GradientCheckSettings(
    atol_check=1e-12,
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
)
settings["Fujita_SciSignal2010"] = GradientCheckSettings(
    atol_check=1e-7,
    rtol_check=5e-4,
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
settings["Oliveira_NatCommun2021"] = GradientCheckSettings(
    # Avoid "root after reinitialization"
    atol_sim=1e-12,
    rtol_sim=1e-12,
)
settings["Raia_CancerResearch2011"] = GradientCheckSettings(
    atol_check=1e-10,
    rtol_check=1e-3,
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


@pytest.fixture(scope="session", params=problems, ids=problems)
def benchmark_problem(request):
    """Fixture providing model and PEtab problem for a problem from
    the benchmark problem collection."""
    problem_id = request.param
    petab_problem = benchmark_models_petab.get_problem(problem_id)
    flat_petab_problem = copy.deepcopy(petab_problem)
    if measurement_table_has_timepoint_specific_mappings(
        petab_problem.measurement_df,
    ):
        petab.flatten_timepoint_specific_output_overrides(flat_petab_problem)

    # Setup AMICI objects.
    amici_model = import_petab_problem(
        flat_petab_problem,
        model_output_dir=benchmark_outdir / problem_id,
    )
    return problem_id, flat_petab_problem, petab_problem, amici_model


@pytest.mark.filterwarnings(
    "ignore:The following problem parameters were not used *",
    "ignore: The environment variable *",
    "ignore:Adjoint sensitivity analysis for models with discontinuous ",
)
def test_jax_llh(benchmark_problem):
    import jax
    import equinox as eqx
    import jax.numpy as jnp
    from amici.jax.petab import run_simulations, JAXProblem

    jax.config.update("jax_enable_x64", True)
    from beartype import beartype

    problem_id, flat_petab_problem, petab_problem, amici_model = (
        benchmark_problem
    )

    amici_solver = amici_model.getSolver()
    cur_settings = settings[problem_id]
    amici_solver.setAbsoluteTolerance(1e-8)
    amici_solver.setRelativeTolerance(1e-8)
    amici_solver.setMaxSteps(10_000)

    simulate_amici = partial(
        simulate_petab,
        petab_problem=flat_petab_problem,
        amici_model=amici_model,
        solver=amici_solver,
        scaled_parameters=True,
        scaled_gradients=True,
        log_level=logging.DEBUG,
    )

    np.random.seed(cur_settings.rng_seed)

    problem_parameters = None
    if problem_id in problems_for_gradient_check:
        point = flat_petab_problem.x_nominal_free_scaled
        for _ in range(20):
            amici_solver.setSensitivityMethod(amici.SensitivityMethod.adjoint)
            amici_solver.setSensitivityOrder(amici.SensitivityOrder.first)
            amici_model.setSteadyStateSensitivityMode(
                cur_settings.ss_sensitivity_mode
            )
            point_noise = (
                np.random.randn(len(point)) * cur_settings.noise_level
            )
            point += point_noise  # avoid small gradients at nominal value

            problem_parameters = dict(
                zip(flat_petab_problem.x_free_ids, point)
            )

            r_amici = simulate_amici(
                problem_parameters=problem_parameters,
            )
            if np.isfinite(r_amici[LLH]):
                break
        else:
            raise RuntimeError("Could not compute expected derivative.")
    else:
        r_amici = simulate_amici()
    llh_amici = r_amici[LLH]

    if problem_id == "Zhao_QuantBiol2020":
        # temp workaround for https://github.com/Benchmarking-Initiative/Benchmark-Models-PEtab/pull/261
        petab_problem.observable_df.loc[
            "observable_unquarantined_infected", petab.NOISE_FORMULA
        ] = "noiseParameter1_observable_unquarantined_infected"

    jax_model = import_petab_problem(
        petab_problem,
        model_output_dir=benchmark_outdir / (problem_id + "_jax"),
        jax=True,
    )
    jax_problem = JAXProblem(jax_model, petab_problem)
    if problem_parameters:
        jax_problem = eqx.tree_at(
            lambda x: x.parameters,
            jax_problem,
            jnp.array(
                [problem_parameters[pid] for pid in jax_problem.parameter_ids]
            ),
        )

    if problem_id in problems_for_gradient_check:
        beartype(run_simulations)(jax_problem)
        (llh_jax, _), sllh_jax = eqx.filter_value_and_grad(
            run_simulations, has_aux=True
        )(jax_problem)
    else:
        llh_jax, _ = beartype(run_simulations)(jax_problem)

    np.testing.assert_allclose(
        llh_jax,
        llh_amici,
        rtol=1e-3,
        atol=1e-3,
        err_msg=f"LLH mismatch for {problem_id}",
    )

    if problem_id in problems_for_gradient_check:
        sllh_amici = r_amici[SLLH]
        np.testing.assert_allclose(
            sllh_jax.parameters,
            np.array([sllh_amici[pid] for pid in jax_problem.parameter_ids]),
            rtol=1e-2,
            atol=1e-2,
            err_msg=f"SLLH mismatch for {problem_id}, {dict(zip(jax_problem.parameter_ids, sllh_jax.parameters))}",
        )


@pytest.mark.filterwarnings(
    "ignore:divide by zero encountered in log",
    # https://github.com/AMICI-dev/AMICI/issues/18
    "ignore:Adjoint sensitivity analysis for models with discontinuous "
    "right hand sides .*:UserWarning",
)
def test_nominal_parameters_llh(benchmark_problem):
    """Test the log-likelihood computation at nominal parameters.

    Also check that the simulation time is within the reference range.
    """
    problem_id, petab_problem, _, amici_model = benchmark_problem
    if problem_id not in problems_for_llh_check:
        pytest.skip("Excluded from log-likelihood check.")

    amici_solver = amici_model.getSolver()
    amici_solver.setAbsoluteTolerance(1e-8)
    amici_solver.setRelativeTolerance(1e-8)
    amici_solver.setMaxSteps(10_000)
    if problem_id in ("Brannmark_JBC2010", "Isensee_JCB2018"):
        amici_model.setSteadyStateSensitivityMode(
            amici.SteadyStateSensitivityMode.integrationOnly
        )

    times = dict()

    for label, sensi_mode in {
        "t_sim": amici.SensitivityMethod.none,
        "t_fwd": amici.SensitivityMethod.forward,
        "t_adj": amici.SensitivityMethod.adjoint,
    }.items():
        amici_solver.setSensitivityMethod(sensi_mode)
        if sensi_mode == amici.SensitivityMethod.none:
            amici_solver.setSensitivityOrder(amici.SensitivityOrder.none)
        else:
            amici_solver.setSensitivityOrder(amici.SensitivityOrder.first)

        res_repeats = [
            simulate_petab(
                petab_problem=petab_problem,
                amici_model=amici_model,
                solver=amici_solver,
                log_level=logging.DEBUG,
            )
            for _ in range(3)  # repeat to get more stable timings
        ]
        res = res_repeats[0]

        times[label] = np.min(
            [
                sum(r.cpu_time + r.cpu_timeB for r in res[RDATAS]) / 1000
                # only forwards/backwards simulation
                for res in res_repeats
            ]
        )

        if sensi_mode == amici.SensitivityMethod.none:
            rdatas = res[RDATAS]
            llh = res[LLH]
    # TODO: check that all llh match, check that all sllh match
    times["np"] = sum(petab_problem.parameter_df[petab.ESTIMATE])

    pd.Series(times).to_csv(script_dir / f"{problem_id}_benchmark.csv")
    for rdata in rdatas:
        assert (
            rdata.status == amici.AMICI_SUCCESS
        ), f"Simulation failed for {rdata.id}"

    if debug:
        # create simulation PEtab table
        sim_df = rdatas_to_measurement_df(
            rdatas=rdatas,
            model=amici_model,
            measurement_df=petab_problem.measurement_df,
        )
        sim_df.rename(
            columns={petab.MEASUREMENT: petab.SIMULATION}, inplace=True
        )
        sim_df.to_csv(
            benchmark_outdir / problem_id / f"{problem_id}_simulation.tsv",
            index=False,
            sep="\t",
        )

        # visualize fit, save to file
        with contextlib.suppress(NotImplementedError):
            axs = plot_problem(
                petab_problem=petab_problem, simulations_df=sim_df
            )
            for plot_id, ax in axs.items():
                fig_path = (
                    benchmark_outdir
                    / problem_id
                    / f"{problem_id}_{plot_id}_vis.png"
                )
                logger.info(f"Saving figure to {fig_path}")
                ax.get_figure().savefig(fig_path, dpi=150)

    try:
        ref_llh = reference_values[problem_id]["llh"]
        rdiff = np.abs((llh - ref_llh) / ref_llh)
        rtol = 1e-3
        adiff = np.abs(llh - ref_llh)
        atol = 1e-3
        tolstr = (
            f" Absolute difference is {adiff:.2e} "
            f"(tol {atol:.2e}) and relative difference is "
            f"{rdiff:.2e} (tol {rtol:.2e})."
        )

        if np.isclose(llh, ref_llh, rtol=rtol, atol=atol):
            logger.info(
                f"Computed llh {llh:.4e} matches reference {ref_llh:.4e}."
                + tolstr
            )
        else:
            pytest.fail(
                f"Computed llh {llh:.4e} does not match reference "
                f"{ref_llh:.4e}." + tolstr
            )
    except KeyError:
        logger.error(
            "No reference likelihood found for "
            f"{problem_id} in {references_yaml}"
        )

    for label, key in {
        "simulation": "t_sim",
        "adjoint sensitivity": "t_adj",
        "forward sensitivity": "t_fwd",
    }.items():
        try:
            ref = reference_values[problem_id][key]
            if times[key] > ref:
                pytest.fail(
                    f"Computation time for {label} ({times[key]:.2e}) "
                    f"exceeds reference ({ref:.2e})."
                )
            else:
                logger.info(
                    f"Computation time for {label} ({times[key]:.2e}) "
                    f"within reference ({ref:.2e})."
                )
        except KeyError:
            logger.error(
                f"No reference time for {label} found for "
                f"{problem_id} in {references_yaml}"
            )


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
def test_benchmark_gradient(
    benchmark_problem, scale, sensitivity_method, request
):
    problem_id, petab_problem, _, amici_model = benchmark_problem
    if problem_id not in problems_for_gradient_check:
        pytest.skip("Excluded from gradient check.")

    if not scale and problem_id in (
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
        pytest.skip("scale=False disabled for this problem")

    if (
        problem_id
        in (
            # events with parameter-dependent triggers
            #  https://github.com/AMICI-dev/AMICI/issues/18
            "Oliveira_NatCommun2021",
        )
        and sensitivity_method == SensitivityMethod.adjoint
    ):
        pytest.skip("Unsupported ASA+events")

    petab_problem = benchmark_models_petab.get_problem(problem_id)
    if measurement_table_has_timepoint_specific_mappings(
        petab_problem.measurement_df,
    ):
        petab.flatten_timepoint_specific_output_overrides(petab_problem)

    # Only compute gradient for estimated parameters.
    parameter_ids = petab_problem.x_free_ids
    cur_settings = settings[problem_id]

    # Setup AMICI objects.
    amici_model = import_petab_problem(
        petab_problem,
        model_output_dir=benchmark_outdir / problem_id,
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

    success_fail = "succeeded" if check_result.success else "failed"
    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        None,
        "display.max_rows",
        None,
    ):
        message = (
            f"Gradient check {success_fail}:\n{df}\n\n"
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
