#!/usr/bin/env python3

"""
Simulate a PEtab problem and compare results to reference values
"""
import argparse
import contextlib
import importlib
import logging
import os
import sys
from pathlib import Path

import amici
import numpy as np
import pandas as pd
import petab
import yaml
from amici.logging import get_logger
from amici.petab.simulations import (
    LLH,
    RDATAS,
    rdatas_to_measurement_df,
    simulate_petab,
)
from petab.visualize import plot_problem

logger = get_logger(f"amici.{__name__}", logging.WARNING)


def parse_cli_args():
    """Parse command line arguments

    Returns:
        Parsed CLI arguments from ``argparse``.
    """

    parser = argparse.ArgumentParser(
        description="Simulate PEtab-format model using AMICI."
    )

    # General options:
    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        action="store_true",
        help="More verbose output",
    )
    parser.add_argument(
        "-c",
        "--check",
        dest="check",
        action="store_true",
        help="Compare to reference value",
    )
    parser.add_argument(
        "-p",
        "--plot",
        dest="plot",
        action="store_true",
        help="Plot measurement and simulation results",
    )

    # PEtab problem
    parser.add_argument(
        "-y",
        "--yaml",
        dest="yaml_file_name",
        required=True,
        help="PEtab YAML problem filename",
    )

    # Corresponding AMICI model
    parser.add_argument(
        "-m",
        "--model-name",
        dest="model_name",
        help="Name of the AMICI module of the model to " "simulate.",
        required=True,
    )
    parser.add_argument(
        "-d",
        "--model-dir",
        dest="model_directory",
        help="Directory containing the AMICI module of the "
        "model to simulate. Required if model is not "
        "in python path.",
    )

    parser.add_argument(
        "-o",
        "--simulation-file",
        dest="simulation_file",
        help="File to write simulation result to, in PEtab"
        "measurement table format.",
    )

    return parser.parse_args()


def main():
    """Simulate the model specified on the command line"""
    script_dir = Path(__file__).parent.absolute()
    args = parse_cli_args()
    loglevel = logging.DEBUG if args.verbose else logging.INFO
    logger.setLevel(loglevel)

    logger.info(
        f"Simulating '{args.model_name}' "
        f"({args.model_directory}) using PEtab data from "
        f"{args.yaml_file_name}"
    )

    # load PEtab files
    problem = petab.Problem.from_yaml(args.yaml_file_name)
    petab.flatten_timepoint_specific_output_overrides(problem)

    # load model
    if args.model_directory:
        sys.path.insert(0, args.model_directory)
    model_module = importlib.import_module(args.model_name)
    amici_model = model_module.getModel()
    amici_solver = amici_model.getSolver()

    amici_solver.setAbsoluteTolerance(1e-8)
    amici_solver.setRelativeTolerance(1e-8)
    amici_solver.setMaxSteps(int(1e4))
    if args.model_name in ("Brannmark_JBC2010", "Isensee_JCB2018"):
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
                petab_problem=problem,
                amici_model=amici_model,
                solver=amici_solver,
                log_level=loglevel,
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

    times["np"] = sum(problem.parameter_df[petab.ESTIMATE])

    pd.Series(times).to_csv(script_dir / f"{args.model_name}_benchmark.csv")
    for rdata in rdatas:
        assert (
            rdata.status == amici.AMICI_SUCCESS
        ), f"Simulation failed for {rdata.id}"

    # create simulation PEtab table
    sim_df = rdatas_to_measurement_df(
        rdatas=rdatas, model=amici_model, measurement_df=problem.measurement_df
    )
    sim_df.rename(columns={petab.MEASUREMENT: petab.SIMULATION}, inplace=True)

    if args.simulation_file:
        sim_df.to_csv(args.simulation_file, index=False, sep="\t")

    if args.plot:
        with contextlib.suppress(NotImplementedError):
            # visualize fit
            axs = plot_problem(petab_problem=problem, simulations_df=sim_df)

            # save figure
            for plot_id, ax in axs.items():
                fig_path = os.path.join(
                    args.model_directory,
                    f"{args.model_name}_{plot_id}_vis.png",
                )
                logger.info(f"Saving figure to {fig_path}")
                ax.get_figure().savefig(fig_path, dpi=150)

    if args.check:
        references_yaml = script_dir / "benchmark_models.yaml"
        with open(references_yaml) as f:
            refs = yaml.full_load(f)

        try:
            ref_llh = refs[args.model_name]["llh"]

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
                logger.error(
                    f"Computed llh {llh:.4e} does not match reference "
                    f"{ref_llh:.4e}." + tolstr
                )
                sys.exit(1)
        except KeyError:
            logger.error(
                "No reference likelihood found for "
                f"{args.model_name} in {references_yaml}"
            )

        for label, key in {
            "simulation": "t_sim",
            "adjoint sensitivity": "t_adj",
            "forward sensitivity": "t_fwd",
        }.items():
            try:
                ref = refs[args.model_name][key]
                if times[key] > ref:
                    logger.error(
                        f"Computation time for {label} ({times[key]:.2e}) "
                        f"exceeds reference ({ref:.2e})."
                    )
                    sys.exit(1)
                else:
                    logger.info(
                        f"Computation time for {label} ({times[key]:.2e}) "
                        f"within reference ({ref:.2e})."
                    )
            except KeyError:
                logger.error(
                    f"No reference time for {label} found for "
                    f"{args.model_name} in {references_yaml}"
                )


if __name__ == "__main__":
    main()
