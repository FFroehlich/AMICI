from yaml import safe_load
import pytest

from pathlib import Path
import petab.v1 as petab
from amici.petab import import_petab_problem
from amici.jax import (
    JAXProblem,
    generate_equinox,
    run_simulations,
    petab_simulate,
)
import amici
import diffrax
import pandas as pd
import jax.numpy as jnp
import jax.random as jr
import jax
import numpy as np
import equinox as eqx
import os
import h5py
from contextlib import contextmanager

from petab_sciml import PetabScimlStandard


@contextmanager
def change_directory(destination):
    # Save the current working directory
    original_directory = os.getcwd()
    try:
        # Change to the new directory
        os.chdir(destination)
        yield
    finally:
        # Change back to the original directory
        os.chdir(original_directory)


jax.config.update("jax_enable_x64", True)


# pip install git+https://github.com/sebapersson/petab_sciml@add_standard#egg=petab_sciml\&subdirectory=src/python

cases_dir = Path(__file__).parent / "testsuite" / "test_cases"


def _reshape_flat_array(array_flat):
    array_flat["ix"] = array_flat["ix"].astype(str)
    ix_cols = [
        f"ix_{i}" for i in range(len(array_flat["ix"].values[0].split(";")))
    ]
    if len(ix_cols) == 1:
        array_flat[ix_cols[0]] = array_flat["ix"].apply(int)
    else:
        array_flat[ix_cols] = pd.DataFrame(
            array_flat["ix"].str.split(";").apply(np.array).to_list(),
            index=array_flat.index,
        ).astype(int)
    array_flat.sort_values(by=ix_cols, inplace=True)
    array_shape = tuple(array_flat[ix_cols].max().astype(int) + 1)
    array = np.array(array_flat["value"].values).reshape(array_shape)
    return array


@pytest.mark.parametrize(
    "test", sorted([d.stem for d in cases_dir.glob("net_[0-9]*")])
)
def test_net(test):
    test_dir = cases_dir / test
    with open(test_dir / "solutions.yaml") as f:
        solutions = safe_load(f)

    if test.endswith("_alt"):
        net_file = cases_dir / test.replace("_alt", "") / solutions["net_file"]
    else:
        net_file = test_dir / solutions["net_file"]
    ml_models = PetabScimlStandard.load_data(net_file)

    nets = {}
    outdir = Path(__file__).parent / "models" / test
    for ml_model in ml_models.models:
        module_dir = outdir / f"{ml_model.mlmodel_id}.py"
        if test in (
            "net_002",
            "net_009",
            "net_018",
            "net_019",
            "net_020",
            "net_021",
            "net_022",
            "net_042",
            "net_043",
            "net_044",
            "net_045",
            "net_046",
            "net_047",
            "net_048",
        ):
            with pytest.raises(NotImplementedError):
                generate_equinox(ml_model, module_dir)
            return
        generate_equinox(ml_model, module_dir)
        nets[ml_model.mlmodel_id] = amici._module_from_path(
            ml_model.mlmodel_id, module_dir
        ).net

    for input_file, par_file, output_file in zip(
        solutions["net_input"],
        solutions.get("net_ps", solutions["net_input"]),
        solutions["net_output"],
    ):
        input_flat = pd.read_csv(test_dir / input_file, sep="\t")
        input = _reshape_flat_array(input_flat)

        output_flat = pd.read_csv(test_dir / output_file, sep="\t")
        output = _reshape_flat_array(output_flat)

        if "net_ps" in solutions:
            par = pd.read_csv(test_dir / par_file, sep="\t")
            for ml_model in ml_models.models:
                net = nets[ml_model.mlmodel_id](jr.PRNGKey(0))
                for layer in net.layers.keys():
                    layer_prefix = f"net_{layer}"
                    if (
                        isinstance(net.layers[layer], eqx.Module)
                        and hasattr(net.layers[layer], "weight")
                        and net.layers[layer].weight is not None
                    ):
                        prefix = layer_prefix + "_weight"
                        df = par[
                            par[petab.PARAMETER_ID].str.startswith(prefix)
                        ]
                        df["ix"] = (
                            df[petab.PARAMETER_ID]
                            .str.split("_")
                            .str[3:]
                            .apply(lambda x: ";".join(x))
                        )
                        w = _reshape_flat_array(df)
                        if isinstance(net.layers[layer], eqx.nn.ConvTranspose):
                            # see FAQ in https://docs.kidger.site/equinox/api/nn/conv/#equinox.nn.ConvTranspose
                            w = np.flip(
                                w, axis=tuple(range(2, w.ndim))
                            ).swapaxes(0, 1)
                        assert w.shape == net.layers[layer].weight.shape
                        net = eqx.tree_at(
                            lambda x: x.layers[layer].weight,
                            net,
                            jnp.array(w),
                        )
                    if (
                        isinstance(net.layers[layer], eqx.Module)
                        and hasattr(net.layers[layer], "bias")
                        and net.layers[layer].bias is not None
                    ):
                        prefix = layer_prefix + "_bias"
                        df = par[
                            par[petab.PARAMETER_ID].str.startswith(prefix)
                        ]
                        df["ix"] = (
                            df[petab.PARAMETER_ID]
                            .str.split("_")
                            .str[3:]
                            .apply(lambda x: ";".join(x))
                        )
                        b = _reshape_flat_array(df)
                        if isinstance(
                            net.layers[layer],
                            eqx.nn.Conv | eqx.nn.ConvTranspose,
                        ):
                            b = np.expand_dims(
                                b,
                                tuple(
                                    range(
                                        1,
                                        net.layers[layer].num_spatial_dims + 1,
                                    )
                                ),
                            )
                        assert b.shape == net.layers[layer].bias.shape
                        net = eqx.tree_at(
                            lambda x: x.layers[layer].bias,
                            net,
                            jnp.array(b),
                        )
                net = eqx.nn.inference_mode(net)

                if test == "net_004_alt":
                    return  # skipping, no support for non-cross-correlation in equinox

                np.testing.assert_allclose(
                    net.forward(input),
                    output,
                    atol=1e-3,
                    rtol=1e-3,
                )


@pytest.mark.parametrize(
    "test", sorted([d.stem for d in cases_dir.glob("[0-9]*")])
)
def test_ude(test):
    test_dir = cases_dir / test
    with open(test_dir / "petab" / "problem_ude.yaml") as f:
        petab_yaml = safe_load(f)
    with open(test_dir / "solutions.yaml") as f:
        solutions = safe_load(f)

    with change_directory(test_dir / "petab"):
        from petab.v2 import Problem

        petab_yaml["format_version"] = "2.0.0"
        for problem in petab_yaml["problems"]:
            problem["model_files"] = {
                problem["model_files"]["location"].split(".")[0]: problem[
                    "model_files"
                ]
            }
            for mapping_file in problem["mapping_files"]:
                df = pd.read_csv(
                    mapping_file,
                    sep="\t",
                )
                if df[petab.PETAB_ENTITY_ID].str.startswith("net").any():
                    df.rename(
                        columns={
                            petab.PETAB_ENTITY_ID: petab.MODEL_ENTITY_ID,
                            petab.MODEL_ENTITY_ID: petab.PETAB_ENTITY_ID,
                        }
                    ).to_csv(mapping_file, sep="\t", index=False)

        petab_problem = Problem.from_yaml(petab_yaml)
        jax_model = import_petab_problem(
            petab_problem,
            model_output_dir=Path(__file__).parent / "models" / test,
            compile_=True,
            jax=True,
        )
        jax_problem = JAXProblem(jax_model, petab_problem)
        for net, net_config in petab_problem.extensions_config[
            "petab_sciml"
        ].items():
            pars = h5py.File(
                net_config["parameters"].replace(".h5", ".hf5"), "r"
            )
            for layer_name, layer in jax_problem.model.nns[net].layers.items():
                for attribute in dir(layer):
                    if not isinstance(
                        getattr(layer, attribute), jax.numpy.ndarray
                    ):
                        continue
                    value = jnp.array(pars[layer_name][attribute])

                    if (
                        isinstance(layer, eqx.nn.ConvTranspose)
                        and attribute == "weight"
                    ):
                        # see FAQ in https://docs.kidger.site/equinox/api/nn/conv/#equinox.nn.ConvTranspose
                        value = jnp.flip(
                            value, axis=tuple(range(2, value.ndim))
                        ).swapaxes(0, 1)
                    jax_problem = eqx.tree_at(
                        lambda x: getattr(
                            x.model.nns[net].layers[layer_name], attribute
                        ),
                        jax_problem,
                        value,
                    )

    # llh
    if test in (
        "004",
        "016",
    ):
        with pytest.raises(NotImplementedError):
            run_simulations(jax_problem)
        return
    llh, r = run_simulations(jax_problem)
    np.testing.assert_allclose(
        llh,
        solutions["llh"],
        atol=solutions["tol_llh"],
        rtol=solutions["tol_llh"],
    )
    simulations = pd.concat(
        [
            pd.read_csv(test_dir / simulation, sep="\t")
            for simulation in solutions["simulation_files"]
        ]
    )

    # simulations
    sort_by = [petab.OBSERVABLE_ID, petab.TIME, petab.SIMULATION_CONDITION_ID]
    actual = petab_simulate(jax_problem).sort_values(by=sort_by)
    expected = simulations.sort_values(by=sort_by)
    np.testing.assert_allclose(
        actual[petab.SIMULATION].values,
        expected[petab.SIMULATION].values,
        atol=solutions["tol_simulations"],
        rtol=solutions["tol_simulations"],
    )

    # gradient
    sllh, _ = eqx.filter_grad(run_simulations, has_aux=True)(
        jax_problem,
        solver=diffrax.Kvaerno5(),
        controller=diffrax.PIDController(atol=1e-14, rtol=1e-14),
        max_steps=2**16,
    )
    for component, file in solutions["grad_llh_files"].items():
        actual_dict = {}
        if component == "mech":
            expected = pd.read_csv(test_dir / file, sep="\t").set_index(
                petab.PARAMETER_ID
            )
            for ip in expected.index:
                if ip in jax_problem.parameter_ids:
                    actual_dict[ip] = sllh.parameters[
                        jax_problem.parameter_ids.index(ip)
                    ].item()
            actual = pd.Series(actual_dict).loc[expected.index].values
            np.testing.assert_allclose(
                actual,
                expected["value"].values,
                atol=solutions["tol_grad_llh"],
                rtol=solutions["tol_grad_llh"],
            )
        else:
            expected = h5py.File(test_dir / file, "r")
            for layer_name, layer in jax_problem.model.nns[
                component
            ].layers.items():
                for attribute in dir(layer):
                    if not isinstance(
                        getattr(layer, attribute), jax.numpy.ndarray
                    ):
                        continue
                    actual = getattr(
                        sllh.model.nns[component].layers[layer_name], attribute
                    )
                    if (
                        isinstance(layer, eqx.nn.ConvTranspose)
                        and attribute == "weight"
                    ):
                        actual = np.flip(
                            actual.swapaxes(0, 1),
                            axis=tuple(range(2, actual.ndim)),
                        )
                    np.testing.assert_allclose(
                        actual,
                        expected[layer_name][attribute][:],
                        atol=solutions["tol_grad_llh"],
                        rtol=solutions["tol_grad_llh"],
                    )
