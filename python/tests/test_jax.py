import pytest
import amici

pytest.importorskip("jax")
import amici.jax

import jax.numpy as jnp
import numpy as np

from amici.pysb_import import pysb2amici
from numpy.testing import assert_allclose

pysb = pytest.importorskip("pysb")


def test_conversion():
    pysb.SelfExporter.cleanup()  # reset pysb
    pysb.SelfExporter.do_export = True

    model = pysb.Model("conversion")
    a = pysb.Monomer("A", sites=["s"], site_states={"s": ["a", "b"]})
    pysb.Initial(a(s="a"), pysb.Parameter("aa0", 1.2))
    pysb.Rule("conv", a(s="a") >> a(s="b"), pysb.Parameter("kcat", 0.05))
    pysb.Observable("ab", a(s="b"))

    outdir = model.name
    pysb2amici(model, outdir, verbose=True, observables=["ab"])

    model_module = amici.import_model_module(
        module_name=model.name, module_path=outdir
    )

    ts = tuple(np.linspace(0, 1, 10))
    p = jnp.stack((1.0, 0.1), axis=-1)
    k = tuple()
    _test_model(model_module, ts, p, k)


@pytest.mark.filterwarnings(
    "ignore:Model does not contain any initial conditions"
)
def test_dimerization():
    pysb.SelfExporter.cleanup()  # reset pysb
    pysb.SelfExporter.do_export = True

    model = pysb.Model("dimerization")
    a = pysb.Monomer("A", sites=["b"])
    b = pysb.Monomer("B", sites=["a"])

    pysb.Rule(
        "turnover_a",
        a(b=None) | None,
        pysb.Parameter("kdeg_a", 10),
        pysb.Parameter("ksyn_a", 0.1),
    )
    pysb.Rule(
        "turnover_b",
        b(a=None) | None,
        pysb.Parameter("kdeg_b", 0.1),
        pysb.Parameter("ksyn_b", 10),
    )
    pysb.Rule(
        "dimer",
        a(b=None) + b(a=None) | a(b=1) % b(a=1),
        pysb.Parameter("kon", 1.0),
        pysb.Parameter("koff", 0.1),
    )

    pysb.Observable("a_obs", a())
    pysb.Observable("b_obs", b())

    outdir = model.name
    pysb2amici(
        model,
        outdir,
        verbose=True,
        observables=["a_obs", "b_obs"],
        constant_parameters=["ksyn_a", "ksyn_b"],
    )

    model_module = amici.import_model_module(
        module_name=model.name, module_path=outdir
    )

    ts = tuple(np.linspace(0, 1, 10))
    p = jnp.stack((5, 0.5, 0.5, 0.5), axis=-1)
    k = (0.5, 5)
    _test_model(model_module, ts, p, k)


def _test_model(model_module, ts, p, k):
    amici_model = model_module.getModel()

    amici_model.setTimepoints(np.asarray(ts, dtype=np.float64))
    sol_amici_ref = amici.runAmiciSimulation(
        amici_model, amici_model.getSolver()
    )

    jax_model = model_module.get_jax_model()

    amici_model.setParameters(np.asarray(p, dtype=np.float64))
    amici_model.setFixedParameters(np.asarray(k, dtype=np.float64))
    edata = amici.ExpData(sol_amici_ref, 1.0, 1.0)
    edata.parameters = amici_model.getParameters()
    edata.fixedParameters = amici_model.getFixedParameters()
    edata.pscale = amici_model.getParameterScale()
    amici_solver = amici_model.getSolver()
    amici_solver.setSensitivityMethod(amici.SensitivityMethod.forward)
    amici_solver.setSensitivityOrder(amici.SensitivityOrder.first)
    rs_amici = amici.runAmiciSimulations(amici_model, amici_solver, [edata])

    check_fields_jax(rs_amici, jax_model, edata, ["x", "y", "llh"])

    check_fields_jax(
        rs_amici,
        jax_model,
        edata,
        ["x", "y", "llh", "sllh"],
        sensi_order=amici.SensitivityOrder.first,
    )

    check_fields_jax(
        rs_amici,
        jax_model,
        edata,
        ["x", "y", "llh", "sllh"],
        sensi_order=amici.SensitivityOrder.second,
    )


def check_fields_jax(
    rs_amici,
    jax_model,
    edata,
    fields,
    sensi_order=amici.SensitivityOrder.none,
):
    r_jax = dict()
    kwargs = {
        "ts": np.array(edata.getTimepoints()),
        "ts_dyn": np.array(edata.getTimepoints()),
        "p": np.array(edata.parameters),
        "k": np.array(edata.fixedParameters),
        "k_preeq": np.array([]),
        "my": np.array(edata.getObservedData()).reshape(
            np.array(edata.getTimepoints()).shape[0], -1
        ),
        "pscale": np.array(edata.pscale),
    }
    if sensi_order == amici.SensitivityOrder.none:
        (
            r_jax["llh"],
            (r_jax["x"], r_jax["y"], r_jax["stats"]),
        ) = jax_model._fun(**kwargs)
    elif sensi_order == amici.SensitivityOrder.first:
        (
            r_jax["llh"],
            r_jax["sllh"],
            (r_jax["x"], r_jax["y"], r_jax["stats"]),
        ) = jax_model._grad(**kwargs)
    elif sensi_order == amici.SensitivityOrder.second:
        (
            r_jax["llh"],
            r_jax["sllh"],
            r_jax["s2llh"],
            (r_jax["x"], r_jax["y"], r_jax["stats"]),
        ) = jax_model._hessian(**kwargs)

    for field in fields:
        for r_amici, r_jax in zip(rs_amici, [r_jax]):
            assert_allclose(
                actual=r_amici[field],
                desired=r_jax[field],
                atol=1e-6,
                rtol=1e-6,
            )
