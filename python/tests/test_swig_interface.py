"""Test SWIG interface using python/examples/test_model_presimulation_pysb

Test getters, setters, etc.
"""

import copy
import numbers

import pytest

import amici
import numpy as np


def test_version_number(pysb_example_presimulation_module):
    model = pysb_example_presimulation_module.getModel()
    assert model.getAmiciVersion() == amici.__version__
    assert model.getAmiciCommit() == amici.__commit__


def test_copy_constructors(pysb_example_presimulation_module):
    model = pysb_example_presimulation_module.getModel()
    solver = model.getSolver()

    for obj in [model, solver]:
        for attr in dir(obj):
            if (
                attr.startswith("__")
                or attr == "this"
                or attr == "thisown"
                or is_callable_but_not_getter(obj, attr)
            ):
                continue

            # objects will be initialized with default values so we
            # can check if the clone routine does the right thing by
            # modifying the value from default, cloning and checking
            # if the change was carried over to the clone
            val = get_val(obj, attr)

            try:
                modval = get_mod_val(val, attr)
            except ValueError:
                # happens for everything that is not bool or scalar
                continue

            try:
                set_val(obj, attr, modval)
            except AttributeError:
                # some attributes cannot be set
                continue

            obj_clone = obj.clone()

            assert get_val(obj, attr) == get_val(
                obj_clone, attr
            ), f"{obj} - {attr}"


# `None` values are skipped in `test_model_instance_settings`.
# Keys are suffixes of `get[...]` and `set[...]` `amici.Model` methods.
# If either the getter or setter is not named with this pattern, then the key
# is a tuple where the first and second elements are the getter and setter
# methods, respectively.
# Values are lists where the first element is the default value of the test
# model, and the second value is some custom value.
# Default values are based on `pysb_example_presimulation_module`.
model_instance_settings0 = {
    # setting name: [default value, custom value]
    "AddSigmaResiduals": [False, True],
    "AlwaysCheckFinite": [False, True],
    # Skipped due to model dependency in `'InitialStates'`.
    "FixedParameters": None,
    "InitialStates": [
        (10.0, 9.0, 1.0, 0.0, 0.0, 0.0),
        tuple([0.1] * 6),
    ],
    ("getInitialStateSensitivities", "setUnscaledInitialStateSensitivities"): [
        tuple([1.0] + [0.0] * 35),
        tuple([0.1] * 36),
    ],
    "MinimumSigmaResiduals": [
        50.0,
        60.0,
    ],
    ("nMaxEvent", "setNMaxEvent"): [
        10,
        20,
    ],
    "Parameters": [(10.0, 0.1, 0.1, 0.1, 0.1, 0.1), tuple([1.0] * 6)],
    # Skipped due to interdependency with `'InitialStateSensitivities'`.
    "ParameterList": None,
    # Skipped due to interdependency with `'InitialStateSensitivities'`.
    "ParameterScale": None,
    # Skipped due to interdependencies with
    # `'ReinitializeFixedParameterInitialStates'`.
    "ReinitializationStateIdxs": None,
    # Skipped due to interdependencies with `'ReinitializationStateIdxs'`.
    "ReinitializeFixedParameterInitialStates": None,
    # Skipped due to conservation laws in the test model
    # `pysb_example_presimulation_module.getModel()`.
    "StateIsNonNegative": None,
    "SteadyStateComputationMode": [
        2,
        1,
    ],
    "SteadyStateSensitivityMode": [
        2,
        1,
    ],
    ("t0", "setT0"): [
        0.0,
        1.0,
    ],
    "Timepoints": [
        tuple(),
        (1.0, 2.0, 3.0),
    ],
}


def test_model_instance_settings(pysb_example_presimulation_module):
    model0 = pysb_example_presimulation_module.getModel()

    # Indexes of values in the `model_instance_settings0` dictionary.
    i_default = 0
    i_custom = 1

    i_getter = 0
    i_setter = 1

    # the default setting for AlwaysCheckFinite depends on whether the amici
    # extension has been built in debug mode
    model_instance_settings0["AlwaysCheckFinite"] = [
        model0.getAlwaysCheckFinite(),
        not model0.getAlwaysCheckFinite(),
    ]

    # All settings are tested.
    assert set(model_instance_settings0) == set(
        amici.swig_wrappers.model_instance_settings
    )

    # Skip settings with interdependencies.
    model_instance_settings = {
        k: v for k, v in model_instance_settings0.items() if v is not None
    }

    # All custom values are different to default values.
    assert all(
        default != custom
        for name, (default, custom) in model_instance_settings.items()
        if name != "ReinitializeFixedParameterInitialStates"
    )

    # All default values are as expected.
    for name, (default, custom) in model_instance_settings.items():
        getter = name[i_getter] if isinstance(name, tuple) else f"get{name}"
        setter = name[i_setter] if isinstance(name, tuple) else f"set{name}"
        # Default values are as expected.
        assert getattr(model0, getter)() == default
        # Custom value is set correctly.
        getattr(model0, setter)(custom)
        assert getattr(model0, getter)() == custom

    # The grouped getter method works.
    custom_settings = amici.get_model_settings(model0)
    for name in model_instance_settings:
        assert custom_settings[name] == model_instance_settings[name][i_custom]

    # Create a new model for comparison.
    model = pysb_example_presimulation_module.getModel()

    # The new model has the default settings.
    model_default_settings = amici.get_model_settings(model)
    for name in model_instance_settings:
        if (
            name == "InitialStates" and not model.hasCustomInitialStates()
        ) or (
            name
            == (
                "getInitialStateSensitivities",
                "setUnscaledInitialStateSensitivities",
            )
            and not model.hasCustomInitialStateSensitivities()
        ):
            # Here the expected value differs from what the getter would return
            assert model_default_settings[name] == []
        else:
            assert (
                model_default_settings[name]
                == model_instance_settings[name][i_default]
            ), name

    # The grouped setter method works.
    custom_settings_not_none = {
        name: value
        for name, value in custom_settings.items()
        if model_instance_settings0[name] is not None
    }
    amici.set_model_settings(model, custom_settings_not_none)
    assert all(
        value == custom_settings_not_none[name]
        for name, value in amici.get_model_settings(model).items()
        if name in custom_settings_not_none
    )


def test_interdependent_settings(pysb_example_presimulation_module):
    """Test settings that were not tested in `test_model_instance_settings`.

    `StateIsNonNegative` is still skipped, due to conservation laws in the
    test model.
    """
    model = pysb_example_presimulation_module.getModel()

    original_settings = {
        "FixedParameters": (9.0, 1.0),
        "ParameterList": (0, 1, 2, 3, 4, 5),
        "ParameterScale": [0, 0, 0, 0, 0, 0],
        "ReinitializationStateIdxs": tuple(),
        "ReinitializeFixedParameterInitialStates": False,
        "StateIsNonNegative": (False, False, False),
    }

    expected_settings = {
        "FixedParameters": (8.0, 2.0),
        "ParameterList": (0, 1, 2, 3, 4),
        "ParameterScale": [1, 0, 0, 0, 0, 0],
        "ReinitializationStateIdxs": (0,),
        "ReinitializeFixedParameterInitialStates": True,
        # Skipped due to conservation laws in the test model.
        # 'StateIsNonNegative': None,
    }

    # Some values need to be transformed to be tested in Python
    # (e.g. SWIG objects). Default transformer is no transformation
    # (the identity function).
    getter_transformers = {
        setting: (lambda x: x) for setting in original_settings
    }
    getter_transformers.update(
        {
            # Convert from SWIG object.
            "ParameterScale": lambda x: list(x)
        }
    )

    default_settings = amici.get_model_settings(model)
    for original_setting, original_setting_value in original_settings.items():
        test_value = getter_transformers[original_setting](
            default_settings[original_setting],
        )
        # The original model is as expected (to ensure the test is still
        # valid).
        assert test_value == original_setting_value

    for setting, expected_value in expected_settings.items():
        input_settings = {setting: copy.deepcopy(expected_value)}

        amici.set_model_settings(model, input_settings)
        output_settings = amici.get_model_settings(model)
        test_value = getter_transformers[setting](output_settings[setting])
        # The setter works.
        assert test_value == expected_value

        input_settings = {setting: output_settings[setting]}
        amici.set_model_settings(model, input_settings)
        output_settings = amici.get_model_settings(model)
        test_value = getter_transformers[setting](output_settings[setting])
        # (round-trip) The output of the getter can be used as input to the
        # setter, and does not change the value.
        assert test_value == expected_value


def test_unhandled_settings(pysb_example_presimulation_module):
    """Detect possible getters and setters that are not yet handled.

    Currently, only model attributes that begin with `'get'` and `'set'` are
    tested.

    The following getters/setters are untested here, due to unusual naming
    (not prefixed by `'get'` or `'set'`).
    - nMaxEvent
    - t0
    """
    model = pysb_example_presimulation_module.getModel()

    not_handled = [
        "get",
        "getAmiciCommit",
        "getAmiciVersion",
        "getExpressionIds",
        "getExpressionNames",
        "getFixedParameterById",
        "getFixedParameterByName",
        "getFixedParameterIds",
        "getFixedParameterNames",
        "getName",
        "getObservableIds",
        "getObservableNames",
        "getObservableScaling",
        "getParameterById",
        "getParameterByName",
        "getParameterIds",
        "getParameterNames",
        "getSolver",
        "getStateIds",
        "getStateNames",
        "getStateIdsSolver",
        "getStateNamesSolver",
        "getTimepoint",
        "getUnscaledParameters",
        "setAllStatesNonNegative",
        "setFixedParameterById",
        "setFixedParameterByName",
        "setFixedParametersByIdRegex",
        "setFixedParametersByNameRegex",
        "setParameterById",
        "setParameterByName",
        "setParametersByIdRegex",
        "setParametersByNameRegex",
        "setInitialStateSensitivities",
        "get_trigger_timepoints",
    ]
    from amici.swig_wrappers import model_instance_settings

    handled = [
        name
        for names in model_instance_settings
        for name in (
            names
            if isinstance(names, tuple)
            else (f"get{names}", f"set{names}")
        )
    ]

    for attribute in dir(model):
        if attribute[:3] in ["get", "set"] and attribute not in not_handled:
            assert attribute in handled, attribute


def is_callable_but_not_getter(obj, attr):
    if not callable(getattr(obj, attr)):
        return False

    if attr.startswith("get"):
        return (
            "set" + attr[3:] not in dir(obj)
            or attr.endswith("ById")
            or attr.endswith("ByName")
        )
    else:
        return True


def get_val(obj, attr):
    if callable(getattr(obj, attr)):
        return getattr(obj, attr)()
    else:
        return getattr(obj, attr)


def get_mod_val(val, attr):
    if attr == "getReturnDataReportingMode":
        return amici.RDataReporting.likelihood
    elif attr == "getParameterList":
        return tuple(get_mod_val(val[0], "") for _ in val)
    elif attr == "getStateIsNonNegative":
        raise ValueError("Cannot modify value")
    elif isinstance(val, bool):
        return not val
    elif isinstance(val, numbers.Number):
        return val + 1
    elif isinstance(val, tuple):
        return tuple(get_mod_val(v, attr) for v in val)

    raise ValueError("Cannot modify value")


def set_val(obj, attr, val):
    if callable(getattr(obj, attr)):
        getattr(obj, "set" + attr[3:])(val)
    else:
        setattr(obj, attr, val)


def test_model_instance_settings_custom_x0(pysb_example_presimulation_module):
    """Check that settings are applied in the correct order, and only if
    required"""
    model = pysb_example_presimulation_module.getModel()

    # ensure no-custom-(s)x0 is restored
    assert not model.hasCustomInitialStates()
    assert not model.hasCustomInitialStateSensitivities()
    settings = amici.get_model_settings(model)
    model.setInitialStates(model.getInitialStates())
    model.setUnscaledInitialStateSensitivities(
        model.getInitialStateSensitivities()
    )
    amici.set_model_settings(model, settings)
    assert not model.hasCustomInitialStates()
    assert not model.hasCustomInitialStateSensitivities()
    # ensure everything was set correctly, and there wasn't any problem
    #  due to, e.g. interactions of different setters
    assert settings == amici.get_model_settings(model)

    # ensure custom (s)x0 is restored
    model.setInitialStates(model.getInitialStates())
    model.setParameterScale(amici.ParameterScaling.log10)
    sx0 = model.getInitialStateSensitivities()
    model.setUnscaledInitialStateSensitivities(sx0)
    assert model.hasCustomInitialStates()
    assert model.hasCustomInitialStateSensitivities()
    settings = amici.get_model_settings(model)
    model2 = pysb_example_presimulation_module.getModel()
    amici.set_model_settings(model2, settings)
    assert model2.hasCustomInitialStates()
    assert model2.hasCustomInitialStateSensitivities()
    assert model2.getInitialStateSensitivities() == sx0
    assert settings == amici.get_model_settings(model2)


def test_solver_repr():
    for solver in (amici.CVodeSolver(), amici.IDASolver()):
        solver_ptr = amici.SolverPtr(solver.this)
        for s in (solver, solver_ptr):
            assert "maxsteps" in str(s)
            assert "maxsteps" in repr(s)


def test_edata_repr():
    ny = 1
    nz = 2
    ne = 3
    nt = 4
    edata = amici.ExpData(ny, nz, ne, range(nt))
    edata_ptr = amici.ExpDataPtr(edata.this)
    expected_strs = (
        f"{nt}x{ny} time-resolved datapoints",
        f"{ne}x{nz} event-resolved datapoints",
        f"(0/{ny * nt} measurements",
        f"(0/{nz * ne} measurements",
    )
    for e in [edata, edata_ptr]:
        for expected_str in expected_strs:
            assert expected_str in str(e)
            assert expected_str in repr(e)


def test_edata_equality_operator():
    e1 = amici.ExpData(1, 2, 3, [3])
    e2 = amici.ExpData(1, 2, 3, [3])
    assert e1 == e2
    # check that comparison with other types works
    # this is not implemented by swig by default
    assert e1 != 1


def test_expdata_and_expdataview_are_deepcopyable():
    edata1 = amici.ExpData(3, 2, 3, range(4))
    edata1.setObservedData(np.zeros((3, 4)).flatten())

    # ExpData
    edata2 = copy.deepcopy(edata1)
    assert edata1 == edata2
    assert edata1.this != edata2.this
    edata2.setTimepoints([0])
    assert edata1 != edata2

    # ExpDataView
    ev1 = amici.ExpDataView(edata1)
    ev2 = copy.deepcopy(ev1)
    assert ev2._swigptr.this != ev1._swigptr.this
    assert ev1 == ev2


def test_solvers_are_deepcopyable():
    for solver_type in (amici.CVodeSolver, amici.IDASolver):
        for solver1 in (solver_type(), amici.SolverPtr(solver_type())):
            solver2 = copy.deepcopy(solver1)
            assert solver1.this != solver2.this
            assert (
                solver1.getRelativeTolerance()
                == solver2.getRelativeTolerance()
            )
            solver2.setRelativeTolerance(100 * solver2.getRelativeTolerance())
            assert (
                solver1.getRelativeTolerance()
                != solver2.getRelativeTolerance()
            )


def test_model_is_deepcopyable(pysb_example_presimulation_module):
    model_module = pysb_example_presimulation_module
    for model1 in (
        model_module.getModel(),
        amici.ModelPtr(model_module.getModel()),
    ):
        model2 = copy.deepcopy(model1)
        assert model1.this != model2.this
        assert model1.t0() == model2.t0()
        model2.setT0(100 + model2.t0())
        assert model1.t0() != model2.t0()


def test_rdataview(sbml_example_presimulation_module):
    """Test some SwigPtrView functionality via ReturnDataView."""
    model_module = sbml_example_presimulation_module
    model = model_module.getModel()
    rdata = amici.runAmiciSimulation(model, model.getSolver())
    assert isinstance(rdata, amici.ReturnDataView)

    # fields are accessible via dot notation and [] operator,
    #  __contains__ and __getattr__ are implemented correctly
    with pytest.raises(AttributeError):
        _ = rdata.nonexisting_attribute

    with pytest.raises(KeyError):
        _ = rdata["nonexisting_attribute"]

    assert not hasattr(rdata, "nonexisting_attribute")
    assert "x" in rdata
    assert rdata.x == rdata["x"]

    # field names are included by dir()
    assert "x" in dir(rdata)
