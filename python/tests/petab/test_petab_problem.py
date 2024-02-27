from amici.petab.petab_problem import PetabProblem
from benchmark_models_petab import get_problem
from amici.testing import skip_on_valgrind


@skip_on_valgrind
def test_amici_petab_problem_pregenerate():
    """PetabProblem with pre-generated ExpDatas"""
    # any example is fine - the only assumption is that we don't have
    #  preequilibration
    petab_problem = get_problem("Boehm_JProteomeRes2014")
    app = PetabProblem(petab_problem, store_edatas=True)

    # ensure all edatas are generated upon construction
    assert len(app._edatas) == len(
        petab_problem.get_simulation_conditions_from_measurement_df()
    )

    # ensure the cached edatas are returned
    for i, (_, condition) in enumerate(
        petab_problem.get_simulation_conditions_from_measurement_df().iterrows()
    ):
        assert app.get_edata(condition.simulationConditionId) is app._edatas[i]

    # ensure parameter are updated
    edatas = app.get_edatas()
    app.set_parameters(
        {app.model.getParameterIds()[0]: 0.12345}, scaled_parameters=True
    )
    for edata in edatas:
        assert edata.parameters[0] == 0.12345


@skip_on_valgrind
def test_amici_petab_problem_on_demand():
    """PetabProblem with on-demand ExpDatas"""
    # any example is fine - the only assumption is that we don't have
    #  preequilibration
    petab_problem = get_problem("Boehm_JProteomeRes2014")
    app = PetabProblem(petab_problem, store_edatas=False)

    # ensure no edatas are generated upon construction
    assert not app._edatas

    edatas = app.get_edatas()
    assert len(edatas) == len(
        petab_problem.get_simulation_conditions_from_measurement_df()
    )

    # ensure parameter are updated
    app.set_parameters(
        {app.model.getParameterIds()[0]: 0.12345}, scaled_parameters=True
    )
    # previously generated ExpDatas are not updated
    for edata in edatas:
        assert edata.parameters[0] != 0.12345
    # but newly generated ExpDatas are
    for edata in app.get_edatas():
        assert edata.parameters[0] == 0.12345

    some_sim_condition = (
        petab_problem.measurement_df.simulationConditionId.iloc[0]
    )
    # different objects for subsequent calls
    assert app.get_edata(some_sim_condition) is not app.get_edata(
        some_sim_condition
    )


@skip_on_valgrind
def test_amici_petab_problem_pregenerate_equals_on_demand():
    """Check that PetabProblem produces the same ExpDatas
    independent of the `store_edatas` parameter."""
    # any example is fine
    petab_problem = get_problem("Boehm_JProteomeRes2014")
    app_store_true = PetabProblem(petab_problem, store_edatas=True)
    app_store_false = PetabProblem(petab_problem, store_edatas=False)

    parameter_update = {app_store_true.model.getParameterIds()[0]: 0.12345}
    app_store_true.set_parameters(parameter_update, scaled_parameters=True)
    app_store_false.set_parameters(parameter_update, scaled_parameters=True)

    for edata_store_true, edata_store_false in zip(
        app_store_true.get_edatas(), app_store_false.get_edatas()
    ):
        assert edata_store_true is not edata_store_false
        assert edata_store_true == edata_store_false
