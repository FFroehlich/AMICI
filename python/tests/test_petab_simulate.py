"""Tests for petab_simulate.py."""
import tempfile
from pathlib import Path

import petab
import petabtests
import pytest
from amici.petab.simulator import PetabSimulator
from amici.testing import skip_on_valgrind


@pytest.fixture
def petab_problem() -> petab.Problem:
    """Create a PEtab problem for use in tests."""
    test_case = "0001"
    test_case_dir = petabtests.get_case_dir(
        id_=test_case, format_="sbml", version="v1.0.0"
    )
    petab_yaml_path = test_case_dir / petabtests.problem_yaml_name(test_case)
    return petab.Problem.from_yaml(str(petab_yaml_path))


@skip_on_valgrind
def test_simulate_without_noise(petab_problem):
    """Test the reproducibility of simulation without noise."""
    simulator = PetabSimulator(petab_problem)
    print(simulator.working_dir)
    synthetic_data_df_a = simulator.simulate()
    synthetic_data_df_b = simulator.simulate()
    simulator.remove_working_dir()
    # Repeated simulation without noise is reproducible.
    assert synthetic_data_df_b.equals(synthetic_data_df_a)

    simulator = PetabSimulator(petab_problem)
    print(simulator.working_dir)
    synthetic_data_df_c = simulator.simulate()
    simulator.remove_working_dir()
    # Repeated simulation without noise, with a different PetabSimulator
    # instance, is reproducible.
    assert synthetic_data_df_c.equals(synthetic_data_df_a)


@skip_on_valgrind
def test_subset_call(petab_problem):
    """
    Test the ability to customize AMICI methods, specifically:
    :py:func:`amici.petab_import.import_petab_problem` (`model_name`,
    `model_output_dir`, import is skipped if `amici_model` is specified), and
    :py:func:`amici.petab_objective.simulate_petab` (`amici_model`, `solver`).
    """
    model_name = "model_name_dummy"
    model_output_dir = tempfile.mkdtemp()

    simulator0 = PetabSimulator(petab_problem)
    assert not (Path(model_output_dir) / model_name).is_dir()
    simulator0.simulate(
        model_name=model_name, model_output_dir=model_output_dir
    )
    # Model name is handled correctly
    assert simulator0.amici_model.getName() == model_name
    # Check model output directory is created, by
    # :py:func:`amici.petab_import.import_petab_problem`
    assert (Path(model_output_dir) / model_name).is_dir()

    simulator = PetabSimulator(petab_problem)
    simulator.simulate(amici_model=simulator0.amici_model)
    # AMICI model is handled correctly to skip import
    assert simulator.amici_model.getName() == model_name
    simulator.simulate()
    # AMICI model persists between :py:func:`PetabSimulator.simulate` calls
    assert simulator.amici_model.getName() == model_name
    # Erroneous solver raises an error
    with pytest.raises(TypeError):
        simulator.simulate(solver=False)

    simulator0.remove_working_dir()
    simulator.remove_working_dir()
