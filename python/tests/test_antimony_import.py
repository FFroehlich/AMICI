import amici
import numpy as np
from amici.antimony_import import antimony2amici
from amici.testing import TemporaryDirectoryWinSafe as TemporaryDirectory
from amici.testing import skip_on_valgrind


@skip_on_valgrind
def test_antimony_example():
    """If this example requires changes, please also update documentation/python_interface.rst."""
    ant_model = """
    model lotka_volterra
        # see https://en.wikipedia.org/wiki/Lotka%E2%80%93Volterra_equations

        # initial conditions
        prey_density = 10;
        predator_density = 10;

        # parameters
        prey_growth_rate = 1.1;
        predator_effect_on_prey = 0.4;
        predator_death_rate = 0.4;
        prey_effect_on_predator = 0.1;

        # dx/dt
        prey_density' = prey_growth_rate * prey_density - predator_effect_on_prey * prey_density * predator_density;
        predator_density' = prey_effect_on_predator * prey_density * predator_density - predator_death_rate * predator_density;
    end
    """
    module_name = "test_antimony_example_lv"
    with TemporaryDirectory(prefix=module_name) as outdir:
        antimony2amici(
            ant_model,
            model_name=module_name,
            output_dir=outdir,
        )
        model_module = amici.import_model_module(
            module_name=module_name, module_path=outdir
        )
        amici_model = model_module.getModel()
        amici_model.setTimepoints(np.linspace(0, 100, 200))
        amici_solver = amici_model.getSolver()
        rdata = amici.runAmiciSimulation(amici_model, amici_solver)
        assert rdata.status == amici.AMICI_SUCCESS
