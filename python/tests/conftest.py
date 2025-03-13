"""pytest configuration file"""

import copy
import importlib
import sys

import amici
import pytest
from amici.testing import TemporaryDirectoryWinSafe as TemporaryDirectory
from pathlib import Path

EXAMPLES_DIR = Path(__file__).parents[2] / "doc" / "examples"
TEST_DIR = Path(__file__).parent
MODEL_STEADYSTATE_SCALED_XML = (
    EXAMPLES_DIR / "getting_started" / "model_steadystate_scaled.xml"
)
MODEL_PRESIMULATION_XML = (
    EXAMPLES_DIR / "example_presimulation" / "model_presimulation.xml"
)
MODEL_CONSTANT_SPECIES_XML = (
    EXAMPLES_DIR / "example_steady_states" / "model_constant_species.xml"
)


@pytest.fixture(scope="session")
def sbml_example_presimulation_module():
    """SBML example_presimulation model module fixture"""

    sbml_importer = amici.SbmlImporter(MODEL_PRESIMULATION_XML)

    constant_parameters = ["DRUG_0", "KIN_0"]

    observables = amici.assignmentRules2observables(
        sbml_importer.sbml,  # the libsbml model object
        filter_function=lambda variable: variable.getName() == "pPROT_obs",
    )
    module_name = "test_model_presimulation"

    with TemporaryDirectory(prefix=module_name) as outdir:
        sbml_importer.sbml2amici(
            model_name=module_name,
            output_dir=outdir,
            verbose=False,
            observables=observables,
            constant_parameters=constant_parameters,
        )

        yield amici.import_model_module(
            module_name=module_name, module_path=outdir
        )


@pytest.fixture(scope="session")
def pysb_example_presimulation_module():
    """PySB example_presimulation model module fixture"""
    pysb = pytest.importorskip("pysb")
    from amici.pysb_import import pysb2amici

    constant_parameters = ["DRUG_0", "KIN_0"]

    pysb.SelfExporter.cleanup()  # reset pysb
    pysb.SelfExporter.do_export = True

    model_path = MODEL_PRESIMULATION_XML.parent

    with amici.add_path(model_path):
        if "createModelPresimulation" in sys.modules:
            importlib.reload(sys.modules["createModelPresimulation"])
            model_module = sys.modules["createModelPresimulation"]
        else:
            model_module = importlib.import_module("createModelPresimulation")
    model = copy.deepcopy(model_module.model)

    model.name = "test_model_presimulation_pysb"

    with TemporaryDirectory(prefix=model.name) as outdir:
        pysb2amici(
            model,
            outdir,
            verbose=True,
            observables=["pPROT_obs"],
            constant_parameters=constant_parameters,
        )

        yield amici.import_model_module(model.name, outdir)


@pytest.fixture(scope="session")
def model_units_module():
    sbml_file = TEST_DIR / "model_units.xml"
    module_name = "test_model_units"

    sbml_importer = amici.SbmlImporter(sbml_file)

    with TemporaryDirectory() as outdir:
        sbml_importer.sbml2amici(model_name=module_name, output_dir=outdir)

        yield amici.import_model_module(
            module_name=module_name, module_path=outdir
        )
