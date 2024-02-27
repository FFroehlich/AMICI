"""Tests for ``amici.pandas``"""

import itertools

import amici
import numpy as np
import pytest
from amici.testing import skip_on_valgrind


# test parameters for test_pandas_import_export
combos = itertools.product([(10, 5), (5, 10), ()], repeat=3)
cases = [
    {
        "fixedParameters": combo[0],
        "fixedParametersPreequilibration": combo[1],
        "fixedParametersPresimulation": combo[2],
    }
    for combo in combos
]


@skip_on_valgrind
@pytest.mark.parametrize("case", cases)
def test_pandas_import_export(sbml_example_presimulation_module, case):
    """TestCase class for testing csv import using pandas"""

    # setup
    model = sbml_example_presimulation_module.getModel()
    model.setTimepoints(np.linspace(0, 60, 61))
    solver = model.getSolver()
    rdata = amici.runAmiciSimulation(model, solver)
    edata = [amici.ExpData(rdata, 0.01, 0)]

    # test copy constructor
    _ = amici.ExpData(edata[0])

    for fp in case:
        setattr(edata[0], fp, case[fp])

    df_edata = amici.getDataObservablesAsDataFrame(model, edata)
    edata_reconstructed = amici.getEdataFromDataFrame(model, df_edata)

    for fp in [
        "fixedParameters",
        "fixedParametersPreequilibration",
        "fixedParametersPresimulation",
    ]:
        if fp != "fixedParameters" or case[fp] != ():
            assert getattr(edata[0], fp) == getattr(edata_reconstructed[0], fp)

            assert case[fp] == getattr(edata_reconstructed[0], fp)

        else:
            assert model.getFixedParameters() == getattr(
                edata_reconstructed[0], fp
            )

            assert model.getFixedParameters() == getattr(
                edata_reconstructed[0], fp
            )

        assert getattr(edata[0], fp) == case[fp]
