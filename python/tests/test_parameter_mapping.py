"""Test for ``amici.parameter_mapping``"""

from amici.petab.parameter_mapping import (
    ParameterMapping,
    ParameterMappingForCondition,
)
from amici.testing import skip_on_valgrind


@skip_on_valgrind
def test_parameter_mapping_for_condition_default_args():
    """Check we can initialize the mapping with default arguments."""

    par_map_for_condition = ParameterMappingForCondition()
    for attr in [
        "map_sim_var",
        "scale_map_sim_var",
        "map_preeq_fix",
        "scale_map_preeq_fix",
        "map_sim_fix",
        "scale_map_sim_fix",
    ]:
        assert not getattr(par_map_for_condition, attr)

    map_sim_var = {"sim_par0": 8, "sim_par1": "opt_par0"}
    map_preeq_fix = {"sim_par2": "opt_par1"}
    map_sim_fix = {"sim_par2": "opt_par2"}
    par_map_for_condition = ParameterMappingForCondition(
        map_sim_var=map_sim_var,
        map_preeq_fix=map_preeq_fix,
        map_sim_fix=map_sim_fix,
    )

    expected_scale_map_sim_var = {"sim_par0": "lin", "sim_par1": "lin"}
    expected_scale_map_preeq_fix = {"sim_par2": "lin"}
    expected_scale_map_sim_fix = {"sim_par2": "lin"}

    assert (
        par_map_for_condition.scale_map_sim_var == expected_scale_map_sim_var
    )
    assert (
        par_map_for_condition.scale_map_preeq_fix
        == expected_scale_map_preeq_fix
    )
    assert (
        par_map_for_condition.scale_map_sim_fix == expected_scale_map_sim_fix
    )


@skip_on_valgrind
def test_parameter_mapping():
    """Test :class:``amici.parameter_mapping.ParameterMapping``."""

    parameter_mapping = ParameterMapping()
    assert len(parameter_mapping) == 0

    map_sim_var = {"sim_par0": 8, "sim_par1": "opt_par0"}
    map_preeq_fix = {"sim_par2": "opt_par1"}
    map_sim_fix = {"sim_par2": "opt_par2"}
    par_map_for_condition = ParameterMappingForCondition(
        map_sim_var=map_sim_var,
        map_preeq_fix=map_preeq_fix,
        map_sim_fix=map_sim_fix,
    )

    parameter_mapping.append(par_map_for_condition)

    assert len(parameter_mapping) == 1

    assert isinstance(parameter_mapping[0], ParameterMappingForCondition)
    assert isinstance(parameter_mapping[:], ParameterMapping)
