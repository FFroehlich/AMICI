"""Pytest configuration for PEtab test suite"""

from typing import List
import re
import sys
from petabtests.core import get_cases


def parse_selection(selection_str: str) -> List[int]:
    """
    Parse comma-separated list of integer ranges, return selected indices as
    integer list

    Valid input e.g.: "1", "1,3", "-3,4,6-7"
    """
    indices = []
    for group in selection_str.split(','):
        if not re.match(r'^(?:-?\d+)|(?:\d+(?:-\d+))$', group):
            print("Invalid selection", group)
            sys.exit()
        spl = group.split('-')
        if len(spl) == 1:
            indices.append(int(spl[0]))
        elif len(spl) == 2:
            begin = int(spl[0]) if spl[0] else 0
            end = int(spl[1])
            indices.extend(range(begin, end + 1))
    return indices


def pytest_addoption(parser):
    """Add pytest CLI options"""
    parser.addoption("--petab-cases", help="Test cases to run")
    parser.addoption("--only-pysb", help="Run only PySB tests",
                     action="store_true")
    parser.addoption("--only-sbml", help="Run only SBML tests",
                     action="store_true", )


def pytest_generate_tests(metafunc):
    """Parameterize tests"""

    # Run for all PEtab test suite cases
    if "case" in metafunc.fixturenames \
            and "model_type" in metafunc.fixturenames:

        # Get CLI option
        cases = metafunc.config.getoption("--petab-cases")
        if cases:
            # Run selected tests
            test_numbers = parse_selection(cases)
        else:
            # Run all tests
            test_numbers = None

        if metafunc.config.getoption("--only-sbml"):
            test_numbers = test_numbers if test_numbers else get_cases("sbml")
            argvalues = [(case, 'sbml') for case in test_numbers]
        elif metafunc.config.getoption("--only-pysb"):
            test_numbers = test_numbers if test_numbers else get_cases("pysb")
            argvalues = [(case, 'pysb') for case in test_numbers]
        else:
            argvalues = []
            for format in ('sbml', test_petab_test_suite.yml'pysb'):
                argvalues.extend((case, format)
                                 for case in test_numbers or get_cases(format))
        metafunc.parametrize("case,model_type", argvalues)
