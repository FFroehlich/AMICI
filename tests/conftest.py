"""Pytest configuration for SBML test suite"""

import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from _pytest.reports import TestReport

# stores passed SBML semantic test suite IDs
passed_ids = []

SBML_SEMANTIC_CASES_DIR = (
    Path(__file__).parent / "sbml-test-suite" / "cases" / "semantic"
)


@pytest.fixture
def sbml_semantic_cases_dir() -> Path:
    """directory with sbml semantic test cases"""
    return SBML_SEMANTIC_CASES_DIR


def parse_selection(selection_str: str, last: int) -> list[int]:
    """
    Parse comma-separated list of integer ranges, return selected indices as
    integer list

    Valid input e.g.: "1", "1,3", "-3,4,6-7"
    """
    indices = []
    for group in selection_str.split(","):
        if not re.match(r"^(?:-?\d+|\d+-\d*)$", group):
            print("Invalid selection", group)
            sys.exit()
        spl = group.split("-")
        if len(spl) == 1:
            indices.append(int(spl[0]))
        elif len(spl) == 2:
            begin = int(spl[0]) if spl[0] else 0
            end = int(spl[1]) if spl[1] else last
            indices.extend(range(begin, end + 1))
    return indices


def get_all_semantic_case_ids():
    """Get iterator over test sorted IDs of all cases in the SBML semantic
    suite"""
    pattern = re.compile(r"\d{5}")
    return sorted(
        str(x.name)
        for x in SBML_SEMANTIC_CASES_DIR.iterdir()
        if pattern.match(x.name)
    )


def pytest_addoption(parser):
    """Add pytest CLI options"""
    parser.addoption("--cases", help="Test cases to run")


def pytest_generate_tests(metafunc):
    """Parameterize tests"""

    # Run for all SBML semantic test suite cases
    if "test_number" in metafunc.fixturenames:
        # Get CLI option
        cases = metafunc.config.getoption("cases")
        if cases:
            # iff specific case IDs are given and the SBML semantic test suite is not there, we should fail.
            if not SBML_SEMANTIC_CASES_DIR.exists():
                raise ValueError(
                    "The SBML semantic cases are missing. You can install them with "
                    "'AMICI/scripts/run-SBMLTestsuite.sh'."
                )

            # Run selected tests
            last_id = int(list(get_all_semantic_case_ids())[-1])
            test_numbers = sorted(set(parse_selection(cases, last_id)))
        else:
            # Run all tests
            test_numbers = get_all_semantic_case_ids()

        metafunc.parametrize("test_number", test_numbers)


def pytest_sessionfinish(session, exitstatus):
    """Process test results"""
    global passed_ids
    terminalreporter = session.config.pluginmanager.get_plugin(
        "terminalreporter"
    )
    terminalreporter.ensure_newline()
    # parse test names to get passed case IDs (don't know any better way to
    # access fixture values)
    from testSBMLSuite import format_test_id

    passed_ids = [format_test_id(_) for _ in passed_ids]
    if passed_ids:
        write_passed_tags(passed_ids, terminalreporter)
    terminalreporter.ensure_newline()


def write_passed_tags(passed_ids, out=sys.stdout):
    """Write tags of passed SBML semantic test cases"""
    passed_component_tags = set()
    passed_test_tags = set()

    for test_id in passed_ids:
        cur_component_tags, cur_test_tags = get_tags_for_test(test_id)
        passed_component_tags |= cur_component_tags
        passed_test_tags |= cur_test_tags

    out.write(
        "\nAt least one test with the following component tags has "
        "passed:\n"
    )
    out.write("  " + "\n  ".join(sorted(passed_component_tags)))
    out.write(
        "\n\nAt least one test with the following test tags has " "passed:\n"
    )
    out.write("  " + "\n  ".join(sorted(passed_test_tags)))


def pytest_runtest_logreport(report: "TestReport") -> None:
    """Collect test case IDs of passed SBML semantic test suite cases"""
    if (
        report.when == "call"
        and report.outcome == "passed"
        and "::test_sbml_testsuite_case[" in report.nodeid
    ):
        test_case_id = re.sub(
            r"^.*::test_sbml_testsuite_case\[(\d+)].*$", r"\1", report.nodeid
        )
        passed_ids.append(test_case_id)


def get_tags_for_test(test_id: str) -> tuple[set[str], set[str]]:
    """Get sbml test suite tags for the given test ID

    Returns:
        Tuple of set of strings for componentTags and testTags
    """
    current_test_path = SBML_SEMANTIC_CASES_DIR / test_id
    info_file = current_test_path / f"{test_id}-model.m"
    with open(info_file) as f:
        component_tags = set()
        test_tags = set()
        for line in f:
            if line.startswith("testTags:"):
                test_tags = set(
                    re.split(r"[ ,:]", line[len("testTags:") :].strip())
                )
                test_tags.discard("")
            if line.startswith("componentTags:"):
                component_tags = set(
                    re.split(r"[ ,:]", line[len("componentTags:") :].strip())
                )
                component_tags.discard("")
            if test_tags and component_tags:
                return component_tags, test_tags
    print(f"No componentTags or testTags found for test case {test_id}.")
    return component_tags, test_tags
