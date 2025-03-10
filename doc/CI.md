# Continuous integration (CI) and tests

AMICI uses a continuous integration pipeline running via
https://github.com/features/actions.
This includes the following steps:

- Checking existence and format of documentation
- Static code analysis (http://cppcheck.sourceforge.net/)
- Unit and integration tests
- Memory leak detection

More details are provided in the sections below.

The CI scripts and tests can be found in `tests/` and `scripts/`. Some of the
tests are integrated with CMake, see `make help` in the build directory.


## C++ unit and integration tests

To run C++ tests, build AMICI with `make` or `scripts/buildAll.sh`,
then run `scripts/run-cpp-tests.sh`.


## Python unit and integration tests

To run Python tests, run `../scripts/run-python-tests.sh` from anywhere
(assumes build directory is `build/`) or run `make python-tests` in your build
directory.

### SBML Test Suite

We test Python-AMICI SBML support using the test cases from the semantic
[SBML Test Suite](https://github.com/sbmlteam/sbml-test-suite/). When making
changes to the model import functions, make sure to run these tests.

To run the SBML Test Suite test cases, the easiest way is:

1. Running `scripts/installAmiciSource.sh` which
   creates a virtual Python environment and performs a development installation
   of AMICI from the current repository. (This needs to be run only once or
   after AMICI model generation or C++ changes).

2. Running `scripts/run-SBMLTestsuite.sh`. This will download the test cases
   if necessary and run them all. A subset of test cases can be selected with
   an optional argument (e.g. `scripts/run-SBMLTestsuite.sh 1,3-6,8`, to run
   cases 1, 3, 4, 5, 6 and 8).

   Once the test cases are available locally, for debugging it might be easier
   to directly use `pytest` with `tests/testSBMLSuite.py`.


## Matlab tests (not included in CI pipeline)

To execute the Matlab test suite, run `tests/testModels.m`.


## Model simulation integration tests

Many of our integration tests are model simulations. The simulation results
obtained from the Python and C++ are compared to results saved in an HDF5 file
(`tests/cpp/expectedResults.h5`).
Settings and data for the test simulations are also specified in this file.

**Note:** The C++ code for the models is included in the repository under
`models/`.
This code is to be updated whenever `amici::Model` changes.


### Regenerating C++ code of the test models

Regeneration of the model code has to be done whenever `amici::Model` or
the Matlab model import routines change.

This is done with

    tests/cpp/wrapTestModels.m

**Note:** This is currently only possible from Matlab < R2018a. This should
change as soon as 1) all second-order sensitivity code is ported to C++/Python,
2) a non-SBML import exists for Python and 3) support for events has been added
for Python.


### Regenerating expected results

To update test results, run `make test` in the build directory,
replace `tests/cpp/expectedResults.h5` by
`tests/cpp/writeResults.h5.bak`
[ONLY DO THIS AFTER TRIPLE CHECKING CORRECTNESS OF RESULTS]
Before replacing the test results, confirm that only expected datasets have
changed, e.g. using

    h5diff -v --relative 1e-8 tests/cpp/expectedResults.h5 tests/cpp/writeResults.h5.bak | less


## Adding/Updating tests

To add new tests add a new corresponding python script (see, e.g.,
`./tests/generateTestConfig/example_dirac.py`) and add it to and run
`tests/generateTestConfigurationForExamples.sh`.
Then regenerate the expected test results (see above).
