#!/bin/bash
# Without arguments: run Python test suite under valgrind
# With arguments: run whatever was passed as arguments under valgrind

script_path=$(dirname "${BASH_SOURCE[0]}")
amici_path=$(cd "$script_path"/.. && pwd)

set -e

if [[ -z "${BNGPATH}" ]]; then
    export BNGPATH=${amici_path}/ThirdParty/BioNetGen-2.7.0
fi
suppressions="${amici_path}/python/tests/valgrind-python.supp"
if [ $# -eq 0 ]
  then
    # No arguments supplied, run all tests
    cd "${amici_path}"/python/tests
    source "${amici_path}"/build/venv/bin/activate
    command=(python -m pytest -vv --ignore-glob=*petab* -W 'ignore:Signature ')
    #                                                    ^ ignores the following warning that occurs only under valgrind,
    # e.g. `valgrind python -c "import h5py"`:
    # UserWarning: Signature b'\x00\xd0\xcc\xcc\xcc\xcc\xcc\xcc\xfb\xbf\x00\x00\x00\x00\x00\x00'
    # for <class 'numpy.longdouble'> does not match any known type: falling back to type probe function.
else
    # Run whatever was passed as arguments
    command=($@)
fi


set -x
PYTHONMALLOC=malloc valgrind \
  --suppressions="${suppressions}" \
  --show-leak-kinds=definite \
  --errors-for-leak-kinds=definite \
  --error-exitcode=1 \
  --leak-check=full \
  --gen-suppressions=all \
  -v \
  "${command[@]}"
