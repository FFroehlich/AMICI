#!/bin/bash
# Run Python test suite inside virtual environment
# Usage: ./run-python-tests.sh [additional pytest arguments]

script_path=$(dirname "${BASH_SOURCE[0]}")
amici_path=$(cd "$script_path"/.. && pwd)

set -e

if [[ -z "${BNGPATH}" ]]; then
    export BNGPATH=${amici_path}/ThirdParty/BioNetGen-2.7.0
fi

cd "${amici_path}"/python/tests
source "${amici_path}"/build/venv/bin/activate

# PEtab tests are run separately
pytest \
  --ignore-glob=*petab* \
  --ignore-glob=*test_splines.py \
  --durations=10 \
  $@
