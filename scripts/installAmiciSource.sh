#!/bin/bash
# Create a virtual environment and perform an editable amici installation
set -e

SCRIPT_PATH=$(dirname $BASH_SOURCE)
AMICI_PATH=$(cd "$SCRIPT_PATH/.." && pwd)

# Disabled until cmake package is made compatible with updated setup.py
#make python-wheel
#pip3 install --user --prefix= `ls -t ${AMICI_PATH}/build/python/amici-*.whl | head -1`

# test install from setup.py
venv_dir="${AMICI_PATH}/venv"
set +e
mkdir -p "${venv_dir}"
python3 -m venv "${venv_dir}" --clear
# in case this fails (usually due to missing ensurepip, try getting pip
# manually
if [[ $? ]]; then
    set -e
    python3 -m venv "${venv_dir}" --clear --without-pip
    source "${venv_dir}/bin/activate"
    get_pip=${AMICI_PATH}/get-pip.py
    curl "https://bootstrap.pypa.io/get-pip.py" -o "${get_pip}"
    python3 "${get_pip}"
    rm "${get_pip}"
else
    set -e
    source "${venv_dir}/bin/activate"
fi

# set python executable for cmake
export PYTHON_EXECUTABLE="${AMICI_PATH}/venv/bin/python"

python -m pip install --upgrade pip wheel
python -m pip install --upgrade pip setuptools cmake_build_extension numpy  git+https://github.com/ICB-DCM/fiddy.git
python -m pip install git+https://github.com/FFroehlich/pysb@fix_pattern_matching # pin to PR for SPM with compartments
AMICI_BUILD_TEMP="${AMICI_PATH}/python/sdist/build/temp" \
  python -m pip install --verbose -e "${AMICI_PATH}/python/sdist[petab,test,vis]" --no-build-isolation
deactivate
