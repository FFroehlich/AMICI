#!/bin/bash
AMICI_PATH="`dirname \"$0\"`"
AMICI_PATH="`( cd \"$AMICI_PATH/..\" && pwd )`"

cd ${AMICI_PATH}/tests/cpputest/build

ctest -V

cd ${AMICI_PATH}
