#!/bin/bash
#
# Build CppUTest
#
set -e

SCRIPT_PATH=$(dirname $BASH_SOURCE)
AMICI_PATH=$(cd $SCRIPT_PATH/.. && pwd)

# Cpputest
mkdir -p ${AMICI_PATH}/ThirdParty
cd ${AMICI_PATH}/ThirdParty

if [ ! -d "BioNetGen-2.3.2" ]; then
    if [ ! -e "bionetgen.tar.gz" ]; then
        if [[ "$OSTYPE" == "linux-gnu" ]]; then
            wget -O bionetgen.tar.gz https://bintray.com/jczech/bionetgen/download_file?file_path=BioNetGen-2.3.2-linux.tar.gz
        elif [[ "$OSTYPE" == "darwin"* ]]; then
            wget -O bionetgen.tar.gz https://bintray.com/jczech/bionetgen/download_file?file_path=BioNetGen-2.3.2-osx.tar.gz
        fi
    fi
    tar -xvzf bionetgen.tar.gz
fi
