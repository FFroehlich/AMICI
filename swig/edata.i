%module edata

// Add necessary symbols to generated header
%{
#include "edata.h"
using namespace amici;
%}

// Process symbols in header
%include "edata.h"