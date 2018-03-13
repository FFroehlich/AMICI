%module model_neuron_o2
%import ../../swig/amici.i
// Add necessary symbols to generated header

%{
#include "wrapfunctions.h"
#include "amici/model_ode.h"
#include "amici/model_dae.h"
using namespace amici;
%}

%include ../swig/std_unique_ptr.i
wrap_unique_ptr(ModelPtr, amici::Model)


// Process symbols in header
%include "wrapfunctions.h"
