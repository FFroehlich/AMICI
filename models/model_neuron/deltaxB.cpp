
#include "amici/symbolic_functions.h"
#include "amici/defines.h" //realtype definition
typedef amici::realtype realtype;
#include <cmath> 

using namespace amici;

namespace amici {

namespace model_model_neuron{

void deltaxB_model_neuron(double *deltaxB, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const int ie, const realtype *xdot, const realtype *xdot_old, const realtype *xB, const realtype *xBdot, const realtype *tcl) {
              switch(ie) { 
              case 0: {
  deltaxB[0] = xB[0];

              } break;

              } 
}

} // namespace model_model_neuron

} // namespace amici

