
#include "amici/symbolic_functions.h"
#include "amici/defines.h" //realtype definition
typedef amici::realtype realtype;
#include <cmath> 

using namespace amici;

namespace amici {

namespace model_model_neuron{

void deltaqB_model_neuron(double *deltaqB, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const int ip, const int ie, const realtype *xdot, const realtype *xdot_old, const realtype *xB, const realtype *xBdot) {
switch (ip) {
  case 2: {
              switch(ie) { 
              case 0: {
  deltaqB[0] = xB[0];

              } break;

              } 

  } break;

  case 3: {
              switch(ie) { 
              case 0: {
  deltaqB[0] = -xB[1];

              } break;

              } 

  } break;

}
}

} // namespace model_model_neuron

} // namespace amici

