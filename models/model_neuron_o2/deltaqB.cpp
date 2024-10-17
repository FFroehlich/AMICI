
#include "amici/symbolic_functions.h"
#include "amici/defines.h" //realtype definition
typedef amici::realtype realtype;
#include <cmath> 

using namespace amici;

namespace amici {

namespace model_model_neuron_o2{

void deltaqB_model_neuron_o2(double *deltaqB, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const int ip, const int ie, const realtype *xdot, const realtype *xdot_old, const realtype *xB, const realtype *xBdot) {
switch (ip) {
  case 0: {
              switch(ie) { 
              case 0: {
  deltaqB[0] = (x[2]*xB[3]*(p[3]+p[1]*p[2]+p[1]*x[0]))/(k[1]+x[0]*5.0-x[1]+(x[0]*x[0])*(1.0/2.5E1)+1.4E2)+(x[4]*xB[5]*(p[3]+p[1]*p[2]+p[1]*x[0]))/(k[1]+x[0]*5.0-x[1]+(x[0]*x[0])*(1.0/2.5E1)+1.4E2)+(x[6]*xB[7]*(p[3]+p[1]*p[2]+p[1]*x[0]))/(k[1]+x[0]*5.0-x[1]+(x[0]*x[0])*(1.0/2.5E1)+1.4E2)+(x[8]*xB[9]*(p[3]+p[1]*p[2]+p[1]*x[0]))/(k[1]+x[0]*5.0-x[1]+(x[0]*x[0])*(1.0/2.5E1)+1.4E2);

              } break;

              } 

  } break;

  case 1: {
              switch(ie) { 
              case 0: {
  deltaqB[0] = (x[2]*xB[3]*(p[0]*p[2]+p[0]*x[0]))/(k[1]+x[0]*5.0-x[1]+(x[0]*x[0])*(1.0/2.5E1)+1.4E2)+(x[4]*xB[5]*(p[0]*p[2]+p[0]*x[0]))/(k[1]+x[0]*5.0-x[1]+(x[0]*x[0])*(1.0/2.5E1)+1.4E2)+(x[6]*xB[7]*(p[0]*p[2]+p[0]*x[0]))/(k[1]+x[0]*5.0-x[1]+(x[0]*x[0])*(1.0/2.5E1)+1.4E2)+(x[8]*xB[9]*(p[0]*p[2]+p[0]*x[0]))/(k[1]+x[0]*5.0-x[1]+(x[0]*x[0])*(1.0/2.5E1)+1.4E2);

              } break;

              } 

  } break;

  case 2: {
              switch(ie) { 
              case 0: {
  deltaqB[0] = xB[0]-(x[2]*xB[2]*(p[2]*(2.0/2.5E1)-5.0))/(k[1]+x[0]*5.0-x[1]+(x[0]*x[0])*(1.0/2.5E1)+1.4E2)-(x[4]*xB[4]*(p[2]*(2.0/2.5E1)-5.0))/(k[1]+x[0]*5.0-x[1]+(x[0]*x[0])*(1.0/2.5E1)+1.4E2)-(x[6]*xB[6]*(p[2]*(2.0/2.5E1)-5.0))/(k[1]+x[0]*5.0-x[1]+(x[0]*x[0])*(1.0/2.5E1)+1.4E2)-(x[8]*xB[8]*(p[2]*(2.0/2.5E1)-5.0))/(k[1]+x[0]*5.0-x[1]+(x[0]*x[0])*(1.0/2.5E1)+1.4E2)+(p[0]*p[1]*x[2]*xB[3])/(k[1]+x[0]*5.0-x[1]+(x[0]*x[0])*(1.0/2.5E1)+1.4E2)+(p[0]*p[1]*x[4]*xB[5])/(k[1]+x[0]*5.0-x[1]+(x[0]*x[0])*(1.0/2.5E1)+1.4E2)+(p[0]*p[1]*x[6]*xB[7])/(k[1]+x[0]*5.0-x[1]+(x[0]*x[0])*(1.0/2.5E1)+1.4E2)+(p[0]*p[1]*x[8]*xB[9])/(k[1]+x[0]*5.0-x[1]+(x[0]*x[0])*(1.0/2.5E1)+1.4E2);

              } break;

              } 

  } break;

  case 3: {
              switch(ie) { 
              case 0: {
  deltaqB[0] = -xB[1]+(x[2]*xB[2])/(k[1]+x[0]*5.0-x[1]+(x[0]*x[0])*(1.0/2.5E1)+1.4E2)+(x[4]*xB[4])/(k[1]+x[0]*5.0-x[1]+(x[0]*x[0])*(1.0/2.5E1)+1.4E2)+(x[6]*xB[6])/(k[1]+x[0]*5.0-x[1]+(x[0]*x[0])*(1.0/2.5E1)+1.4E2)+(x[8]*xB[8])/(k[1]+x[0]*5.0-x[1]+(x[0]*x[0])*(1.0/2.5E1)+1.4E2)+(p[0]*x[2]*xB[3])/(k[1]+x[0]*5.0-x[1]+(x[0]*x[0])*(1.0/2.5E1)+1.4E2)+(p[0]*x[4]*xB[5])/(k[1]+x[0]*5.0-x[1]+(x[0]*x[0])*(1.0/2.5E1)+1.4E2)+(p[0]*x[6]*xB[7])/(k[1]+x[0]*5.0-x[1]+(x[0]*x[0])*(1.0/2.5E1)+1.4E2)+(p[0]*x[8]*xB[9])/(k[1]+x[0]*5.0-x[1]+(x[0]*x[0])*(1.0/2.5E1)+1.4E2);

              } break;

              } 

  } break;

}
}

} // namespace model_model_neuron_o2

} // namespace amici

