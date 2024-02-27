
#include "amici/symbolic_functions.h"
#include "amici/defines.h" //realtype definition
typedef amici::realtype realtype;
#include <cmath> 

using namespace amici;

namespace amici {

namespace model_model_neuron_o2{

void w_model_neuron_o2(realtype *w, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *tcl, const realtype *spl, bool include_static) {
  w[0] = x[0]*(2.0/2.5E1);
  w[1] = w[0]+5.0;
}

} // namespace model_model_neuron_o2

} // namespace amici

