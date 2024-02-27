
#include "amici/symbolic_functions.h"
#include "amici/defines.h" //realtype definition
typedef amici::realtype realtype;
#include <cmath> 

using namespace amici;

namespace amici {

namespace model_model_steadystate{

void w_model_steadystate(realtype *w, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *tcl, const realtype *spl, bool include_static) {
  w[0] = p[3]*x[2];
  w[1] = x[0]*x[0];
}

} // namespace model_model_steadystate

} // namespace amici

