
#include "amici/symbolic_functions.h"
#include "amici/defines.h" //realtype definition
typedef amici::realtype realtype;
#include <cmath> 

using namespace amici;

namespace amici {

namespace model_model_steadystate{

void dwdx_model_steadystate(realtype *dwdx, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *w, const realtype *tcl, const realtype *spl, bool include_static) {
  dwdx[0] = x[0]*2.0;
  dwdx[1] = p[3];
}

} // namespace model_model_steadystate

} // namespace amici

