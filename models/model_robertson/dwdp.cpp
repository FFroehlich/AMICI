
#include "amici/symbolic_functions.h"
#include "amici/defines.h" //realtype definition
typedef amici::realtype realtype;
#include <cmath> 

using namespace amici;

namespace amici {

namespace model_model_robertson{

void dwdp_model_robertson(realtype *dwdp, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *w, const realtype *tcl, const realtype *stcl, const realtype *spl, const realtype *sspl, bool include_static) {
  dwdp[0] = x[1]*x[2];
}

} // namespace model_model_robertson

} // namespace amici

