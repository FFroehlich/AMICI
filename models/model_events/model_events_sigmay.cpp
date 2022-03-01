
#include "amici/symbolic_functions.h"
#include "amici/defines.h" //realtype definition
typedef amici::realtype realtype;
#include <cmath> 

using namespace amici;

namespace amici {

namespace model_model_events{

void sigmay_model_events(double *sigmay, const realtype t, const realtype *p, const realtype *k, const realtype *y) {
  sigmay[0] = 1.0;
}

} // namespace model_model_events

} // namespace amici

