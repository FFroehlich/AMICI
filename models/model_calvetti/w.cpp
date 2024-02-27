
#include "amici/symbolic_functions.h"
#include "amici/defines.h" //realtype definition
typedef amici::realtype realtype;
#include <cmath> 

using namespace amici;

namespace amici {

namespace model_model_calvetti{

void w_model_calvetti(realtype *w, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *tcl, const realtype *spl, bool include_static) {
  w[0] = 1.0/k[0];
  w[1] = k[2]*k[2];
  w[2] = 1.0/(x[1]*x[1]);
  w[3] = k[3]*w[1]*w[2];
  w[4] = k[4]*k[4];
  w[5] = 1.0/(x[2]*x[2]);
  w[6] = 1.0/k[2];
  w[7] = k[5]*w[4]*w[5];
  w[8] = w[3]+w[7];
  w[9] = x[4]*w[8];
  w[10] = k[5]*w[4]*w[5]*x[5];
  w[11] = x[2]*x[2];
  w[12] = h[1]*(1.0/3.1E1);
  w[13] = k[1]*(1.0/2.0);
  w[14] = w[13]-1.0;
  w[15] = k[0]*k[0];
  w[16] = 1.0/(x[0]*x[0]);
  w[17] = k[1]*w[15]*w[16];
  w[18] = w[3]+w[17];
  w[19] = x[3]*w[18];
  w[20] = w[9]+w[10]+w[19];
  w[21] = 1.0/w[20];
  w[22] = w[0]*x[0]*w[14]*w[21]*(2.0/3.1E1);
  w[23] = 1.0/(k[0]*k[0]);
  w[24] = 1.0/k[1];
  w[25] = x[0]*x[0];
  w[26] = w[9]+w[10];
  w[27] = 1.0/w[26];
  w[28] = k[3]*(1.0/2.0);
  w[29] = k[1]+w[28]-1.0;
  w[30] = x[1]*w[6]*w[27]*w[29]*(2.0/1.63E2);
  w[31] = 1.0/k[4];
  w[32] = 1.0/(k[4]*k[4]*k[4]);
  w[33] = 1.0/k[5];
  w[34] = 1.0/x[5];
  w[35] = k[5]*(1.0/2.0);
  w[36] = k[1]+k[3]+w[35]-1.0;
  w[37] = x[2]*w[11]*w[32]*w[33]*w[34]*w[36]*(1.0/6.1E1);
}

} // namespace model_model_calvetti

} // namespace amici

