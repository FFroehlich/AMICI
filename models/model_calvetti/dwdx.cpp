
#include "amici/symbolic_functions.h"
#include "amici/defines.h" //realtype definition
typedef amici::realtype realtype;
#include <cmath> 

using namespace amici;

namespace amici {

namespace model_model_calvetti{

void dwdx_model_calvetti(realtype *dwdx, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *w, const realtype *tcl, const realtype *spl, bool include_static) {
  dwdx[0] = 1.0/(x[0]*x[0]*x[0])*-2.0;
  dwdx[1] = k[1]*w[15]*dwdx[0];
  dwdx[2] = dwdx[1];
  dwdx[3] = x[3]*dwdx[2];
  dwdx[4] = dwdx[3];
  dwdx[5] = -1.0/(w[20]*w[20])*dwdx[4];
  dwdx[6] = w[0]*w[14]*w[21]*(2.0/3.1E1)+w[0]*x[0]*w[14]*dwdx[5]*(2.0/3.1E1);
  dwdx[7] = x[0]*2.0;
  dwdx[8] = 1.0/(x[1]*x[1]*x[1])*-2.0;
  dwdx[9] = k[3]*w[1]*dwdx[8];
  dwdx[10] = dwdx[9];
  dwdx[11] = x[4]*dwdx[10];
  dwdx[12] = dwdx[9];
  dwdx[13] = x[3]*dwdx[12];
  dwdx[14] = dwdx[11]+dwdx[13];
  dwdx[15] = -1.0/(w[20]*w[20])*dwdx[14];
  dwdx[16] = w[0]*x[0]*w[14]*dwdx[15]*(2.0/3.1E1);
  dwdx[17] = dwdx[11];
  dwdx[18] = -1.0/(w[26]*w[26])*dwdx[17];
  dwdx[19] = w[6]*w[27]*w[29]*(2.0/1.63E2)+x[1]*w[6]*w[29]*dwdx[18]*(2.0/1.63E2);
  dwdx[20] = 1.0/(x[2]*x[2]*x[2])*-2.0;
  dwdx[21] = k[5]*w[4]*dwdx[20];
  dwdx[22] = dwdx[21];
  dwdx[23] = x[4]*dwdx[22];
  dwdx[24] = k[5]*w[4]*x[5]*dwdx[20];
  dwdx[25] = x[2]*2.0;
  dwdx[26] = dwdx[23]+dwdx[24];
  dwdx[27] = -1.0/(w[20]*w[20])*dwdx[26];
  dwdx[28] = w[0]*x[0]*w[14]*dwdx[27]*(2.0/3.1E1);
  dwdx[29] = dwdx[23]+dwdx[24];
  dwdx[30] = -1.0/(w[26]*w[26])*dwdx[29];
  dwdx[31] = x[1]*w[6]*w[29]*dwdx[30]*(2.0/1.63E2);
  dwdx[32] = w[11]*w[32]*w[33]*w[34]*w[36]*(1.0/6.1E1)+x[2]*w[32]*w[33]*w[34]*w[36]*dwdx[25]*(1.0/6.1E1);
  dwdx[33] = w[18];
  dwdx[34] = dwdx[33];
  dwdx[35] = -1.0/(w[20]*w[20])*dwdx[34];
  dwdx[36] = w[0]*x[0]*w[14]*dwdx[35]*(2.0/3.1E1);
  dwdx[37] = w[8];
  dwdx[38] = dwdx[37];
  dwdx[39] = -1.0/(w[20]*w[20])*dwdx[38];
  dwdx[40] = w[0]*x[0]*w[14]*dwdx[39]*(2.0/3.1E1);
  dwdx[41] = dwdx[37];
  dwdx[42] = -1.0/(w[26]*w[26])*dwdx[41];
  dwdx[43] = x[1]*w[6]*w[29]*dwdx[42]*(2.0/1.63E2);
  dwdx[44] = k[5]*w[4]*w[5];
  dwdx[45] = dwdx[44];
  dwdx[46] = -1.0/(w[20]*w[20])*dwdx[45];
  dwdx[47] = w[0]*x[0]*w[14]*dwdx[46]*(2.0/3.1E1);
  dwdx[48] = dwdx[44];
  dwdx[49] = -1.0/(w[26]*w[26])*dwdx[48];
  dwdx[50] = x[1]*w[6]*w[29]*dwdx[49]*(2.0/1.63E2);
  dwdx[51] = -1.0/(x[5]*x[5]);
  dwdx[52] = x[2]*w[11]*w[32]*w[33]*w[36]*dwdx[51]*(1.0/6.1E1);
}

} // namespace model_model_calvetti

} // namespace amici

