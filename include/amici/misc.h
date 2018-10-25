#ifndef AMICI_MISC_H
#define AMICI_MISC_H

#include "amici/defines.h"

#include <algorithm>
#include <vector>


namespace amici {

int checkFinite(const int N,const realtype *array, const char* fun);


/**
  * @brief Remove parameter scaling according to the parameter scaling in pscale
  *
  * @param[in] bufferScaled scaled parameters
  * @param[in] pscale parameter scaling
  * @param[in] n number of elements in bufferScaled, pscale and bufferUnscaled
  * @param[out] bufferUnscaled unscaled parameters are written to the array
  *
  * @return status flag indicating success of execution @type int
  */
void unscaleParameters(const double *bufferScaled,
                       const ParameterScaling *pscale,
                       int n,
                       double *bufferUnscaled);

/**
  * @brief Remove parameter scaling according to the parameter scaling in pscale.
  *
  * All vectors must be of same length
  *
  * @param[in] bufferScaled scaled parameters
  * @param[in] pscale parameter scaling
  * @param[out] bufferUnscaled unscaled parameters are written to the array
  *
  * @return status flag indicating success of execution @type int
  */
void unscaleParameters(std::vector<double> const& bufferScaled,
                       std::vector<ParameterScaling> const& pscale,
                       std::vector<double> & bufferUnscaled);

/**
  * @brief Remove parameter scaling according to `scaling`
  *
  * @param scaledParameter scaled parameter
  * @param scaling parameter scaling
  *
  * @return Unscaled parameter
  */
double getUnscaledParameter(double scaledParameter, ParameterScaling scaling);

} // namespace amici
#endif // AMICI_MISC_H
