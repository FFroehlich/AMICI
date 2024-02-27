#include "amici/exception.h"
#include "amici/misc.h"

#include <cstdarg>
#include <cstdio>

namespace amici {

AmiException::AmiException(int const first_frame) {
    storeBacktrace(12, first_frame);
}

AmiException::AmiException(char const* fmt, ...)
    : AmiException(4) {
    va_list ap;
    va_start(ap, fmt);
    storeMessage(fmt, ap);
    va_end(ap);
}

char const* AmiException::what() const noexcept { return msg_.data(); }

char const* AmiException::getBacktrace() const { return trace_.data(); }

void AmiException::storeBacktrace(int const nMaxFrames, int const first_frame) {
    snprintf(
        trace_.data(), trace_.size(), "%s",
        backtraceString(nMaxFrames, first_frame).c_str()
    );
}

void AmiException::storeMessage(char const* fmt, va_list argptr) {
    vsnprintf(msg_.data(), msg_.size(), fmt, argptr);
}

CvodeException::CvodeException(
    int const error_code, char const* function, char const* extra
)
    : AmiException(
        "CVODE routine %s failed with error code %i. %s", function, error_code,
        extra ? extra : ""
    ) {}

IDAException::IDAException(
    int const error_code, char const* function, char const* extra
)
    : AmiException(
        "IDA routine %s failed with error code %i. %s", function, error_code,
        extra ? extra : ""
    ) {}

IntegrationFailure::IntegrationFailure(int code, realtype t)
    : AmiException("AMICI failed to integrate the forward problem")
    , error_code(code)
    , time(t) {}

IntegrationFailureB::IntegrationFailureB(int code, realtype t)
    : AmiException("AMICI failed to integrate the backward problem")
    , error_code(code)
    , time(t) {}

NewtonFailure::NewtonFailure(int code, char const* function)
    : AmiException(
        "NewtonSolver routine %s failed with error code %i", function, code
    ) {
    error_code = code;
}

SetupFailure::SetupFailure(char const* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    storeMessage(fmt, ap);
    va_end(ap);
}

} // namespace amici
