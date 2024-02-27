#ifndef amici_exception_h
#define amici_exception_h

#include "amici/defines.h" // necessary for realtype

#include <array>
#include <cstdarg>
#include <exception>

namespace amici {

/**
 * @brief AMICI exception class
 *
 * Has a printf style interface to allow easy generation of error messages
 */
class AmiException : public std::exception {
  public:
    /**
     * @brief Default ctor.
     * @param first_frame Index of first frame to include
     */
    AmiException(int const first_frame = 3);

    /**
     * @brief Constructor with printf style interface
     * @param fmt error message with printf format
     * @param ... printf formatting variables
     */
    explicit AmiException(char const* fmt, ...);

    /**
     * @brief Override of default error message function
     * @return msg error message
     */
    char const* what() const noexcept override;

    /**
     * @brief Returns the stored backtrace
     * @return trace backtrace
     */
    char const* getBacktrace() const;

    /**
     * @brief Stores the current backtrace
     * @param nMaxFrames number of frames to go back in stacktrace
     * @param first_frame Index of first frame to include
     */
    void storeBacktrace(int nMaxFrames, int const first_frame);

  protected:
    /**
     * @brief Store the provided message
     * @param fmt error message with printf format
     * @param argptr pointer to variadic argument list
     */
    void storeMessage(char const* fmt, va_list argptr);

  private:
    std::array<char, 500> msg_;
    std::array<char, 500> trace_;
};

/**
 * @brief CVODE exception handler class
 */
class CvodeException : public AmiException {
  public:
    /**
     * @brief Constructor
     * @param error_code error code returned by CVODE function
     * @param function CVODE function name
     * @param extra Extra text to append to error message
     */
    CvodeException(
        int error_code, char const* function, char const* extra = nullptr
    );
};

/**
 * @brief IDA exception handler class
 */
class IDAException : public AmiException {
  public:
    /**
     * @brief Constructor
     * @param error_code error code returned by IDA function
     * @param function IDA function name
     * @param extra Extra text to append to error message
     */
    IDAException(
        int error_code, char const* function, char const* extra = nullptr
    );
};

/**
 * @brief Integration failure exception for the forward problem
 *
 * This exception should be thrown when an integration failure occurred
 * for this exception we can assume that we can recover from the exception
 * and return a solution struct to the user
 */
class IntegrationFailure : public AmiException {
  public:
    /**
     * @brief Constructor
     * @param code error code returned by cvode/ida
     * @param t time of integration failure
     */
    IntegrationFailure(int code, realtype t);

    /** error code returned by cvodes/idas */
    int error_code;

    /** time of integration failure */
    realtype time;
};

/**
 * @brief Integration failure exception for the backward problem
 *
 * This exception should be thrown when an integration failure occurred
 * for this exception we can assume that we can recover from the exception
 * and return a solution struct to the user
 */
class IntegrationFailureB : public AmiException {
  public:
    /**
     * @brief Constructor
     * @param code error code returned by cvode/ida
     * @param t time of integration failure
     */
    IntegrationFailureB(int code, realtype t);

    /** error code returned by cvode/ida */
    int error_code;

    /** time of integration failure */
    realtype time;
};

/**
 * @brief Setup failure exception
 *
 * This exception should be thrown when the solver setup failed
 * for this exception we can assume that we cannot recover from the exception
 * and an error will be thrown
 */
class SetupFailure : public AmiException {
  public:
    /**
     * @brief Constructor with printf style interface
     * @param fmt error message with printf format
     * @param ... printf formatting variables
     */
    explicit SetupFailure(char const* fmt, ...);
};

/**
 * @brief Newton failure exception
 *
 * This exception should be thrown when the steady state computation
 * failed to converge for this exception we can assume that we can
 * recover from the exception and return a solution struct to the user
 */
class NewtonFailure : public AmiException {
  public:
    /**
     * @brief Constructor, simply calls AmiException constructor
     * @param function name of the function in which the error occurred
     * @param code error code
     */
    NewtonFailure(int code, char const* function);

    /** error code returned by solver */
    int error_code;
};

} // namespace amici

#endif /* amici_exception_h */
