#ifndef AMICI_MISC_H
#define AMICI_MISC_H

#include "amici/defines.h"
#include "amici/exception.h"
#include "amici/vector.h"
#include <sunmatrix/sunmatrix_sparse.h> // SUNMatrixContent_Sparse

#include <algorithm>
#include <ctime>
#include <functional>
#include <regex>
#include <vector>

#ifdef HAS_BOOST_CHRONO
#include <boost/chrono/thread_clock.hpp>
#endif

#include <gsl/gsl-lite.hpp>

namespace amici {

/**
 * @brief creates a slice from existing data
 *
 * @param data to be sliced
 * @param index slice index
 * @param size slice size
 * @return span of the slice
 */

template <class T>
gsl::span<T> slice(std::vector<T>& data, int index, unsigned size) {
    if ((index + 1) * size > data.size())
        throw std::out_of_range("requested slice is out of data range");
    if (size > 0)
        return gsl::make_span(&data.at(index * size), size);

    return gsl::make_span(static_cast<T*>(nullptr), 0);
}

/**
 * @brief creates a constant slice from existing constant data
 *
 * @param data to be sliced
 * @param index slice index
 * @param size slice size
 * @return span of the slice
 */

template <class T>
gsl::span<T const> slice(std::vector<T> const& data, int index, unsigned size) {
    if ((index + 1) * size > data.size())
        throw std::out_of_range("requested slice is out of data range");
    if (size > 0)
        return gsl::make_span(&data.at(index * size), size);

    return gsl::make_span(static_cast<T*>(nullptr), 0);
}

/**
 * @brief local helper to check whether the provided buffer has the expected
 * size
 * @param buffer buffer to which values are to be written
 * @param expected_size expected size of the buffer
 */
template <class T>
void checkBufferSize(
    gsl::span<T> buffer, typename gsl::span<T>::index_type expected_size
) {
    if (buffer.size() != expected_size)
        throw AmiException(
            "Incorrect buffer size! Was %u, expected %u.", buffer.size(),
            expected_size
        );
}

/* TODO: templating writeSlice breaks implicit conversion between vector & span
 not sure whether this is fixable */

/**
 * @brief local helper function to write computed slice to provided buffer
 * (span)
 * @param slice computed value
 * @param buffer buffer to which values are to be written
 */
template <class T>
void writeSlice(gsl::span<T const> const slice, gsl::span<T> buffer) {
    checkBufferSize(buffer, slice.size());
    std::copy(slice.begin(), slice.end(), buffer.data());
};

/**
 * @brief local helper function to add the computed slice to provided buffer
 * (span)
 * @param slice computed value
 * @param buffer buffer to which values are to be added
 */
template <class T>
void addSlice(gsl::span<T const> const slice, gsl::span<T> buffer) {
    checkBufferSize(buffer, slice.size());
    std::transform(
        slice.begin(), slice.end(), buffer.begin(), buffer.begin(),
        std::plus<T>()
    );
};

/**
 * @brief local helper function to write computed slice to provided buffer
 * (vector)
 * @param s computed value
 * @param b buffer to which values are to be written
 */
template <class T> void writeSlice(std::vector<T> const& s, std::vector<T>& b) {
    writeSlice(
        gsl::make_span(s.data(), s.size()), gsl::make_span(b.data(), b.size())
    );
};

/**
 * @brief local helper function to write computed slice to provided buffer
 * (vector/span)
 * @param s computed value
 * @param b buffer to which values are to be written
 */
template <class T> void writeSlice(std::vector<T> const& s, gsl::span<T> b) {
    writeSlice(gsl::make_span(s.data(), s.size()), b);
};

/**
 * @brief local helper function to add the computed slice to provided buffer
 * (vector/span)
 * @param s computed value
 * @param b buffer to which values are to be written
 */
template <class T> void addSlice(std::vector<T> const& s, gsl::span<T> b) {
    addSlice(gsl::make_span(s.data(), s.size()), b);
};

/**
 * @brief local helper function to write computed slice to provided buffer
 * (AmiVector/span)
 * @param s computed value
 * @param b buffer to which values are to be written
 */
void writeSlice(AmiVector const& s, gsl::span<realtype> b);

/**
 * @brief Remove parameter scaling according to the parameter scaling in pscale
 *
 * All vectors must be of same length.
 *
 * @param bufferScaled scaled parameters
 * @param pscale parameter scaling
 * @param bufferUnscaled unscaled parameters are written to the array
 */
void unscaleParameters(
    gsl::span<realtype const> bufferScaled,
    gsl::span<ParameterScaling const> pscale, gsl::span<realtype> bufferUnscaled
);

/**
 * @brief Remove parameter scaling according to `scaling`
 *
 * @param scaledParameter scaled parameter
 * @param scaling parameter scaling
 *
 * @return Unscaled parameter
 */
double getUnscaledParameter(double scaledParameter, ParameterScaling scaling);

/**
 * @brief Apply parameter scaling according to `scaling`
 * @param unscaledParameter
 * @param scaling parameter scaling
 * @return Scaled parameter
 */
double getScaledParameter(double unscaledParameter, ParameterScaling scaling);

/**
 * @brief Apply parameter scaling according to `scaling`
 * @param bufferUnscaled
 * @param pscale parameter scaling
 * @param bufferScaled destination
 */
void scaleParameters(
    gsl::span<realtype const> bufferUnscaled,
    gsl::span<ParameterScaling const> pscale, gsl::span<realtype> bufferScaled
);

/**
 * @brief Returns the current backtrace as std::string
 * @param maxFrames Number of frames to include
 * @param first_frame Index of first frame to include
 * @return Backtrace
 */
std::string backtraceString(int maxFrames, int const first_frame = 0);

/**
 * @brief Convert std::regex_constants::error_type to string
 * @param err_type error type
 * @return Error type as string
 */
std::string regexErrorToString(std::regex_constants::error_type err_type);

/**
 * @brief Format printf-style arguments to std::string
 * @param fmt Format string
 * @param ap Argument list pointer
 * @return Formatted String
 */
std::string printfToString(char const* fmt, va_list ap);

/**
 * @brief Generic implementation for a context manager, explicitly deletes copy
 * and move operators for derived classes
 */
class ContextManager {
  public:
    ContextManager() = default;
    ContextManager(ContextManager& other) = delete;
    ContextManager(ContextManager&& other) = delete;
};

/**
 * @brief Convert a flat index to a pair of row/column indices,
 * assuming row-major order.
 * @param flat_idx flat index
 * @param num_cols number of columns of referred to matrix
 * @return row index, column index
 */
auto unravel_index(size_t flat_idx, size_t num_cols)
    -> std::pair<size_t, size_t>;

/**
 * @brief Check if two spans are equal, treating NaNs in the same position as
 * equal.
 * @param a
 * @param b
 * @return Whether the contents of the two spans are equal.
 */
template <class T> bool is_equal(T const& a, T const& b) {
    if (a.size() != b.size())
        return false;

    auto a_data = a.data();
    auto b_data = b.data();
    for (typename T::size_type i = 0; i < a.size(); ++i) {
        if (a_data[i] != b_data[i]
            && !(std::isnan(a_data[i]) && std::isnan(b_data[i])))
            return false;
    }
    return true;
}

#ifdef BOOST_CHRONO_HAS_THREAD_CLOCK
/** Tracks elapsed CPU time using boost::chrono::thread_clock. */
class CpuTimer {
    using clock = boost::chrono::thread_clock;
    using time_point = clock::time_point;
    using d_seconds = boost::chrono::duration<double>;
    using d_milliseconds = boost::chrono::duration<double, boost::milli>;

  public:
    /**
     * @brief Constructor
     */
    CpuTimer()
        : start_(clock::now()) {}

    /**
     * @brief Reset the timer
     */
    void reset() { start_ = clock::now(); }

    /**
     * @brief Get elapsed CPU time in seconds since initialization or last reset
     * @return CPU time in seconds
     */
    double elapsed_seconds() const {
        return d_seconds(clock::now() - start_).count();
    }

    /**
     * @brief Get elapsed CPU time in milliseconds since initialization or last
     * reset
     * @return CPU time in milliseconds
     */
    double elapsed_milliseconds() const {
        return d_milliseconds(clock::now() - start_).count();
    }

    /**
     * @brief Whether the timer uses a thread clock (i.e. provides proper,
     * thread-specific CPU time).
     */
    static bool const uses_thread_clock = true;

  private:
    /** Start time */
    time_point start_;
};
#else
/** Tracks elapsed CPU time using std::clock. */
class CpuTimer {
  public:
    /**
     * @brief Constructor
     */
    CpuTimer()
        : start_(std::clock()) {}

    /**
     * @brief Reset the timer
     */
    void reset() { start_ = std::clock(); }

    /**
     * @brief Get elapsed CPU time in seconds since initialization or last reset
     * @return CPU time in seconds
     */
    double elapsed_seconds() const {
        return static_cast<double>(std::clock() - start_) / CLOCKS_PER_SEC;
    }

    /**
     * @brief Get elapsed CPU time in milliseconds since initialization or last
     * reset
     * @return CPU time in milliseconds
     */
    double elapsed_milliseconds() const {
        return static_cast<double>(std::clock() - start_) * 1000.0
               / CLOCKS_PER_SEC;
    }

    /**
     * @brief Whether the timer uses a thread clock (i.e. provides proper,
     * thread-specific CPU time).
     */
    static bool const uses_thread_clock = false;

  private:
    /** Start time */
    std::clock_t start_;
};
#endif

} // namespace amici

#endif // AMICI_MISC_H
