#ifndef AMICI_LOGGER_H
#define AMICI_LOGGER_H

#include <string>
#include <vector>

namespace amici {

struct LogItem;

/**
 * @brief Severity levels for logging.
 */
enum class LogSeverity {
    error,
    warning,
    debug,
};

/**
 * @brief A logger, holding a list of error messages.
 */
class Logger {
  public:
    Logger() = default;
    /**
     * @brief Add a log entry
     * @param severity Severity level
     * @param identifier Short identifier for the logged event
     * @param message A more detailed message
     */
    void
    log(LogSeverity severity, std::string const& identifier,
        std::string const& message);

#if SWIG_VERSION >= 0x040002
    /**
     * @brief Add a log entry with printf-like message formatting
     * @param severity Severity level
     * @param identifier Short identifier for the logged event
     * @param format printf format string
     * @param ... arguments to be formatted
     */
#else
    // swig 4.0.1 segfaults on "@param ..."
    // see https://github.com/swig/swig/issues/1643
    /**
     * @brief Add a log entry with printf-like message formatting
     * @param severity Severity level
     * @param identifier Short identifier for the logged event
     * @param format printf format string
     */
#endif
    void
    log(LogSeverity severity, std::string const& identifier, char const* format,
        ...);

    /** The log items */
    std::vector<LogItem> items;
};

/**
 * @brief A log item.
 */
struct LogItem {
    /**
     * @brief Default ctor.
     */
    LogItem() = default;

    /**
     * @brief Construct a LogItem
     * @param severity
     * @param identifier
     * @param message
     */
    LogItem(
        LogSeverity severity, std::string const& identifier,
        std::string const& message
    )
        : severity(severity)
        , identifier(identifier)
        , message(message){};

    /** Severity level */
    LogSeverity severity = LogSeverity::error;

    /** Short identifier for the logged event */
    std::string identifier;

    /** A more detailed and readable message */
    std::string message;
};

} // namespace amici
#endif // AMICI_LOGGER_H
