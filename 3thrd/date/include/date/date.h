#pragma once

#include <chrono>
#include <ctime>
#include <iomanip>
#include <ostream>
#include <type_traits>

namespace date {

namespace detail {

inline std::ostream& write_time(std::ostream& os, std::time_t value) {
  std::tm tm_value{};
#if defined(_WIN32)
  gmtime_s(&tm_value, &value);
#else
  gmtime_r(&value, &tm_value);
#endif
  return os << std::put_time(&tm_value, "%F %T");
}

inline std::wostream& write_time(std::wostream& os, std::time_t value) {
  std::tm tm_value{};
#if defined(_WIN32)
  gmtime_s(&tm_value, &value);
#else
  gmtime_r(&value, &tm_value);
#endif
  return os << std::put_time(&tm_value, L"%F %T");
}

}  // namespace detail

template <typename Clock, typename Duration,
          typename = std::enable_if_t<std::is_same_v<Clock, std::chrono::system_clock>>>
inline std::ostream& operator<<(std::ostream& os,
                                const std::chrono::time_point<Clock, Duration>& tp) {
  return detail::write_time(os, Clock::to_time_t(
                                    std::chrono::time_point_cast<std::chrono::system_clock::duration>(tp)));
}

template <typename Clock, typename Duration,
          typename = std::enable_if_t<std::is_same_v<Clock, std::chrono::system_clock>>>
inline std::wostream& operator<<(std::wostream& os,
                                 const std::chrono::time_point<Clock, Duration>& tp) {
  return detail::write_time(os, Clock::to_time_t(
                                    std::chrono::time_point_cast<std::chrono::system_clock::duration>(tp)));
}

}  // namespace date
