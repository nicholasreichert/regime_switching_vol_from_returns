#pragma once
#include <stdexcept>
#include <string>
#include <string_view>

namespace util {
	inline void check(const bool cond, const std::string_view msg) {
		if (!cond) [[unlikely]] {
			throw std::runtime_error(std::string{msg});
		}
	}
}