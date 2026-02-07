#pragma once
#include <stdexcept>
#include <string>

namespace util {
	inline void check(bool cond, const std::string& msg) {
		if (!cond) throw std::runtime_error(msg);
	}
}