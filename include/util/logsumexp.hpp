#pragma once
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>

namespace util {

	[[nodiscard]] inline double logsumexp(const std::vector<double>& v) {
		// using identity log(sum(e^x_i)) = m + log(sum(e^(x_i-m)), where m is the max value in the set
		if (v.empty()) [[unlikely]] {
			return -std::numeric_limits<double>::infinity();
		}
		const double m{*std::max_element(v.begin(), v.end())};
		if (!std::isfinite(m)) [[unlikely]] {
			return m;
		}
		double s{0.0};
		for (const double x : v) {
			s += std::exp(x - m);
		}
		return m + std::log(s);
	}

	[[nodiscard]] inline double logaddexp(const double a, const double b) {
		// calculates log(e^a + e^b)
		if (!std::isfinite(a)) [[unlikely]] return b;
		if (!std::isfinite(b)) [[unlikely]] return a;
		const double m{std::max(a, b)};
		return m + std::log(std::exp(a - m) + std::exp(b - m));
	}
}