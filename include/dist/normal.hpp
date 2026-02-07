#pragma once
#include <cmath>
#include <numbers>

namespace dist {

	struct Normal {
		// log N(0, sigma^2) at x
		[[nodiscard]] static constexpr double logpdf(const double x, const double sigma) noexcept {
			const double s2{sigma * sigma};
			const double log_norm{-0.5 * std::log(2.0 * std::numbers::pi * s2)};
			return log_norm - 0.5 * (x * x) / s2;
		}
	};
}