#pragma once
#include <vector>
#include <cmath>
#include <numeric>
#include "../util/check.hpp"

namespace hmm {

	// represent parameters (pi, matrix P, values of volatility (sigma)
	struct Params {
		int K; // finite state space
		std::vector<double> pi;
		std::vector<std::vector<double>> P; // vector of vectors to represent K x K transition matrix
		std::vector<double> sigma;

		// check that pi and each row of P sum to 1
		inline void validate() const {
			util::check(static_cast<int>(pi.size()) == K, "pi wrong size");
			util::check(static_cast<int>(P.size()) == K, "P wrong rows");
			util::check(static_cast<int>(sigma.size()) == K, "sigma wrong size");
			for (int i{0}; i < K; ++i) {
				util::check(static_cast<int>(P[i].size()) == K, "P wrong cols");
			}
			for (const double s : sigma) {
				util::check(s > 0.0, "sigma must be > 0");
			}
		}

		// prevents probs from becoming "true zero" which would break log-space math
		static inline void normalize_probs(std::vector<double>& v, const double floor = 1e-12) {
			// single-pass optimization: compute sum and apply floor simultaneously
			double sum{0.0};
			for (double& x : v) {
				x = std::max(x, floor);
				sum += x;
			}
			if (sum > 0.0) [[likely]] {
				const double inv_sum{1.0 / sum};  // single division, then multiply
				for (double& x : v) {
					x *= inv_sum;
				}
			}
		}

		// ensures volatility sigma_k is +, otherwise density will be undefined
		inline void normalize(const double floor = 1e-12) {
			normalize_probs(pi, floor);
			for (int i{0}; i < K; ++i) {
				normalize_probs(P[i], floor);
			}
			constexpr double sigma_floor{1e-8};
			for (double& s : sigma) {
				s = std::max(s, sigma_floor);
			}
		}
	};
}