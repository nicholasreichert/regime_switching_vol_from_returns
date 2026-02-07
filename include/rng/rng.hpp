#pragma once
#include <random>
#include <vector>
#include <numeric>
#include <cstdint>

namespace rng {

	struct Rng {
		std::mt19937_64 gen; // mersenne twister gen 64-bit ints
		explicit Rng(const uint64_t seed) : gen(seed) {}

		// continuous shocks
		[[nodiscard]] double normal01() {
			static thread_local std::normal_distribution<double> nd(0.0, 1.0);
			return nd(gen);
		}
		
		// discrete state jumps
		[[nodiscard]] int categorical(const std::vector<double>& p) {
			std::discrete_distribution<int> dd(p.begin(), p.end());
			return dd(gen);
		}
	};
}