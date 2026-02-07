#pragma once
#include <vector>
#include "params.hpp"

namespace hmm {

	struct FBResult {
		double loglik;
		std::vector<std::vector<double>> log_alpha; // T x K
		std::vector<std::vector<double>> log_beta; // T x K
		std::vector<std::vector<double>> gamma; // T x K
		std::vector<std::vector<std::vector<double>>> xi; // (T-1) x K x K
	};

	FBResult forward_backward_normal(const Params& params, const std::vector<double>& r);
}