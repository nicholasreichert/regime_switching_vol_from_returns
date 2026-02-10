#pragma once
#include <vector>
#include "params.hpp"

namespace hmm {

struct EMOptions {
	int max_iters{200};
	double tol{1e-6};
	double prob_floor{1e-12};
	double sigma_floor{1e-6};
	bool verbose{true};
};

struct EMResult {
	Params params;
	std::vector<double> loglik_trace;
};

[[nodiscard]] EMResult fit_em_normal(const Params& init, const std::vector<double>& r, const EMOptions& opt);

}