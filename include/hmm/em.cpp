#include "../../include/hmm/em.hpp"
#include "../../include/hmm/forward_backward.hpp"
#include "../../include/util/check.hpp"
#include <iostream>
#include <cmath>
#include <numbers>

namespace hmm {

    EMResult fit_em_normal(const Params& init, const std::vector<double>& r, const EMOptions& opt) {
        util::check(init.K >= 2, "K must be >=2");
        const int T{ static_cast<int>(r.size()) };
        util::check(T >= 2, "Need T>=2");

        Params cur{ init };
        cur.normalize(opt.prob_floor);
        cur.validate();

        const int K{ cur.K };

        EMResult out;
        out.params = cur;
        out.loglik_trace.reserve(opt.max_iters);  // avoid reallocations

        double prev_ll{ -1e300 };

        // precompute squared returns (used in sigma update)
        std::vector<double> r_squared(T);
        for (int t{ 0 }; t < T; ++t) {
            r_squared[t] = r[t] * r[t];
        }

        // precompute sigma floor squared (avoid repeated multiplication)
        const double sigma_floor_sq{ opt.sigma_floor * opt.sigma_floor };

        // pre-allocate buffers for M-step to avoid repeated allocations
        std::vector<double> gamma_sum_i(K);  // Sum of gamma[t][i] over t=0..T-2
        std::vector<double> gamma_sum_k(K);  // Sum of gamma[t][k] over all t

        for (int it{ 0 }; it < opt.max_iters; ++it) {
            // E-step
            const FBResult fb{ forward_backward_normal(cur, r) };
            const double ll{ fb.loglik };
            out.loglik_trace.push_back(ll);

            if (opt.verbose) [[unlikely]] {
                std::cerr << "[EM] iter " << it << " loglik=" << ll << "\n";
            }

            // convergence check (after first iter)
            if (it > 0 && std::abs(ll - prev_ll) < opt.tol) [[unlikely]] {
                break;
            }
            prev_ll = ll;

            // M-step

            // update pi (just copy first gamma)
            for (int k{ 0 }; k < K; ++k) {
                cur.pi[k] = fb.gamma[0][k];
            }

            // precompute gamma sums for efficiency
            std::fill(gamma_sum_i.begin(), gamma_sum_i.end(), 0.0);
            for (int i{ 0 }; i < K; ++i) {
                for (int t{ 0 }; t < T - 1; ++t) {
                    gamma_sum_i[i] += fb.gamma[t][i];
                }
            }

            // update P (single pass over time series)
            for (int i{ 0 }; i < K; ++i) {
                const double denom{ gamma_sum_i[i] };
                const bool valid_denom{ denom > 0.0 };
                const double inv_K{ 1.0 / static_cast<double>(K) };

                for (int j{ 0 }; j < K; ++j) {
                    double numer{ 0.0 };
                    for (int t{ 0 }; t < T - 1; ++t) {
                        numer += fb.xi[t][i][j];
                    }
                    cur.P[i][j] = valid_denom ? (numer / denom) : inv_K;
                }
            }

            // update sigma (Normal, mean=0 fixed) - optimized single pass
            std::fill(gamma_sum_k.begin(), gamma_sum_k.end(), 0.0);
            std::vector<double> weighted_x2_sum(K, 0.0);

            for (int t{ 0 }; t < T; ++t) {
                const double r_sq{ r_squared[t] };
                for (int k{ 0 }; k < K; ++k) {
                    const double w{ fb.gamma[t][k] };
                    gamma_sum_k[k] += w;
                    weighted_x2_sum[k] += w * r_sq;
                }
            }

            for (int k{ 0 }; k < K; ++k) {
                const double wsum{ gamma_sum_k[k] };
                const double var{ (wsum > 0.0) ? (weighted_x2_sum[k] / wsum)
                                               : (cur.sigma[k] * cur.sigma[k]) };
                cur.sigma[k] = std::sqrt(std::max(var, sigma_floor_sq));
            }

            // normalize + floors to prevent collapse
            cur.normalize(opt.prob_floor);
        }

        out.params = cur;
        return out;
    }

} // namespace hmm