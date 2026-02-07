#include "../../include/hmm/forward_backward.hpp"
#include "../../include/dist/normal.hpp"
#include "../../include/util/logsumexp.hpp"
#include <limits>
#include <cmath>

namespace hmm {

    static std::vector<std::vector<double>> make_mat(const int T, const int K, const double val) {
        return std::vector<std::vector<double>>(T, std::vector<double>(K, val));
    }

    FBResult forward_backward_normal(const Params& params, const std::vector<double>& r) {
        params.validate();
        const int K{params.K};
        const int T{static_cast<int>(r.size())};
        util::check(T >= 2, "Need T>=2");

        FBResult out;
        out.log_alpha = make_mat(T, K, -std::numeric_limits<double>::infinity());
        out.log_beta = make_mat(T, K, -std::numeric_limits<double>::infinity());
        out.gamma = make_mat(T, K, 0.0);
        out.xi = std::vector<std::vector<std::vector<double>>>(T - 1,
            std::vector<std::vector<double>>(K, std::vector<double>(K, 0.0)));

        // precompute log P and log pi for speed.
        std::vector<double> log_pi(K);
        std::vector<std::vector<double>> logP(K, std::vector<double>(K));
        for (int i{0}; i < K; ++i) {
            log_pi[i] = std::log(params.pi[i]);
            for (int j{0}; j < K; ++j) {
                logP[i][j] = std::log(params.P[i][j]);
            }
        }

        // precompute all emissions
        std::vector<std::vector<double>> log_emissions(T, std::vector<double>(K));
        for (int t{0}; t < T; ++t) {
            for (int k{0}; k < K; ++k) {
                log_emissions[t][k] = dist::Normal::logpdf(r[t], params.sigma[k]);
            }
        }

        // allocate temporary buffers once 
        std::vector<double> tmp(K);

        // fwd init
        for (int k{0}; k < K; ++k) {
            out.log_alpha[0][k] = log_pi[k] + log_emissions[0][k];
        }

        // fwd recursion
        for (int t{1}; t < T; ++t) {
            for (int k{0}; k < K; ++k) {
                for (int i{0}; i < K; ++i) {
                    tmp[i] = out.log_alpha[t - 1][i] + logP[i][k];
                }
                out.log_alpha[t][k] = log_emissions[t][k] + util::logsumexp(tmp);
            }
        }

        // loglik = log sum_k alpha[T-1][k]
        out.loglik = util::logsumexp(out.log_alpha[T - 1]);

        // backward init
        for (int k{0}; k < K; ++k) {
            out.log_beta[T - 1][k] = 0.0;
        }

        // backward recursion
        for (int t{T - 2}; t >= 0; --t) {
            for (int i{0}; i < K; ++i) {
                for (int j{0}; j < K; ++j) {
                    tmp[j] = logP[i][j] + log_emissions[t + 1][j] + out.log_beta[t + 1][j];
                }
                out.log_beta[t][i] = util::logsumexp(tmp);
            }
        }

        // gamma
        for (int t{0}; t < T; ++t) {
            // normalize in prob space using logsumexp
            for (int k{0}; k < K; ++k) {
                tmp[k] = out.log_alpha[t][k] + out.log_beta[t][k];
            }
            const double logZ{util::logsumexp(tmp)};
            for (int k{0}; k < K; ++k) {
                out.gamma[t][k] = std::exp(tmp[k] - logZ);
            }
        }

        // xi (t=0..T-2)
        std::vector<double> log_xi_flat(K * K);  // flattened for better cache locality
        for (int t{0}; t < T - 1; ++t) {
            // compute unnormalized log xi_{ij}
            double logZ{-std::numeric_limits<double>::infinity()};
            for (int i{0}; i < K; ++i) {
                for (int j{0}; j < K; ++j) {
                    const double v{out.log_alpha[t][i] + logP[i][j] + 
                                   log_emissions[t + 1][j] + out.log_beta[t + 1][j]};
                    log_xi_flat[i * K + j] = v;
                    logZ = util::logaddexp(logZ, v);
                }
            }
            // normalize
            for (int i{0}; i < K; ++i) {
                for (int j{0}; j < K; ++j) {
                    out.xi[t][i][j] = std::exp(log_xi_flat[i * K + j] - logZ);
                }
            }
        }

        return out;
    }

} // namespace hmm
