#include "interface/distribution.hpp"
#include "interface/mcmc.hpp"
#include "interface/model.hpp"
#include "interface/plugin.hpp"
#include "interface/random.hpp"
#include "interface/redirect_stdio.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace {

class WarmupResult : public theta::ResultMeanCov {
public:
    explicit WarmupResult(std::size_t parameter_count)
        : theta::ResultMeanCov(parameter_count), last(parameter_count, 0.0) {}

    const std::vector<double> &get_last() const {
        return last;
    }

protected:
    void fill2(const double *values, double, std::size_t) {
        std::copy(values, values + getnpar(), last.begin());
    }

private:
    std::vector<double> last;
};

theta::Matrix covariance_from_cholesky(const theta::Matrix &lower) {
    const std::size_t count = lower.get_n_rows();
    theta::Matrix covariance(count, count);
    for (std::size_t row = 0; row < count; ++row) {
        for (std::size_t column = 0; column <= row; ++column) {
            double value = 0.0;
            for (std::size_t index = 0; index <= column; ++index) {
                value += lower(row, index) * lower(column, index);
            }
            covariance(row, column) = value;
            covariance(column, row) = value;
        }
    }
    return covariance;
}

theta::Matrix regularized_cholesky(
    const theta::Matrix &empirical,
    const theta::Matrix &reference,
    double requested_shrinkage
) {
    const std::size_t count = empirical.get_n_rows();
    double shrinkage = requested_shrinkage;
    std::runtime_error last_error("regularized Cholesky failed");
    for (unsigned int attempt = 0; attempt < 8; ++attempt) {
        theta::Matrix covariance(count, count);
        for (std::size_t row = 0; row < count; ++row) {
            for (std::size_t column = 0; column < count; ++column) {
                covariance(row, column) =
                    (1.0 - shrinkage) * empirical(row, column)
                    + shrinkage * reference(row, column);
            }
        }
        try {
            theta::Matrix lower;
            theta::get_cholesky(covariance, lower);
            if (shrinkage > requested_shrinkage) {
                theta::out
                    << "target_adaptive_mcmc_cov: increased covariance "
                    << "shrinkage to " << shrinkage << std::endl;
            }
            return lower;
        } catch (const std::exception &error) {
            last_error = std::runtime_error(error.what());
            shrinkage = 0.5 * (1.0 + shrinkage);
        }
    }
    throw last_error;
}

class target_adaptive_mcmc_cov : public theta::MCMCStrategy {
public:
    explicit target_adaptive_mcmc_cov(const theta::Configuration &cfg)
        : theta::MCMCStrategy(cfg),
          adapt_iterations(100000),
          adapt_burn_in(10000),
          adapt_passes(3),
          adapt_factor(options.factor),
          shrinkage(0.10),
          start_scale(1.0),
          start_attempts(100) {
        theta::Setting setting = cfg.setting;
        if (setting.exists("adapt-iterations")) {
            adapt_iterations = setting["adapt-iterations"];
        }
        if (setting.exists("adapt-burn-in")) {
            adapt_burn_in = setting["adapt-burn-in"];
        }
        if (setting.exists("adapt-passes")) {
            adapt_passes = setting["adapt-passes"];
        }
        if (setting.exists("adapt-factor")) {
            adapt_factor = setting["adapt-factor"];
        }
        if (setting.exists("shrinkage")) {
            shrinkage = setting["shrinkage"];
        }
        if (setting.exists("start-scale")) {
            start_scale = setting["start-scale"];
        }
        if (setting.exists("start-attempts")) {
            start_attempts = setting["start-attempts"];
        }
        if (adapt_iterations == 0 || adapt_passes == 0) {
            throw theta::ConfigurationException(
                "adapt-iterations and adapt-passes must be positive"
            );
        }
        if (shrinkage <= 0.0 || shrinkage > 1.0) {
            throw theta::ConfigurationException(
                "shrinkage must be in the interval (0, 1]"
            );
        }
        if (start_scale < 0.0 || start_attempts == 0) {
            throw theta::ConfigurationException(
                "start-scale must be non-negative and start-attempts positive"
            );
        }
    }

    void init(
        const theta::Model &model,
        const boost::shared_ptr<theta::Distribution>
            &override_parameter_distribution
    ) {
        const theta::Distribution &distribution =
            override_parameter_distribution.get()
            ? *override_parameter_distribution
            : model.get_parameter_distribution();
        theta::ParValues mode;
        distribution.mode(mode);
        options.startvalues.resize(distribution.get_parameters().size());
        mode.fill(&options.startvalues[0], distribution.get_parameters());
        initial_sqrt_covariance = theta::get_sqrt_cov2(
            *rnd_gen, model, override_parameter_distribution
        );
        initial_covariance = covariance_from_cholesky(initial_sqrt_covariance);
    }

    void run_mcmc(
        const theta::Function &negative_log_likelihood,
        theta::MCMCResult &result
    ) const {
        std::vector<double> start = dispersed_start(negative_log_likelihood);
        theta::Matrix proposal = initial_sqrt_covariance;

        for (unsigned int pass = 0; pass < adapt_passes; ++pass) {
            theta::MCMCOptions warmup;
            warmup.startvalues = start;
            warmup.iterations = adapt_iterations;
            warmup.burn_in = adapt_burn_in;
            warmup.factor = adapt_factor;
            warmup.ignore_inf_nll_start = false;

            WarmupResult warmup_result(start.size());
            theta::metropolis_hastings_multigauss(
                negative_log_likelihood,
                warmup,
                warmup_result,
                *rnd_gen,
                proposal
            );
            start = warmup_result.get_last();
            proposal = regularized_cholesky(
                warmup_result.getCov(),
                initial_covariance,
                shrinkage
            );
            theta::out
                << "target_adaptive_mcmc_cov: pass " << (pass + 1)
                << "/" << adapt_passes
                << ", acceptance="
                << static_cast<double>(warmup_result.getCountDifferent())
                    / warmup_result.getCount()
                << std::endl;
        }

        theta::MCMCOptions production = options;
        production.startvalues = start;
        theta::metropolis_hastings_multigauss(
            negative_log_likelihood,
            production,
            result,
            *rnd_gen,
            proposal
        );
    }

private:
    std::vector<double> dispersed_start(
        const theta::Function &negative_log_likelihood
    ) const {
        if (start_scale == 0.0) {
            return options.startvalues;
        }
        const std::size_t count = options.startvalues.size();
        std::vector<double> normal(count);
        std::vector<double> candidate(count);
        for (unsigned int attempt = 0; attempt < start_attempts; ++attempt) {
            for (std::size_t index = 0; index < count; ++index) {
                normal[index] = rnd_gen->gauss();
            }
            for (std::size_t row = 0; row < count; ++row) {
                candidate[row] = options.startvalues[row];
                for (std::size_t column = 0; column <= row; ++column) {
                    candidate[row] +=
                        start_scale
                        * initial_sqrt_covariance(row, column)
                        * normal[column];
                }
            }
            const double nll = negative_log_likelihood(&candidate[0]);
            if (std::isfinite(nll)) {
                theta::out
                    << "target_adaptive_mcmc_cov: dispersed start accepted "
                    << "after " << (attempt + 1) << " attempt(s)"
                    << std::endl;
                return candidate;
            }
        }
        theta::out
            << "target_adaptive_mcmc_cov: no finite dispersed start found; "
            << "using the distribution mode" << std::endl;
        return options.startvalues;
    }

    unsigned int adapt_iterations;
    unsigned int adapt_burn_in;
    unsigned int adapt_passes;
    double adapt_factor;
    double shrinkage;
    double start_scale;
    unsigned int start_attempts;
    theta::Matrix initial_sqrt_covariance;
    theta::Matrix initial_covariance;
};

}  // namespace

REGISTER_PLUGIN(target_adaptive_mcmc_cov)
