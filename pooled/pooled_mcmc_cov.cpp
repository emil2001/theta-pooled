#include "interface/distribution.hpp"
#include "interface/mcmc.hpp"
#include "interface/model.hpp"
#include "interface/plugin.hpp"
#include "interface/random.hpp"
#include "interface/redirect_stdio.hpp"
#include "interface/variables.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct CovarianceFile {
    std::vector<std::string> parameters;
    std::vector<double> mean;
    theta::Matrix covariance;
};

CovarianceFile read_covariance_file(
    const std::string &path,
    const std::vector<std::string> &expected_parameters
) {
    std::ifstream input(path.c_str());
    if (!input) {
        throw std::runtime_error("pooled_mcmc_cov: cannot open " + path);
    }
    std::string token;
    std::size_t count = 0;
    input >> token >> count;
    if (token != "theta_pooled_covariance_v1") {
        throw std::runtime_error(
            "pooled_mcmc_cov: unsupported covariance file format"
        );
    }
    if (count != expected_parameters.size()) {
        throw std::runtime_error(
            "pooled_mcmc_cov: covariance dimension does not match model"
        );
    }

    CovarianceFile result;
    result.parameters.resize(count);
    result.mean.resize(count);
    result.covariance.reset(count, count);
    input >> token;
    if (token != "parameters") {
        throw std::runtime_error(
            "pooled_mcmc_cov: expected parameter names"
        );
    }
    for (std::size_t index = 0; index < count; ++index) {
        input >> result.parameters[index];
        if (result.parameters[index] != expected_parameters[index]) {
            throw std::runtime_error(
                "pooled_mcmc_cov: parameter order mismatch at "
                + expected_parameters[index]
            );
        }
    }
    input >> token;
    if (token != "mean") {
        throw std::runtime_error("pooled_mcmc_cov: expected pooled mean");
    }
    for (std::size_t index = 0; index < count; ++index) {
        input >> result.mean[index];
        if (!std::isfinite(result.mean[index])) {
            throw std::runtime_error(
                "pooled_mcmc_cov: non-finite pooled mean"
            );
        }
    }
    input >> token;
    if (token != "covariance") {
        throw std::runtime_error("pooled_mcmc_cov: expected covariance");
    }
    for (std::size_t row = 0; row < count; ++row) {
        for (std::size_t column = 0; column < count; ++column) {
            input >> result.covariance(row, column);
            if (!std::isfinite(result.covariance(row, column))) {
                throw std::runtime_error(
                    "pooled_mcmc_cov: non-finite covariance"
                );
            }
        }
    }
    if (!input) {
        throw std::runtime_error(
            "pooled_mcmc_cov: truncated covariance file"
        );
    }
    return result;
}

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

theta::Matrix shrunk_cholesky(
    const theta::Matrix &pooled,
    const theta::Matrix &reference,
    double requested_shrinkage
) {
    const std::size_t count = pooled.get_n_rows();
    double shrinkage = requested_shrinkage;
    std::runtime_error last_error("pooled_mcmc_cov: Cholesky failed");
    for (unsigned int attempt = 0; attempt < 8; ++attempt) {
        theta::Matrix covariance(count, count);
        for (std::size_t row = 0; row < count; ++row) {
            for (std::size_t column = 0; column < count; ++column) {
                covariance(row, column) =
                    (1.0 - shrinkage) * pooled(row, column)
                    + shrinkage * reference(row, column);
            }
        }
        try {
            theta::Matrix lower;
            theta::get_cholesky(covariance, lower);
            theta::out
                << "pooled_mcmc_cov: loaded shared covariance with shrinkage="
                << shrinkage << std::endl;
            return lower;
        } catch (const std::exception &error) {
            last_error = std::runtime_error(error.what());
            shrinkage = 0.5 * (1.0 + shrinkage);
        }
    }
    throw last_error;
}

std::size_t fix_delta_coordinates(
    CovarianceFile &pooled,
    const theta::Distribution &distribution,
    const theta::ParIds &parameter_ids
) {
    theta::ParValues mode;
    distribution.mode(mode);
    std::size_t fixed_count = 0;
    std::size_t index = 0;
    for (
        theta::ParIds::const_iterator parameter = parameter_ids.begin();
        parameter != parameter_ids.end();
        ++parameter, ++index
    ) {
        const std::pair<double, double> bounds =
            distribution.support(*parameter);
        if (bounds.first != bounds.second) {
            continue;
        }
        pooled.mean[index] = mode.get(*parameter);
        for (
            std::size_t other = 0;
            other < parameter_ids.size();
            ++other
        ) {
            pooled.covariance(index, other) = 0.0;
            pooled.covariance(other, index) = 0.0;
        }
        ++fixed_count;
    }
    return fixed_count;
}

class pooled_mcmc_cov : public theta::MCMCStrategy {
public:
    explicit pooled_mcmc_cov(const theta::Configuration &cfg)
        : theta::MCMCStrategy(cfg),
          vm(cfg.pm->get<theta::VarIdManager>()),
          shrinkage(0.10),
          start_scale(1.0),
          start_attempts(100) {
        theta::Setting setting = cfg.setting;
        if (!setting.exists("covariance-file")) {
            throw theta::ConfigurationException(
                "pooled_mcmc_cov requires covariance-file"
            );
        }
        covariance_path = static_cast<std::string>(
            setting["covariance-file"]
        );
        if (setting.exists("shrinkage")) {
            shrinkage = setting["shrinkage"];
        }
        if (setting.exists("start-scale")) {
            start_scale = setting["start-scale"];
        }
        if (setting.exists("start-attempts")) {
            start_attempts = setting["start-attempts"];
        }
        if (shrinkage < 0.0 || shrinkage > 1.0) {
            throw theta::ConfigurationException(
                "shrinkage must be in the interval [0, 1]"
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
        const theta::ParIds &parameter_ids = distribution.get_parameters();
        std::vector<std::string> expected_parameters;
        expected_parameters.reserve(parameter_ids.size());
        for (
            theta::ParIds::const_iterator parameter = parameter_ids.begin();
            parameter != parameter_ids.end();
            ++parameter
        ) {
            expected_parameters.push_back(vm->get_name(*parameter));
        }

        theta::Matrix asimov_sqrt = theta::get_sqrt_cov2(
            *rnd_gen, model, override_parameter_distribution
        );
        const theta::Matrix asimov_covariance =
            covariance_from_cholesky(asimov_sqrt);
        CovarianceFile pooled = read_covariance_file(
            covariance_path, expected_parameters
        );
        const std::size_t fixed_count = fix_delta_coordinates(
            pooled, distribution, parameter_ids
        );
        if (fixed_count > 0) {
            theta::out
                << "pooled_mcmc_cov: holding "
                << fixed_count
                << " delta-distributed parameter(s) fixed"
                << std::endl;
        }
        sqrt_covariance = shrunk_cholesky(
            pooled.covariance, asimov_covariance, shrinkage
        );
        center = pooled.mean;
        options.startvalues = center;
    }

    void run_mcmc(
        const theta::Function &negative_log_likelihood,
        theta::MCMCResult &result
    ) const {
        theta::MCMCOptions production = options;
        production.startvalues = dispersed_start(negative_log_likelihood);
        theta::metropolis_hastings_multigauss(
            negative_log_likelihood,
            production,
            result,
            *rnd_gen,
            sqrt_covariance
        );
    }

private:
    std::vector<double> dispersed_start(
        const theta::Function &negative_log_likelihood
    ) const {
        if (start_scale == 0.0) {
            return center;
        }
        const std::size_t count = center.size();
        std::vector<double> normal(count);
        std::vector<double> candidate(count);
        for (unsigned int attempt = 0; attempt < start_attempts; ++attempt) {
            for (std::size_t index = 0; index < count; ++index) {
                normal[index] = rnd_gen->gauss();
            }
            for (std::size_t row = 0; row < count; ++row) {
                candidate[row] = center[row];
                for (std::size_t column = 0; column <= row; ++column) {
                    candidate[row] +=
                        start_scale
                        * sqrt_covariance(row, column)
                        * normal[column];
                }
            }
            const double nll = negative_log_likelihood(&candidate[0]);
            if (std::isfinite(nll)) {
                theta::out
                    << "pooled_mcmc_cov: dispersed start accepted after "
                    << (attempt + 1) << " attempt(s)" << std::endl;
                return candidate;
            }
        }
        theta::out
            << "pooled_mcmc_cov: no finite dispersed start found; "
            << "using the pooled mean" << std::endl;
        return center;
    }

    boost::shared_ptr<theta::VarIdManager> vm;
    std::string covariance_path;
    double shrinkage;
    double start_scale;
    unsigned int start_attempts;
    theta::Matrix sqrt_covariance;
    std::vector<double> center;
};

}  // namespace

REGISTER_PLUGIN(pooled_mcmc_cov)
