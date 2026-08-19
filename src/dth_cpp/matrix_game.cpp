#include "dth.hpp"
#include "highs_backend.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <ranges>
#include <stdexcept>

namespace dth {

double matrix_cell(
    const TransitionValues& t,
    const std::size_t drop,
    const std::size_t check)
{
    if (check >= drop) {
        std::size_t lag = (check - drop);
        return t.success[lag];
    } else {
        return t.failed;
    }
}

//SECTION 9
std::optional<Policy> normalize_policy(Policy raw, double negative_limit)
noexcept{
  double total{0.0};
  for (std::size_t action : std::views::iota(std::size_t{0}, kActions)) {
    const double mass = raw.mass[action];
    if (!std::isfinite(mass) || mass < -negative_limit) {
    return std::nullopt;
  }
    if (mass < 0.0) {
        raw.mass[action] = 0.0;
    }
    total += raw.mass[action];
}
if (!std::isfinite(total) || total <= 0.0) {
    return std::nullopt;
}
for (double& mass : raw.mass) {
    mass /= total;
}
return raw;
}

std::optional<Certified> certify(
    const TransitionValues& t,
    Policy raw_drop,
    Policy raw_check,
    double negative_limit) {
    const auto normalized_drop = normalize_policy(raw_drop, negative_limit);
    const auto normalized_check = normalize_policy(raw_check, negative_limit);

    if(!normalized_drop || !normalized_check) {
        return std::nullopt;
    }
    const Policy& p = *normalized_drop;
    const Policy& q = *normalized_check;

    double lower = std::numeric_limits<double>::infinity();
    for (int check : std::views::iota(0, static_cast<int>(kActions))){
        double payoff = 0.0;
        for (int drop : std::views::iota(0, static_cast<int>(kActions))) {
            payoff += p.mass[drop] * matrix_cell(t, drop, check);
        }
        lower = std::min(lower, payoff);
    }

    double upper = -std::numeric_limits<double>::infinity();
    for (int drop : std::views::iota(0, static_cast<int>(kActions))) {
        double payoff = 0.0;
        for (int check : std::views::iota(0, static_cast<int>(kActions))) {
            payoff += matrix_cell(t, drop, check) * q.mass[check];
        }
        upper = std::max(upper, payoff);
    }

    const double gap = std::max(0.0, upper - lower);
    if (gap > kSaddleTolerance) {
        return std::nullopt;
    }
    const double midpoint = (lower + upper) / 2.0;

    if (!std::isfinite(midpoint) || midpoint < -1.0 - 1e-9 || midpoint > 1.0 + 1e-9) {
        throw std::logic_error("certified midpoint outside the payoff box");
    }
    return Certified{p, q, Certificate{lower, upper, midpoint, gap}};
}

//SECTION 10
PureSaddleScan scan_pure_saddle(const TransitionValues& t) {
    std::array<double, kActions> prefix_min{};
    std::array<double, kActions> prefix_max{};
    double running_min = std::numeric_limits<double>::infinity();
    double running_max = -std::numeric_limits<double>::infinity();

    for (std::size_t j : std::views::iota(std::size_t{0}, kActions)) {
        running_min = std::min(running_min, t.success[j]);
        running_max = std::max(running_max, t.success[j]);
        prefix_min[j] = running_min;
        prefix_max[j] = running_max;
    }

    double maximin = -std::numeric_limits<double>::infinity();
    int best_drop = 0;
    for (int drop : std::views::iota(0, static_cast<int>(kActions))) {
        double candidate = prefix_min[kActions - static_cast<std::size_t>(1 + drop)];
        if (drop > 0) {
            candidate = std::min(candidate, t.failed);
        }
        if (candidate > maximin) {
            maximin = candidate;
            best_drop = drop;
        }
    }

    double minimax = std::numeric_limits<double>::infinity();
    int best_check = 0;
    for (int check : std::views::iota(0, static_cast<int>(kActions))) {
        double candidate = prefix_max[static_cast<std::size_t>(check)];
        if (check + 1 < static_cast<int>(kActions)) {
            candidate = std::max(candidate, t.failed);
        }
        if (candidate < minimax) {
            minimax = candidate;
            best_check = check;
        }
    }

    return PureSaddleScan{
        maximin,
        minimax, 
        static_cast<std::size_t>(best_drop), 
        static_cast<std::size_t>(best_check)
    };
}

std::optional<Certified> try_pure_saddle(const TransitionValues& t) {
    const auto scan = scan_pure_saddle(t);

    if (scan.minimax - scan.maximin > kSaddleTolerance) {
        return std::nullopt;
    }

    dth::Policy p{};
    p.mass[scan.best_drop] = 1;
    dth::Policy q{};
    q.mass[scan.best_check] = 1;
    return certify(t, p, q, 0.0);

}

//SECTION 12
std::optional<Certified> try_support(
    const TransitionValues& t,
    std::span<const std::size_t> drop_indices,
    std::span<const std::size_t> check_indices,
    HighsBackend& backend,
    MatrixScratch& scratch
    ) {
        const std::size_t k = std::min(drop_indices.size(), check_indices.size());

        if(k == 0) {
            return std::nullopt;
        }

        for (std::size_t i = 0; i < k; ++i) {
            for (std::size_t j = 0; j < k; ++j) {
                scratch.matrix[i * k + j] = matrix_cell(t, drop_indices[i], check_indices[j]);
            }
        }
        EqualizerRaw result{};
        const auto matrix = std::span<const double>{
            scratch.matrix.data(),
            k*k,
        };
        auto status = backend.solve_equalizer(matrix, k, result);

        if (status == NumericStatus::Infeasible || status == NumericStatus::InfeasibleOrUnbounded
        || status == NumericStatus::IterationLimit || status == NumericStatus:: Failure) {
            return std::nullopt;
        }
        if (status != NumericStatus::Optimal) {
            throw std::logic_error("invalid equalizer model or impossible backend status");
        }

        for (std::size_t i = 0; i < k; ++i) {
            if (!std::isfinite(result.check_mass[i]) || !std::isfinite(result.drop_mass[i])) {
                throw std::logic_error("raw policy masses must be finite");
            }
        }

        Policy raw_drop = Policy{};
        Policy raw_check = Policy{};

        for (std::size_t i = 0; i < k; ++i) {
            raw_drop.mass[drop_indices[i]] = result.drop_mass[i];
            raw_check.mass[check_indices[i]] = result.check_mass[i];
        }

        return certify(t, raw_drop, raw_check, 1e-10);
    }
} // namespace dth
