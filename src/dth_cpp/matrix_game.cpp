#include "dth.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <ranges>
#include <stdexcept>

double dth::matrix_cell(const TransitionValues& t, int drop, int check) {
    if (check >= drop) {
        std::size_t lag = static_cast<std::size_t>(check - drop);
        return t.success[lag];
    } else {
        return t.failed;
    }
}

//SECTION 9
std::optional<dth::Policy> dth::normalize_policy(Policy raw, double negative_limit)
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

std::optional<dth::Certified> dth::certify(
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
dth::PureSaddleScan dth::scan_pure_saddle(const TransitionValues& t) {
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

std::optional<dth::Certified> dth::try_pure_saddle(const TransitionValues& t) {
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

//SECTION 11
bool dth::solve_linear(std::span<double> a, std::span<double> b, std::size_t n, std::span<double> x) {
    inline constexpr std::size_t kLinearDimension = kActions + 1;
    
    for (std::size_t col : std::views::iota(std::size_t{0}, n-1)){
        std::size_t pivot_row = col;
        double best = std::abs(A[col * kLinearDimension + col]);
        for (std::size_t row : std::views::iota(col+1, n-1)) {
            double magnitude = std::abs(A[row * kLinearDimension + col]);
            if (magnitude > best) {
                best = magnitude;
                pivot_row = row;
            }
        }
        if (best < kPivotTolerance) {
            return false;
        }
        
    }
}