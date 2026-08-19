#pragma once

#include "dth.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>

namespace dth {

// HiGHS statuses are translated here so no third-party type leaks into the
// game solver's public header or rung implementations.
enum class NumericStatus : std::uint8_t {
    Optimal,
    Infeasible,
    Unbounded,
    InfeasibleOrUnbounded,
    IterationLimit,
    InvalidInput,
    Failure,
};

// Masses are indexed within the supplied support, not by literal action.
// The rung that selected the support remains responsible for embedding them
// into a 60-action Policy and invoking the independent certificate.
struct EqualizerRaw {
    std::array<double, kActions> drop_mass{};
    std::array<double, kActions> check_mass{};
    double drop_value{};
    double check_value{};
    std::uint64_t iterations{};
};

// These are the unnormalised variables of BUILD.md's shifted covering and
// packing programs. They are deliberately not certified policies.
struct CoveringRaw {
    Policy x{};
    Policy y{};
    double sum_x{};
    double sum_y{};
    std::uint64_t iterations{};
};

class HighsBackend {
public:
    HighsBackend();
    ~HighsBackend();

    HighsBackend(const HighsBackend&) = delete;
    HighsBackend& operator=(const HighsBackend&) = delete;
    HighsBackend(HighsBackend&&) noexcept;
    HighsBackend& operator=(HighsBackend&&) noexcept;

    // Solve the two equality-feasibility systems for a square support matrix.
    // The matrix is row-major with exactly dimension*dimension entries.
    [[nodiscard]] NumericStatus solve_equalizer(
        std::span<const double> support_matrix,
        std::size_t dimension,
        EqualizerRaw& output);

    // Solve BUILD.md's explicit shifted covering and packing programs. The
    // matrix is row-major by Dropper action, then Checker action.
    [[nodiscard]] NumericStatus solve_covering(
        std::span<const double> shifted_matrix,
        std::size_t dimension,
        CoveringRaw& output);

    [[nodiscard]] std::string version() const;
    [[nodiscard]] const std::string& last_error() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace dth
