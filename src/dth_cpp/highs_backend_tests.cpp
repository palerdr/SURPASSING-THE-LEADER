#include "highs_backend.hpp"

#include <array>
#include <cmath>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

void require(const bool condition, const char* const message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void test_version_and_invalid_input()
{
    dth::HighsBackend backend;
    require(backend.version() == "1.15.1", "unexpected HiGHS version");

    dth::EqualizerRaw equalizer{};
    const std::array<double, 1> nonfinite = {
        std::numeric_limits<double>::quiet_NaN(),
    };
    require(
        backend.solve_equalizer(nonfinite, 1, equalizer)
            == dth::NumericStatus::InvalidInput,
        "the backend accepted a nonfinite equalizer");
}

void test_equalizer()
{
    dth::HighsBackend backend;
    const std::array<double, 4> matching_pennies = {
        1.0, -1.0,
        -1.0, 1.0,
    };
    dth::EqualizerRaw output{};
    require(
        backend.solve_equalizer(matching_pennies, 2, output)
            == dth::NumericStatus::Optimal,
        "HiGHS rejected a feasible equalizer");
    require(
        std::abs(output.drop_mass[0] - 0.5) <= 1e-10
            && std::abs(output.drop_mass[1] - 0.5) <= 1e-10
            && std::abs(output.check_mass[0] - 0.5) <= 1e-10
            && std::abs(output.check_mass[1] - 0.5) <= 1e-10,
        "HiGHS returned the wrong equalizer masses");
    require(
        std::abs(output.drop_value) <= 1e-10
            && std::abs(output.check_value) <= 1e-10,
        "HiGHS returned the wrong equalizer value");

    const std::array<double, 4> asymmetric = {
        4.0, 0.0,
        1.0, 2.0,
    };
    require(
        backend.solve_equalizer(asymmetric, 2, output)
            == dth::NumericStatus::Optimal,
        "HiGHS rejected an asymmetric feasible equalizer");
    require(
        std::abs(output.drop_mass[0] - 0.2) <= 1e-10
            && std::abs(output.drop_mass[1] - 0.8) <= 1e-10
            && std::abs(output.check_mass[0] - 0.4) <= 1e-10
            && std::abs(output.check_mass[1] - 0.6) <= 1e-10
            && std::abs(output.drop_value - 1.6) <= 1e-10
            && std::abs(output.check_value - 1.6) <= 1e-10,
        "HiGHS transposed an equalizer policy incorrectly");

    const std::array<double, 4> impossible_equalizer = {
        0.0, 0.0,
        1.0, 1.0,
    };
    require(
        backend.solve_equalizer(impossible_equalizer, 2, output)
            == dth::NumericStatus::Infeasible,
        "HiGHS accepted an infeasible equalizer");

    const std::array<double, 4> singular_feasible = {
        0.0, 0.0,
        0.0, 0.0,
    };
    require(
        backend.solve_equalizer(singular_feasible, 2, output)
            == dth::NumericStatus::Optimal,
        "HiGHS rejected a singular but feasible equalizer");
    require(
        output.drop_mass[0] >= 0.0 && output.drop_mass[1] >= 0.0
            && output.check_mass[0] >= 0.0 && output.check_mass[1] >= 0.0
            && std::abs(output.drop_mass[0] + output.drop_mass[1] - 1.0)
                <= 1e-10
            && std::abs(output.check_mass[0] + output.check_mass[1] - 1.0)
                <= 1e-10
            && std::abs(output.drop_value) <= 1e-10
            && std::abs(output.check_value) <= 1e-10,
        "HiGHS returned an invalid singular equalizer point");

    require(
        backend.solve_equalizer(matching_pennies, 2, output)
            == dth::NumericStatus::Optimal
            && std::abs(output.drop_mass[0] - 0.5) <= 1e-10
            && std::abs(output.check_mass[0] - 0.5) <= 1e-10,
        "the backend reused stale equalizer state");
}

void test_covering_and_packing()
{
    dth::HighsBackend backend;
    const std::array<double, 4> shifted_matching_pennies = {
        3.0, 1.0,
        1.0, 3.0,
    };
    dth::CoveringRaw output{};
    require(
        backend.solve_covering(shifted_matching_pennies, 2, output)
            == dth::NumericStatus::Optimal,
        "HiGHS rejected the shifted covering game");
    require(
        std::abs(output.x.mass[0] - 0.25) <= 1e-10
            && std::abs(output.x.mass[1] - 0.25) <= 1e-10
            && std::abs(output.y.mass[0] - 0.25) <= 1e-10
            && std::abs(output.y.mass[1] - 0.25) <= 1e-10,
        "HiGHS returned the wrong covering or packing variables");
    require(
        std::abs(output.sum_x - 0.5) <= 1e-10
            && std::abs(output.sum_y - 0.5) <= 1e-10,
        "HiGHS returned mismatched covering and packing objectives");

    const std::array<double, 4> invalid_shifted = {
        3.0, 1.0,
        1.0, std::numeric_limits<double>::infinity(),
    };
    require(
        backend.solve_covering(invalid_shifted, 2, output)
            == dth::NumericStatus::InvalidInput,
        "the backend accepted a nonfinite shifted matrix");
    require(
        backend.solve_covering(shifted_matching_pennies, 2, output)
            == dth::NumericStatus::Optimal
            && std::abs(output.x.mass[0] - 0.25) <= 1e-10
            && std::abs(output.y.mass[0] - 0.25) <= 1e-10,
        "the backend reused stale covering or packing state");
}

} // namespace

int main()
{
    try {
        test_version_and_invalid_input();
        test_equalizer();
        test_covering_and_packing();
        std::cout << "All HiGHS backend tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "HiGHS backend test failure: " << error.what() << '\n';
        return 1;
    }
}
