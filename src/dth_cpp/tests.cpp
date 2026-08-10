#include "dth.hpp"

#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {

void require(const bool condition, const char* const message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void test_constant_products()
{
    using namespace dth;

    const auto profile_count =
        static_cast<ClassId>(kCanonicalProfiles);

    require(
        profile_count * profile_count == kCanonicalClasses,
        "canonical class count is inconsistent");

    require(
        kCanonicalClasses
            <= static_cast<ClassId>(
                std::numeric_limits<std::uint32_t>::max()),
        "canonical class count does not fit in uint32");

    require(
        kCanonicalClasses - 1
            <= std::numeric_limits<ClassId>::max(),
        "ClassId cannot represent the last class");

    require(
        kDeadProfileBase + kCapacity == kCanonicalProfiles,
        "dead-profile range is inconsistent");

    require(
        kMaxProfilePotential + 1 == 601,
        "potential bucket count is inconsistent");
}

void test_default_construction()
{
    const dth::ProfileTable table{};
    const dth::TransitionValues transitions{};
    const dth::Policy policy{};
    const dth::Certificate certificate{};
    const dth::SolveResult result{};
    const dth::RouteCounters counters{};

    require(table.profile_count == 0, "profile count is not zero");
    require(table.st.empty(), "ST storage is not empty");
    require(table.ttd.empty(), "TTD storage is not empty");
    require(
        table.potential.empty(),
        "potential storage is not empty");
    require(
        table.revival.empty(),
        "revival storage is not empty");
    require(
        table.success_child.empty(),
        "success-child storage is not empty");
    require(
        table.failure_child.empty(),
        "failure-child storage is not empty");
    require(
        table.alive_id.empty(),
        "alive lookup is not empty");
    for (const auto& bucket : table.buckets) {
        require(
            bucket.empty(),
            "default potential bucket is not empty");
    }

    require(
        transitions.success.front() == 0.0,
        "success value is uninitialized");
    require(
        transitions.failed == 0.0,
        "failure value is uninitialized");
    require(
        policy.mass.front() == 0.0,
        "policy mass is uninitialized");
    require(
        certificate.gap == 0.0,
        "certificate is uninitialized");
    require(
        result.route == dth::SolverRoute::Pure,
        "default solver route is inconsistent");
    require(
        counters.pure == 0,
        "route counter is uninitialized");
}

void test_solver_kind_mapping()
{
    using dth::SolverKind;
    using dth::SolverRoute;

    require(
        dth::solver_kind_for(SolverRoute::Pure)
            == SolverKind::Pure,
        "pure route mapping is incorrect");

    require(
        dth::solver_kind_for(SolverRoute::WarmSupport)
            == SolverKind::Support,
        "warm-support route mapping is incorrect");

    require(
        dth::solver_kind_for(SolverRoute::FullSupport)
            == SolverKind::Support,
        "full-support route mapping is incorrect");

    require(
        dth::solver_kind_for(SolverRoute::LinearProgram)
            == SolverKind::LinearProgram,
        "linear-program route mapping is incorrect");
}

} // namespace

int main()
{
    try {
        test_constant_products();
        test_default_construction();
        test_solver_kind_mapping();

        std::cout << "All DTH tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr
            << "DTH test failure: "
            << error.what()
            << '\n';
        return 1;
    }
}
