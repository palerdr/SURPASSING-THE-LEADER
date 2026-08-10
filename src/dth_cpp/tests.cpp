#include "dth.hpp"

#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>

// Inclusive lag is not part of the public Section 2 interface yet.
int lag(int drop, int check);

namespace {

using dth::revival_eligibility;
using dth::revival_probability;

void require(const bool condition, const char* const message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename Function>
void require_out_of_range(Function&& function, const char* const message)
{
    bool threw = false;
    try {
        function();
    } catch (const std::out_of_range&) {
        threw = true;
    }
    require(threw, message);
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

void test_revival_eligibility_boundaries()
{
    require(
        revival_eligibility(0, 0),
        "fresh profile should be revival-eligible");
    require(
        revival_eligibility(239, 0),
        "largest individually survivable dose should be eligible");
    require(
        revival_eligibility(239, 1),
        "cumulative load exactly 300 should be eligible");
    require(
        revival_eligibility(0, 240),
        "zero ST at cumulative load 300 should be eligible");
    require(
        revival_eligibility(180, 60),
        "interior cumulative-load boundary should be eligible");

    require(
        !revival_eligibility(240, 0),
        "individual dose exactly 300 should be fatal");
    require(
        !revival_eligibility(239, 2),
        "cumulative load above 300 should be fatal");
    require(
        !revival_eligibility(0, 241),
        "TTD beyond the cumulative boundary should be fatal");
    require(
        !revival_eligibility(299, 0),
        "ST above the dose boundary should be fatal");
}

void test_revival_probability_boundaries()
{
    const double origin = revival_probability(0, 0);
    require(
        std::abs(origin - 0.95)
            <= std::numeric_limits<double>::epsilon(),
        "revival probability at the origin should be 0.95");

    require(
        revival_probability(240, 0) == 0.0,
        "fatal individual dose should have zero revival probability");
    require(
        revival_probability(239, 2) == 0.0,
        "fatal cumulative load should have zero revival probability");
    require(
        revival_probability(0, 241) == 0.0,
        "fatal TTD boundary should have zero revival probability");

    require(
        revival_probability(0, 60) < origin,
        "positive TTD should reduce revival probability");
    require(
        revival_probability(60, 0) < origin,
        "positive ST should reduce revival probability");
}

void test_revival_surface_exhaustively()
{
    constexpr int capacity = static_cast<int>(dth::kCapacity);

    for (int st = 0; st < capacity; ++st) {
        for (int ttd = 0; ttd <= capacity; ++ttd) {
            const bool expected = st <= 239 && st + ttd <= 240;
            const bool actual = revival_eligibility(st, ttd);
            require(
                actual == expected,
                "revival eligibility disagrees with the frozen predicate");

            const double probability = revival_probability(st, ttd);
            require(
                std::isfinite(probability),
                "revival probability should always be finite");

            if (expected) {
                require(
                    probability > 0.0 && probability < 1.0,
                    "eligible revival probability should lie in (0, 1)");
            } else {
                require(
                    probability == 0.0,
                    "fatal profile should have exactly zero probability");
            }
        }
    }
}

void test_scalar_coordinate_validation()
{
    require_out_of_range(
        [] { static_cast<void>(revival_eligibility(-1, 0)); },
        "negative ST should be rejected");
    require_out_of_range(
        [] { static_cast<void>(revival_eligibility(300, 0)); },
        "ST at capacity should be rejected");
    require_out_of_range(
        [] { static_cast<void>(revival_eligibility(0, -1)); },
        "negative TTD should be rejected");
    require_out_of_range(
        [] { static_cast<void>(revival_eligibility(0, 301)); },
        "TTD above capacity should be rejected");

    require_out_of_range(
        [] { static_cast<void>(revival_probability(-1, 0)); },
        "probability should reject negative ST");
    require_out_of_range(
        [] { static_cast<void>(revival_probability(300, 0)); },
        "probability should reject ST at capacity");
    require_out_of_range(
        [] { static_cast<void>(revival_probability(0, -1)); },
        "probability should reject negative TTD");
    require_out_of_range(
        [] { static_cast<void>(revival_probability(0, 301)); },
        "probability should reject TTD above capacity");
}

void test_inclusive_lag()
{
    require(lag(1, 1) == 1, "equal actions should produce one second of ST");
    require(lag(10, 20) == 11, "successful lag should be inclusive");
    require(lag(60, 60) == 1, "last equal actions should remain inclusive");
    require(lag(20, 10) == -9, "lag arithmetic should preserve action order");

    require_out_of_range(
        [] { static_cast<void>(lag(0, 1)); },
        "drop action zero should be rejected");
    require_out_of_range(
        [] { static_cast<void>(lag(1, 0)); },
        "check action zero should be rejected");
    require_out_of_range(
        [] { static_cast<void>(lag(61, 1)); },
        "drop action above 60 should be rejected");
    require_out_of_range(
        [] { static_cast<void>(lag(1, 61)); },
        "check action above 60 should be rejected");
}

} // namespace

int main()
{
    try {
        test_constant_products();
        test_default_construction();
        test_solver_kind_mapping();
        test_revival_eligibility_boundaries();
        test_revival_probability_boundaries();
        test_revival_surface_exhaustively();
        test_scalar_coordinate_validation();
        test_inclusive_lag();

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
