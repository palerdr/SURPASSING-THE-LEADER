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

template <typename Function>
void require_logic_error(Function&& function, const char* const message)
{
    bool threw = false;
    try {
        function();
    } catch (const std::logic_error&) {
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

void test_failure_fatal_quotient_exhaustively()
{
    using namespace dth;

    const ProfileTable table = begin_canonical_profile_table();

    require(
        table.profile_count == kCanonicalProfiles,
        "canonical profile count is incorrect");
    require(
        table.st.size() == kCanonicalProfiles,
        "canonical ST storage has the wrong size");
    require(
        table.ttd.size() == kCanonicalProfiles,
        "canonical TTD storage has the wrong size");
    require(
        table.alive_id.size() == (kCapacity + 1) * kCapacity,
        "canonical alive lookup has the wrong size");

    std::size_t eligible_profiles = 0;
    for (std::size_t id = 0; id < table.profile_count; ++id) {
        if (table.ttd[id] >= 0) {
            ++eligible_profiles;
        }
    }
    require(
        eligible_profiles == kAliveProfiles,
        "canonical enumeration did not emit exactly 16,711 eligible profiles");

    require(
        table.st.front() == 0 && table.ttd.front() == 0,
        "the first canonical profile is not (0,0)");

    for (std::size_t st = 0; st < kCapacity; ++st) {
        const std::size_t id = kDeadProfileBase + st;
        require(
            table.st[id] == static_cast<std::int16_t>(st),
            "dead profile ST does not match its sentinel id");
        require(
            table.ttd[id] == -1,
            "dead profile does not use the TTD sentinel");
    }

    for (std::size_t id = 0; id < kAliveProfiles; ++id) {
        const int st = table.st[id];
        const int ttd = table.ttd[id];

        require(
            revival_eligibility(st, ttd),
            "canonical eligible range contains a fatal profile");
        require(
            ttd == 0 || ttd >= static_cast<int>(kPenalty),
            "canonical enumeration contains an eligible TTD in 1..59");
        require(
            quotient_profile_id(table, st, ttd)
                == static_cast<ProfileId>(id),
            "canonical eligible representative does not round-trip to its id");
    }

    constexpr int capacity = static_cast<int>(kCapacity);
    constexpr int penalty = static_cast<int>(kPenalty);

    for (int st = 0; st < capacity; ++st) {
        const auto dead_id = static_cast<ProfileId>(
            kDeadProfileBase + static_cast<std::size_t>(st));

        for (int ttd = 0; ttd <= capacity; ++ttd) {
            if (!revival_eligibility(st, ttd)) {
                require(
                    quotient_profile_id(table, st, ttd) == dead_id,
                    "fatal coordinates did not collapse to Dead(st)");
            }
        }
    }

    for (int ttd = 1; ttd < penalty; ++ttd) {
        for (int st = 0; st < capacity; ++st) {
            if (revival_eligibility(st, ttd)) {
                require_logic_error(
                    [&table, st, ttd] {
                        static_cast<void>(
                            quotient_profile_id(table, st, ttd));
                    },
                    "eligible off-domain TTD in 1..59 did not fail");
            }
        }
    }
}

void test_profile_transition_table_exhaustively()
{
    using namespace dth;

    ProfileTable table = begin_canonical_profile_table();
    finish_profile_table(table);

    require(
        table.potential.size() == table.profile_count,
        "potential storage has the wrong size");
    require(
        table.success_child.size() == table.profile_count * kActions,
        "success-child storage has the wrong size");
    require(
        table.failure_child.size() == table.profile_count,
        "failure-child storage has the wrong size");

    std::size_t live_success_entries = 0;
    std::size_t live_failure_entries = 0;

    constexpr int capacity = static_cast<int>(kCapacity);
    constexpr int dead_rho = static_cast<int>(kDeadRho);

    for (std::size_t id = 0; id < table.profile_count; ++id) {
        const int st = table.st[id];
        const int ttd = table.ttd[id];
        const bool alive = ttd >= 0;
        const auto expected_potential = static_cast<Potential>(
            alive ? st + ttd : st + dead_rho);

        require(
            table.potential[id] <= kMaxProfilePotential,
            "profile potential lies outside 0..600");
        require(
            table.potential[id] == expected_potential,
            "profile potential does not match its quotient representative");

        for (std::size_t action = 0; action < kActions; ++action) {
            const std::size_t index = id * kActions + action;
            const ChildId child = table.success_child[index];
            const int lag = static_cast<int>(action + 1);
            const int grown_st = st + lag;

            require(
                child == ChildId{-1}
                    || (child >= 0
                        && static_cast<std::size_t>(child)
                            < table.profile_count),
                "success child is neither terminal nor a valid profile id");

            if (grown_st >= capacity) {
                require(
                    child == ChildId{-1},
                    "fatal successful check does not use the terminal sentinel");
                continue;
            }

            require(
                child >= 0,
                "live successful check incorrectly uses the terminal sentinel");
            ++live_success_entries;

            const std::size_t child_id = static_cast<std::size_t>(child);
            require(
                table.st[child_id] == grown_st,
                "success child ST did not increase by the inclusive lag");

            if (!alive) {
                const auto expected_child = static_cast<ChildId>(
                    kDeadProfileBase
                    + static_cast<std::size_t>(grown_st));
                require(
                    child == expected_child && table.ttd[child_id] == -1,
                    "dead success did not preserve the dead sentinel");
                continue;
            }

            if (revival_eligibility(grown_st, ttd)) {
                require(
                    table.ttd[child_id] == ttd,
                    "alive success did not preserve TTD");
                require(
                    child
                        == static_cast<ChildId>(
                            quotient_profile_id(table, grown_st, ttd)),
                    "alive success did not map to its quotient representative");
            } else {
                const auto expected_child = static_cast<ChildId>(
                    kDeadProfileBase
                    + static_cast<std::size_t>(grown_st));
                require(
                    child == expected_child && table.ttd[child_id] == -1,
                    "alive success did not cross to the correct dead sentinel");
            }
        }

        const ChildId failure_child = table.failure_child[id];
        require(
            failure_child == ChildId{-1}
                || (failure_child >= 0
                    && static_cast<std::size_t>(failure_child)
                        < table.profile_count),
            "failure child is neither terminal nor a valid profile id");

        if (alive) {
            require(
                failure_child >= 0,
                "alive profile has a terminal failure child");
            ++live_failure_entries;

            const int revived_ttd =
                ttd + st + static_cast<int>(kPenalty);
            require(
                failure_child
                    == static_cast<ChildId>(
                        quotient_profile_id(table, 0, revived_ttd)),
                "alive failure child is not the revived quotient profile");
        } else {
            require(
                failure_child == ChildId{-1},
                "dead profile does not have terminal failure");
        }
    }

    require(
        live_success_entries == 1'018'830,
        "live success-child count is not exactly 1,018,830");
    require(
        live_failure_entries == kAliveProfiles,
        "live failure-child count is not exactly 16,711");
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
        test_failure_fatal_quotient_exhaustively();
        test_profile_transition_table_exhaustively();

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
