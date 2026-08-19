#include "dth.hpp"
#include "highs_backend.hpp"
#include "storage/durable_store.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <cerrno>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

// Inclusive lag is not part of the public Section 2 interface yet.
int lag(int drop, int check);

namespace {

using dth::survives_injection;
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

template <typename Function>
void require_exception(Function&& function, const char* const message)
{
    bool threw = false;
    try {
        function();
    } catch (const std::exception&) {
        threw = true;
    }
    require(threw, message);
}

class TemporaryDirectory {
public:
    TemporaryDirectory()
    {
        const auto stamp =
            std::chrono::high_resolution_clock::now()
                .time_since_epoch()
                .count();
        path_ = std::filesystem::temp_directory_path()
            / ("dth-cpp-section7-" + std::to_string(stamp));
        if (!std::filesystem::create_directory(path_)) {
            throw std::runtime_error("could not create Section 7 test directory");
        }
    }

    ~TemporaryDirectory()
    {
        std::error_code ignored;
        std::filesystem::remove_all(path_, ignored);
    }

    TemporaryDirectory(const TemporaryDirectory&) = delete;
    TemporaryDirectory& operator=(const TemporaryDirectory&) = delete;

    const std::filesystem::path& path() const noexcept
    {
        return path_;
    }

private:
    std::filesystem::path path_;
};

constexpr std::uint64_t kSection7ProfileCount = 17'011;
constexpr dth::ClassId kSection7ClassCount = 1'000;
constexpr std::size_t kCrashValueIndex = 17;
constexpr double kCrashValue = 91.25;
constexpr std::uint8_t kCrashSolverKind = 2;
constexpr int kCrashExitCode = 73;

[[noreturn]] void run_section7_crash_child(
    const std::filesystem::path& output_dir)
{
    dth::DurableStores stores = dth::open_resume(
        output_dir,
        kSection7ProfileCount,
        kSection7ClassCount);
    stores.values[kCrashValueIndex] = kCrashValue;
    stores.solver_kind[kCrashValueIndex] = kCrashSolverKind;
    stores.values.flush();
    stores.solver_kind.flush();
    std::_Exit(kCrashExitCode);
}

int run_section7_crash_process(
    const std::filesystem::path& test_executable,
    const std::filesystem::path& output_dir)
{
#ifdef _WIN32
    const std::filesystem::path absolute_executable =
        std::filesystem::absolute(test_executable);
    std::wstring command_line =
        L"\"" + absolute_executable.wstring()
        + L"\" --section7-crash-child \"" + output_dir.wstring() + L"\"";
    std::vector<wchar_t> mutable_command(
        command_line.begin(),
        command_line.end());
    mutable_command.push_back(L'\0');

    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    PROCESS_INFORMATION process{};
    if (!CreateProcessW(
            absolute_executable.c_str(),
            mutable_command.data(),
            nullptr,
            nullptr,
            FALSE,
            0,
            nullptr,
            nullptr,
            &startup,
            &process)) {
        throw std::system_error(
            static_cast<int>(GetLastError()),
            std::system_category(),
            "could not start Section 7 crash child");
    }
    CloseHandle(process.hThread);

    if (WaitForSingleObject(process.hProcess, INFINITE) != WAIT_OBJECT_0) {
        const DWORD error = GetLastError();
        CloseHandle(process.hProcess);
        throw std::system_error(
            static_cast<int>(error),
            std::system_category(),
            "could not wait for Section 7 crash child");
    }

    DWORD exit_code = 0;
    if (!GetExitCodeProcess(process.hProcess, &exit_code)) {
        const DWORD error = GetLastError();
        CloseHandle(process.hProcess);
        throw std::system_error(
            static_cast<int>(error),
            std::system_category(),
            "could not read Section 7 crash-child status");
    }
    CloseHandle(process.hProcess);
    return static_cast<int>(exit_code);
#else
    const std::filesystem::path absolute_executable =
        std::filesystem::absolute(test_executable);
    const pid_t child = ::fork();
    if (child == -1) {
        throw std::system_error(
            errno,
            std::generic_category(),
            "could not fork Section 7 crash child");
    }
    if (child == 0) {
        ::execl(
            absolute_executable.c_str(),
            absolute_executable.c_str(),
            "--section7-crash-child",
            output_dir.c_str(),
            static_cast<char*>(nullptr));
        std::_Exit(127);
    }

    int status = 0;
    while (::waitpid(child, &status, 0) == -1) {
        if (errno == EINTR) {
            continue;
        }
        throw std::system_error(
            errno,
            std::generic_category(),
            "could not wait for Section 7 crash child");
    }
    if (WIFEXITED(status)) {
        return WEXITSTATUS(status);
    }
    return -1;
#endif
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

void test_survives_injection_boundaries()
{
    require(
        survives_injection(0, 0),
        "fresh profile should be revival-eligible");
    require(
        survives_injection(239, 0),
        "largest individually survivable dose should be eligible");
    require(
        survives_injection(239, 1),
        "cumulative load exactly 300 should be eligible");
    require(
        survives_injection(0, 240),
        "zero ST at cumulative load 300 should be eligible");
    require(
        survives_injection(180, 60),
        "interior cumulative-load boundary should be eligible");

    require(
        !survives_injection(240, 0),
        "individual dose exactly 300 should be fatal");
    require(
        !survives_injection(239, 2),
        "cumulative load above 300 should be fatal");
    require(
        !survives_injection(0, 241),
        "TTD beyond the cumulative boundary should be fatal");
    require(
        !survives_injection(299, 0),
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
            const bool actual = survives_injection(st, ttd);
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
        [] { static_cast<void>(survives_injection(-1, 0)); },
        "negative ST should be rejected");
    require_out_of_range(
        [] { static_cast<void>(survives_injection(300, 0)); },
        "ST at capacity should be rejected");
    require_out_of_range(
        [] { static_cast<void>(survives_injection(0, -1)); },
        "negative TTD should be rejected");
    require_out_of_range(
        [] { static_cast<void>(survives_injection(0, 301)); },
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
            survives_injection(st, ttd),
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
            if (!survives_injection(st, ttd)) {
                require(
                    quotient_profile_id(table, st, ttd) == dead_id,
                    "fatal coordinates did not collapse to Dead(st)");
            }
        }
    }

    for (int ttd = 1; ttd < penalty; ++ttd) {
        for (int st = 0; st < capacity; ++st) {
            if (survives_injection(st, ttd)) {
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

            if (survives_injection(grown_st, ttd)) {
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

void test_class_encoding_gate()
{
    using namespace dth;

    ProfileTable table = begin_canonical_profile_table();
    const auto last_profile = static_cast<ProfileId>(
        table.profile_count - 1);
    const ClassId last_class = kCanonicalClasses - 1;

    const auto verify_pair = [&table](
                                 const ProfileId checker,
                                 const ProfileId dropper) {
        const ClassId expected =
            ClassId{checker}
                * static_cast<ClassId>(table.profile_count)
            + ClassId{dropper};
        const ClassId encoded = encode_class(table, checker, dropper);

        require(
            encoded == expected,
            "class encoding does not use checker-major row order");

        const auto [decoded_checker, decoded_dropper] =
            decode_class(table, encoded);
        require(
            decoded_checker == checker && decoded_dropper == dropper,
            "class encode/decode did not round-trip its profile pair");
    };

    verify_pair(ProfileId{0}, ProfileId{0});
    const auto [first_checker, first_dropper] =
        decode_class(table, ClassId{0});
    require(
        first_checker == ProfileId{0} && first_dropper == ProfileId{0},
        "the first class does not decode to the first profile pair");

    verify_pair(last_profile, last_profile);
    const auto [last_checker, last_dropper] =
        decode_class(table, last_class);
    require(
        last_checker == last_profile && last_dropper == last_profile,
        "the last class does not decode to the last profile pair");

    const std::array<std::pair<ProfileId, ProfileId>, 4> corners{{
        {ProfileId{0}, ProfileId{0}},
        {ProfileId{0}, last_profile},
        {last_profile, ProfileId{0}},
        {last_profile, last_profile},
    }};
    for (const auto& [checker, dropper] : corners) {
        verify_pair(checker, dropper);
    }

    constexpr std::size_t sample_count = 100'000;
    static_assert(sample_count >= 100'000);

    const ClassId profile_count =
        static_cast<ClassId>(table.profile_count);
    const ClassId sample_stride = profile_count + 1;
    ClassId sampled_class = 12'345;

    for (std::size_t sample = 0; sample < sample_count; ++sample) {
        const auto expected_checker = static_cast<ProfileId>(
            sampled_class / profile_count);
        const auto expected_dropper = static_cast<ProfileId>(
            sampled_class % profile_count);
        const auto [checker, dropper] =
            decode_class(table, sampled_class);

        require(
            checker == expected_checker && dropper == expected_dropper,
            "sampled class did not decode by quotient and remainder");
        require(
            encode_class(table, checker, dropper) == sampled_class,
            "sampled profile pair did not round-trip through class encoding");

        sampled_class =
            (sampled_class + sample_stride) % kCanonicalClasses;
    }

    const ProfileId origin_profile = quotient_profile_id(table, 0, 0);
    require(
        origin_profile == ProfileId{0}
            && encode_class(table, origin_profile, origin_profile)
                == ClassId{0},
        "canonical state (0,0,0,0) does not encode to class zero");
}

void test_potential_bucket_dag_gate()
{
    using namespace dth;

    ProfileTable table = begin_canonical_profile_table();
    finish_profile_table(table);
    build_buckets(table);
    validate_profile_edges(table);

    std::array<std::vector<ProfileId>, kMaxProfilePotential + 1>
        expected_buckets{};

    for (std::size_t profile = 0;
         profile < table.profile_count;
         ++profile) {
        const int st = table.st[profile];
        const int ttd = table.ttd[profile];
        const int rho = ttd >= 0
            ? ttd
            : static_cast<int>(kDeadRho);
        const auto expected_potential = static_cast<Potential>(st + rho);

        expected_buckets[expected_potential].push_back(
            static_cast<ProfileId>(profile));
    }

    for (std::size_t potential = 0;
         potential <= kMaxProfilePotential;
         ++potential) {
        require(
            table.buckets[potential] == expected_buckets[potential],
            "potential bucket contents differ from the quotient structure");
    }

    for (std::size_t potential = 241; potential <= 300; ++potential) {
        require(
            table.buckets[potential].empty(),
            "a structurally impossible bucket in 241..300 is nonempty");
    }

    std::size_t live_success = 0;
    std::size_t live_failure = 0;

    for (std::size_t profile = 0;
         profile < table.profile_count;
         ++profile) {
        const Potential parent_potential = table.potential[profile];
        const std::size_t row_begin = profile * kActions;
        const std::size_t row_end = row_begin + kActions;

        for (std::size_t index = row_begin; index < row_end; ++index) {
            const ChildId child = table.success_child[index];
            if (child < 0) {
                continue;
            }

            ++live_success;
            const auto child_id = static_cast<std::size_t>(child);
            require(
                table.potential[child_id] > parent_potential,
                "a live success edge has equal or lower potential");
        }

        const ChildId child = table.failure_child[profile];
        if (child >= 0) {
            ++live_failure;
            const auto child_id = static_cast<std::size_t>(child);
            require(
                table.potential[child_id] > parent_potential,
                "a live failure edge has equal or lower potential");
        }
    }

    require(
        live_success + live_failure == 1'035'541,
        "total live profile-transition count is not 1,035,541");

    ClassId total_classes = 0;
    int largest_layer_size = -1;
    std::size_t largest_layer_potential = 0;
    std::size_t largest_layer_count = 0;

    for (std::size_t potential = 0;
         potential <= kMaxClassPotential;
         ++potential) {
        const int actual = layer_size(
            table,
            static_cast<Potential>(potential));

        require(actual > 0, "a canonical class layer is empty");

        const std::size_t first = potential > kMaxProfilePotential
            ? potential - kMaxProfilePotential
            : 0;
        const std::size_t last =
            std::min(potential, kMaxProfilePotential);
        ClassId expected = 0;

        for (std::size_t checker_potential = first;
             checker_potential <= last;
             ++checker_potential) {
            const std::size_t dropper_potential =
                potential - checker_potential;
            expected +=
                static_cast<ClassId>(
                    expected_buckets[checker_potential].size())
                * static_cast<ClassId>(
                    expected_buckets[dropper_potential].size());
        }

        require(
            static_cast<ClassId>(actual) == expected,
            "layer size differs from the independent bucket product");

        total_classes += static_cast<ClassId>(actual);

        if (actual > largest_layer_size) {
            largest_layer_size = actual;
            largest_layer_potential = potential;
            largest_layer_count = 1;
        } else if (actual == largest_layer_size) {
            ++largest_layer_count;
        }
    }

    require(
        total_classes == kCanonicalClasses,
        "the 1,201 layers do not partition all canonical classes");
    require(
        largest_layer_potential == 374
            && largest_layer_size == 1'678'715
            && largest_layer_count == 1,
        "the unique largest layer is not P=374 with 1,678,715 classes");
}

void require_checkpoint_matches(
    const dth::CheckpointRecord& actual,
    const dth::CheckpointRecord& expected)
{
    require(
        actual.profile_count == expected.profile_count,
        "checkpoint profile count changed");
    require(
        actual.class_count == expected.class_count,
        "checkpoint class count changed");
    require(
        actual.completed_potential == expected.completed_potential,
        "checkpoint completed potential changed");
    require(
        actual.counters.pure == expected.counters.pure,
        "checkpoint pure counter changed");
    require(
        actual.counters.warm_support == expected.counters.warm_support,
        "checkpoint warm-support counter changed");
    require(
        actual.counters.full_support == expected.counters.full_support,
        "checkpoint full-support counter changed");
    require(
        actual.counters.linear_program == expected.counters.linear_program,
        "checkpoint linear-program counter changed");
}

void test_durable_store_gate(
    const std::filesystem::path& test_executable)
{
    using namespace dth;

    TemporaryDirectory temporary;
    const std::filesystem::path output_dir = temporary.path();
    const std::filesystem::path values_path = output_dir / "values.bin";
    const std::filesystem::path kinds_path = output_dir / "solver_kind.bin";
    const std::filesystem::path checkpoint_path =
        output_dir / "checkpoint.bin";
    CheckpointRecord initial_checkpoint{};

    {
        DurableStores stores = create_stores(
            output_dir,
            kSection7ProfileCount,
            kSection7ClassCount);
        initial_checkpoint = stores.checkpoint;

        require(stores.values.size() == 1'000, "values mapping size is wrong");
        require(
            stores.solver_kind.size() == 1'000,
            "solver-kind mapping size is wrong");
        require(
            stores.checkpoint.completed_potential
                == kInitialCompletedPotential,
            "initial checkpoint does not report potential 1201");
        require(
            stores.checkpoint.counters.pure == 0
                && stores.checkpoint.counters.warm_support == 0
                && stores.checkpoint.counters.full_support == 0
                && stores.checkpoint.counters.linear_program == 0,
            "initial checkpoint counters are not zero");

        for (std::size_t index = 0; index < stores.values.size(); ++index) {
            require(
                std::isnan(stores.values[index]),
                "a newly created value is not NaN");
            require(
                stores.solver_kind[index] == kUnsolvedKind,
                "a newly created solver kind is not 255");
        }

        stores.values[0] = 1.25;
        stores.values[499] = -7.5;
        stores.values[999] = 42.0;
        stores.solver_kind[0] = static_cast<std::uint8_t>(SolverKind::Pure);
        stores.solver_kind[499] =
            static_cast<std::uint8_t>(SolverKind::Support);
        stores.solver_kind[999] =
            static_cast<std::uint8_t>(SolverKind::LinearProgram);
        stores.values.flush();
        stores.solver_kind.flush();
    }

    require(
        std::filesystem::file_size(values_path) == 1'000 * sizeof(double),
        "values.bin does not have exactly 1,000 doubles");
    require(
        std::filesystem::file_size(kinds_path) == 1'000,
        "solver_kind.bin does not have exactly 1,000 bytes");

    require_exception(
        [&] {
            auto duplicate = create_stores(
                output_dir,
                kSection7ProfileCount,
                kSection7ClassCount);
        },
        "store creation overwrote an existing artifact");

    {
        DurableStores stores = open_resume(
            output_dir,
            kSection7ProfileCount,
            kSection7ClassCount);
        require_checkpoint_matches(stores.checkpoint, initial_checkpoint);
        require(
            stores.values[0] == 1.25
                && stores.values[499] == -7.5
                && stores.values[999] == 42.0,
            "mapped values did not survive close and reopen");
        require(
            stores.solver_kind[0]
                    == static_cast<std::uint8_t>(SolverKind::Pure)
                && stores.solver_kind[499]
                    == static_cast<std::uint8_t>(SolverKind::Support)
                && stores.solver_kind[999]
                    == static_cast<std::uint8_t>(SolverKind::LinearProgram),
            "solver kinds did not survive close and reopen");

        stores.values[0] = 1.25;
        stores.values[499] = -7.5;
        stores.values[999] = 42.0;
        stores.solver_kind[0] = static_cast<std::uint8_t>(SolverKind::Pure);
        stores.solver_kind[499] =
            static_cast<std::uint8_t>(SolverKind::Support);
        stores.solver_kind[999] =
            static_cast<std::uint8_t>(SolverKind::LinearProgram);
        stores.values.flush();
        stores.solver_kind.flush();
    }

    {
        const DurableStores stores = open_resume(
            output_dir,
            kSection7ProfileCount,
            kSection7ClassCount);
        require(
            stores.values[0] == 1.25
                && stores.values[499] == -7.5
                && stores.values[999] == 42.0,
            "idempotent value rewrite changed persisted data");
    }

    const std::uintmax_t correct_value_bytes =
        std::filesystem::file_size(values_path);
    std::filesystem::resize_file(values_path, correct_value_bytes + 1);
    require_exception(
        [&] {
            auto wrong_size = open_resume(
                output_dir,
                kSection7ProfileCount,
                kSection7ClassCount);
        },
        "resume accepted a values file with the wrong byte size");
    std::filesystem::resize_file(values_path, correct_value_bytes);

    constexpr std::streamoff config_offset = 8 + 4 + 4;
    {
        std::fstream checkpoint(
            checkpoint_path,
            std::ios::in | std::ios::out | std::ios::binary);
        require(
            static_cast<bool>(checkpoint),
            "could not open checkpoint for config corruption test");
        checkpoint.seekp(config_offset);
        checkpoint.put('X');
        checkpoint.flush();
        require(
            static_cast<bool>(checkpoint),
            "could not corrupt checkpoint config id");
    }
    require_exception(
        [&] {
            auto wrong_config = open_resume(
                output_dir,
                kSection7ProfileCount,
                kSection7ClassCount);
        },
        "resume accepted a checkpoint with the wrong config id");

    atomically_write_checkpoint(output_dir, initial_checkpoint);
    atomically_write_checkpoint(output_dir, initial_checkpoint);
    {
        const DurableStores stores = open_resume(
            output_dir,
            kSection7ProfileCount,
            kSection7ClassCount);
        require_checkpoint_matches(stores.checkpoint, initial_checkpoint);
    }

    CheckpointRecord prior_checkpoint = initial_checkpoint;
    prior_checkpoint.completed_potential = 777;
    prior_checkpoint.counters.pure = 11;
    prior_checkpoint.counters.warm_support = 22;
    prior_checkpoint.counters.full_support = 33;
    prior_checkpoint.counters.linear_program = 44;
    atomically_write_checkpoint(output_dir, prior_checkpoint);

    require(
        run_section7_crash_process(test_executable, output_dir)
            == kCrashExitCode,
        "Section 7 crash child did not stop at the fault-injection point");

    {
        const DurableStores stores = open_resume(
            output_dir,
            kSection7ProfileCount,
            kSection7ClassCount);
        require_checkpoint_matches(stores.checkpoint, prior_checkpoint);
        require(
            stores.values[kCrashValueIndex] == kCrashValue,
            "crash child did not durably flush its mapped value");
        require(
            stores.solver_kind[kCrashValueIndex] == kCrashSolverKind,
            "crash child did not durably flush its routing byte");
    }

    require(
        !std::filesystem::exists(output_dir / "checkpoint.tmp"),
        "atomic checkpoint replacement left checkpoint.tmp behind");
}

void test_transition_value_assembly_gate()
{
    using namespace dth;

    constexpr std::size_t profile_count = 64;
    constexpr ProfileId checker{1};
    constexpr ProfileId dropper{2};
    constexpr ProfileId first_success_child{3};
    constexpr ProfileId failure_child{63};
    constexpr double default_stored_value = 0.9375;
    constexpr double failure_stored_value = 0.5;
    constexpr double revival_chance = 0.25;
    constexpr double expected_failed = 0.625;

    ProfileTable table{};
    table.profile_count = profile_count;
    table.revival.assign(profile_count, 0.0);
    table.success_child.assign(
        profile_count * kActions,
        ChildId{-1});
    table.failure_child.assign(profile_count, ChildId{-1});

    const std::size_t checker_index = static_cast<std::size_t>(checker);
    const std::size_t success_row = checker_index * kActions;
    table.revival[checker_index] = revival_chance;
    table.failure_child[checker_index] =
        static_cast<ChildId>(failure_child);

    std::array<double, kActions> expected_success{};
    for (std::size_t action = 0; action < kActions; ++action) {
        const auto child_profile = static_cast<ProfileId>(
            static_cast<std::size_t>(first_success_child) + action);
        table.success_child[success_row + action] =
            static_cast<ChildId>(child_profile);

        const double stored_value =
            -0.75 + static_cast<double>(action) / 40.0;
        expected_success[action] = -stored_value;
    }

    TemporaryDirectory temporary;
    MappedArray<double> values = MappedArray<double>::create(
        temporary.path() / "section8-values.bin",
        profile_count * profile_count,
        default_stored_value);

    for (std::size_t action = 0; action < kActions; ++action) {
        const auto child_profile = static_cast<ProfileId>(
            table.success_child[success_row + action]);
        const ClassId child_class = encode_class(
            table,
            dropper,
            child_profile);
        values[static_cast<std::size_t>(child_class)] =
            -expected_success[action];
    }

    const ClassId failed_child_class = encode_class(
        table,
        dropper,
        failure_child);
    values[static_cast<std::size_t>(failed_child_class)] =
        failure_stored_value;

    TransitionValues transitions = assemble_transition_values(
        table,
        values,
        checker,
        dropper);

    for (std::size_t action = 0; action < kActions; ++action) {
        require(
            transitions.success[action] == expected_success[action],
            "success continuation did not read the role-swapped child class");
    }
    require(
        transitions.failed == expected_failed,
        "probabilistic failure does not match the hand-computed expectation");

    std::size_t checked_cells = 0;
    for (int drop = 0; drop < static_cast<int>(kActions); ++drop) {
        for (int check = 0; check < static_cast<int>(kActions); ++check) {
            const double expected = check < drop
                ? expected_failed
                : expected_success[static_cast<std::size_t>(check - drop)];
            require(
                matrix_cell(transitions, drop, check) == expected,
                "implicit matrix cell differs from literal action expansion");
            ++checked_cells;
        }
    }
    require(
        checked_cells == kActions * kActions,
        "the Section 8 gate did not check all 3,600 matrix cells");

    for (int action = 0; action < static_cast<int>(kActions); ++action) {
        require(
            matrix_cell(transitions, action, action)
                == expected_success[0],
            "a main-diagonal cell does not read success[0]");
    }
    require(
        matrix_cell(
            transitions,
            0,
            static_cast<int>(kActions) - 1)
            == expected_success[kActions - 1],
        "the top-right cell does not read success[59]");
    for (int drop = 1; drop < static_cast<int>(kActions); ++drop) {
        for (int check = 0; check < drop; ++check) {
            require(
                matrix_cell(transitions, drop, check) == expected_failed,
                "a below-diagonal cell does not read the common failure value");
        }
    }

    ProfileTable terminal_table = table;
    constexpr std::size_t terminal_success_index = 17;
    terminal_table.success_child[
        success_row + terminal_success_index] = ChildId{-1};
    terminal_table.failure_child[checker_index] = ChildId{-1};

    TransitionValues terminal = assemble_transition_values(
        terminal_table,
        values,
        checker,
        dropper);
    require(
        terminal.success[terminal_success_index] == 1.0,
        "terminal success is not a Dropper payoff of +1");
    require(
        terminal.failed == 1.0,
        "terminal failure is not a Dropper payoff of +1");

    constexpr std::size_t unsolved_action = 23;
    const auto unsolved_child = static_cast<ProfileId>(
        table.success_child[success_row + unsolved_action]);
    const ClassId unsolved_class = encode_class(
        table,
        dropper,
        unsolved_child);
    values[static_cast<std::size_t>(unsolved_class)] =
        std::numeric_limits<double>::quiet_NaN();

    require_logic_error(
        [&] {
            static_cast<void>(assemble_transition_values(
                table,
                values,
                checker,
                dropper));
        },
        "an unsolved NaN child was not rejected immediately");
}

void test_policy_certification_gate()
{
    using namespace dth;

    constexpr double tolerance = 1e-12;

    const auto total_mass = [](const Policy& policy) {
        double total = 0.0;
        for (const double mass : policy.mass) {
            total += mass;
        }
        return total;
    };

    // A uniform positive policy normalizes to 1/60 per action.
    {
        Policy raw{};
        raw.mass.fill(2.0);
        const auto normalized = normalize_policy(raw, 0.0);
        require(
            normalized.has_value(),
            "a uniform positive policy was rejected");
        for (const double mass : normalized->mass) {
            require(
                std::abs(mass - 1.0 / 60.0) <= tolerance,
                "uniform normalization did not divide mass evenly");
        }
        require(
            std::abs(total_mass(*normalized) - 1.0) <= tolerance,
            "normalized policy mass does not sum to one");
    }

    // Negative rounding noise inside the limit is clipped, not rejected.
    {
        Policy raw{};
        raw.mass[0] = 1.0;
        raw.mass[1] = 1.0;
        raw.mass[2] = -1e-15;
        const auto normalized = normalize_policy(raw, 1e-9);
        require(
            normalized.has_value(),
            "negative rounding noise inside the limit was rejected");
        require(
            normalized->mass[2] == 0.0,
            "negative rounding noise was not clipped to zero");
        require(
            std::abs(normalized->mass[0] - 0.5) <= tolerance,
            "clipping disturbed the surviving mass ratio");
    }

    // Materially negative, non-finite, and zero-total policies are rejected.
    {
        Policy materially_negative{};
        materially_negative.mass[0] = 1.0;
        materially_negative.mass[1] = -0.5;
        require(
            !normalize_policy(materially_negative, 1e-9).has_value(),
            "materially negative mass was accepted");

        Policy not_a_number{};
        not_a_number.mass[0] = 1.0;
        not_a_number.mass[1] = std::numeric_limits<double>::quiet_NaN();
        require(
            !normalize_policy(not_a_number, 1e-9).has_value(),
            "a NaN mass was accepted");

        Policy unbounded{};
        unbounded.mass[0] = std::numeric_limits<double>::infinity();
        require(
            !normalize_policy(unbounded, 1e-9).has_value(),
            "an infinite mass was accepted");

        const Policy zero{};
        require(
            !normalize_policy(zero, 1e-9).has_value(),
            "a zero-total policy was accepted");
    }

    // A constant matrix has that constant as its value under any policy pair.
    {
        TransitionValues constant{};
        constant.success.fill(0.25);
        constant.failed = 0.25;

        Policy raw{};
        raw.mass.fill(1.0);

        const auto certified = certify(constant, raw, raw, 0.0);
        require(
            certified.has_value(),
            "a constant matrix failed to certify");
        require(
            std::abs(certified->certificate.lower - 0.25) <= tolerance,
            "constant matrix lower bound is not the constant");
        require(
            std::abs(certified->certificate.upper - 0.25) <= tolerance,
            "constant matrix upper bound is not the constant");
        require(
            certified->certificate.gap <= kSaddleTolerance,
            "constant matrix gap exceeded the acceptance tolerance");
        require(
            std::abs(certified->certificate.midpoint - 0.25) <= tolerance,
            "constant matrix midpoint is not the constant");
        require(
            std::abs(total_mass(certified->drop) - 1.0) <= tolerance
                && std::abs(total_mass(certified->check) - 1.0) <= tolerance,
            "certify did not return normalized policies");
    }

    // Asymmetric matrix whose saddle sits at unequal row and column indices.
    // Transposing the two security computations moves the upper bound to 0.5
    // and the gap to 0.75, and negating the payoffs moves the midpoint to
    // +0.25, so this single case pins both orientation and sign.
    {
        TransitionValues offset_saddle{};
        offset_saddle.success.fill(-0.25);
        offset_saddle.failed = 0.5;

        Policy drop{};
        drop.mass[0] = 1.0;
        Policy check{};
        check.mass[kActions - 1] = 1.0;

        const auto certified = certify(offset_saddle, drop, check, 0.0);
        require(
            certified.has_value(),
            "the offset pure saddle failed to certify");
        require(
            std::abs(certified->certificate.midpoint + 0.25) <= tolerance,
            "the offset saddle value is wrong, sign, or transposed");
        require(
            certified->certificate.gap <= kSaddleTolerance,
            "the offset saddle produced a nonzero gap");
    }

    // Asymmetric matrix with a strictly increasing success row. Any shift in
    // the Toeplitz lag index changes the certified value.
    {
        TransitionValues rising_saddle{};
        for (std::size_t action = 0; action < kActions; ++action) {
            rising_saddle.success[action] =
                -0.5 + 0.01 * static_cast<double>(action);
        }
        rising_saddle.failed = -1.0;

        Policy drop{};
        drop.mass[0] = 1.0;
        Policy check{};
        check.mass[0] = 1.0;

        const auto certified = certify(rising_saddle, drop, check, 0.0);
        require(
            certified.has_value(),
            "the rising pure saddle failed to certify");
        require(
            std::abs(certified->certificate.midpoint + 0.5) <= tolerance,
            "the rising saddle value is not the first success entry");
    }

    // Matching pennies embedded in the first two actions. Actions 2..59 are
    // dominated for both players, so the value is 0 at the half-half mixture.
    {
        TransitionValues pennies{};
        pennies.success.fill(1.0);
        pennies.success[1] = -1.0;
        pennies.failed = -1.0;

        Policy mixed{};
        mixed.mass[0] = 7.0;
        mixed.mass[1] = 7.0;

        const auto certified = certify(pennies, mixed, mixed, 0.0);
        require(
            certified.has_value(),
            "embedded matching pennies failed to certify");
        require(
            std::abs(certified->certificate.midpoint) <= tolerance,
            "the matching pennies value is not zero");
        require(
            certified->certificate.gap <= kSaddleTolerance,
            "matching pennies produced a nonzero gap");
        require(
            std::abs(certified->drop.mass[0] - 0.5) <= tolerance
                && std::abs(certified->drop.mass[1] - 0.5) <= tolerance,
            "certify did not normalize the unscaled matching pennies mixture");

        // A pure pair against the same matrix is exploitable on both sides.
        Policy pure{};
        pure.mass[0] = 1.0;
        const auto exploitable = certify(pennies, pure, pure, 0.0);
        require(
            !exploitable.has_value(),
            "an exploitable pure pair on matching pennies was certified");
    }

    // A rejected normalization propagates as a rejected certificate.
    {
        TransitionValues constant{};
        constant.success.fill(0.0);
        constant.failed = 0.0;

        Policy raw{};
        raw.mass.fill(1.0);
        const Policy zero{};
        require(
            !certify(constant, zero, raw, 0.0).has_value()
                && !certify(constant, raw, zero, 0.0).has_value(),
            "certify accepted a policy that normalization rejected");
    }
}

dth::PureSaddleScan brute_force_pure_saddle_scan(
    const dth::TransitionValues& transitions)
{
    using namespace dth;

    PureSaddleScan result{
        -std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity(),
        0,
        0,
    };

    for (std::size_t drop = 0; drop < kActions; ++drop) {
        double row_minimum = std::numeric_limits<double>::infinity();
        for (std::size_t check = 0; check < kActions; ++check) {
            row_minimum = std::min(
                row_minimum,
                matrix_cell(
                    transitions,
                    static_cast<int>(drop),
                    static_cast<int>(check)));
        }
        if (row_minimum > result.maximin) {
            result.maximin = row_minimum;
            result.best_drop = drop;
        }
    }

    for (std::size_t check = 0; check < kActions; ++check) {
        double column_maximum = -std::numeric_limits<double>::infinity();
        for (std::size_t drop = 0; drop < kActions; ++drop) {
            column_maximum = std::max(
                column_maximum,
                matrix_cell(
                    transitions,
                    static_cast<int>(drop),
                    static_cast<int>(check)));
        }
        if (column_maximum < result.minimax) {
            result.minimax = column_maximum;
            result.best_check = check;
        }
    }

    return result;
}

void test_pure_saddle_reduction_gate()
{
    using namespace dth;

    const auto same_bits = [](const double left, const double right) {
        return std::bit_cast<std::uint64_t>(left)
            == std::bit_cast<std::uint64_t>(right);
    };

    // The fixed LCG and integer grid make this corpus deterministic across
    // standard-library implementations while still covering the full payoff
    // interval, duplicates, and exact zero.
    std::uint64_t random_state = 0x8e9d'5aaa'27c1'4f3bULL;
    const auto next_random_value = [&random_state] {
        random_state = random_state * 6'364'136'223'846'793'005ULL
            + 1'442'695'040'888'963'407ULL;
        constexpr std::uint64_t value_count = 2'000'001ULL;
        const auto sample = static_cast<std::int64_t>(
            (random_state >> 32U) % value_count)
            - 1'000'000LL;
        return static_cast<double>(sample) / 1'000'000.0;
    };

    for (std::size_t trial = 0; trial < 10'000; ++trial) {
        TransitionValues transitions{};
        for (double& success : transitions.success) {
            success = next_random_value();
        }
        transitions.failed = next_random_value();

        const PureSaddleScan reduced = scan_pure_saddle(transitions);
        const PureSaddleScan expanded =
            brute_force_pure_saddle_scan(transitions);

        require(
            same_bits(reduced.maximin, expanded.maximin),
            "O(60) maximin differs from the expanded row-minimum bound");
        require(
            same_bits(reduced.minimax, expanded.minimax),
            "O(60) minimax differs from the expanded column-maximum bound");
        require(
            reduced.best_drop == expanded.best_drop,
            "O(60) scan chose the wrong lowest-index maximizing row");
        require(
            reduced.best_check == expanded.best_check,
            "O(60) scan chose the wrong lowest-index minimizing column");
    }

    // A strict asymmetric pure saddle is accepted with the expected one-hot
    // actions and value.
    {
        TransitionValues pure{};
        pure.success.fill(-0.25);
        pure.failed = 0.5;

        const PureSaddleScan scan = scan_pure_saddle(pure);
        require(
            scan.best_drop == 0 && scan.best_check == kActions - 1,
            "pure scan chose the wrong asymmetric saddle actions");

        const auto certified = try_pure_saddle(pure);
        require(certified.has_value(), "a strict pure saddle was rejected");
        require(
            certified->drop.mass[0] == 1.0
                && certified->check.mass[kActions - 1] == 1.0,
            "pure saddle policies are not one-hot at the selected actions");
        require(
            certified->certificate.midpoint == -0.25,
            "pure saddle certificate has the wrong value");
    }

    // Matching pennies in the first two actions has a strict pure-saddle gap
    // and must advance to a mixed-strategy route.
    {
        TransitionValues mixed{};
        mixed.success.fill(1.0);
        mixed.success[1] = -1.0;
        mixed.failed = -1.0;

        const PureSaddleScan scan = scan_pure_saddle(mixed);
        require(
            scan.maximin == -1.0 && scan.minimax == 1.0,
            "embedded matching pennies has the wrong pure security bounds");
        require(
            !try_pure_saddle(mixed).has_value(),
            "a genuinely mixed game was accepted as a pure saddle");
    }

    // Exact ties must retain the lowest row and column indices.
    {
        TransitionValues tied{};
        tied.success.fill(0.125);
        tied.failed = 0.125;

        const PureSaddleScan scan = scan_pure_saddle(tied);
        require(
            scan.maximin == 0.125 && scan.minimax == 0.125,
            "constant tied matrix has the wrong pure bounds");
        require(
            scan.best_drop == 0 && scan.best_check == 0,
            "exact ties did not retain the lowest action indices");

        const auto certified = try_pure_saddle(tied);
        require(certified.has_value(), "an exact tied saddle was rejected");
        require(
            certified->drop.mass[0] == 1.0
                && certified->check.mass[0] == 1.0,
            "exact tied saddle did not use the lowest-index actions");
    }

    // Row zero contains only successes. A low failure value must not leak into
    // that row's minimum.
    {
        TransitionValues all_success_row{};
        all_success_row.success.fill(0.5);
        all_success_row.failed = -0.75;

        const PureSaddleScan scan = scan_pure_saddle(all_success_row);
        require(
            scan.maximin == 0.5 && scan.minimax == 0.5,
            "failure value leaked into the all-success boundary row");
        require(
            scan.best_drop == 0 && scan.best_check == 0,
            "all-success boundary row chose the wrong saddle actions");
        require(
            try_pure_saddle(all_success_row).has_value(),
            "all-success boundary layout was rejected");
    }

    // Column 59 contains only successes. A high failure value must affect every
    // earlier column but not the final column.
    {
        TransitionValues failure_heavy_columns{};
        failure_heavy_columns.success.fill(-0.5);
        failure_heavy_columns.failed = 0.75;

        const PureSaddleScan scan = scan_pure_saddle(failure_heavy_columns);
        require(
            scan.maximin == -0.5 && scan.minimax == -0.5,
            "failure value leaked into the all-success boundary column");
        require(
            scan.best_drop == 0 && scan.best_check == kActions - 1,
            "failure-heavy boundary layout chose the wrong saddle actions");
        require(
            try_pure_saddle(failure_heavy_columns).has_value(),
            "failure-heavy boundary layout was rejected");
    }
}

void test_square_support_equalizer_gate()
{
    using namespace dth;

    constexpr double numeric_tolerance = 1e-8;
    HighsBackend backend;
    MatrixScratch scratch{};

    std::array<std::size_t, kActions> full_support{};
    for (std::size_t action = 0; action < kActions; ++action) {
        full_support[action] = action;
    }

    // Empty candidate supports do not submit an invalid zero-dimensional model.
    {
        TransitionValues constant{};
        constant.success.fill(0.0);
        constant.failed = 0.0;
        const std::array<std::size_t, 0> empty{};
        require(
            !try_support(
                constant,
                empty,
                full_support,
                backend,
                scratch).has_value(),
            "an empty candidate support produced a solution");
    }

    // Matching pennies occupies the first two actions. The longer Dropper
    // guess proves that try_support trims both candidates to the same k and
    // embeds the two returned masses at the selected literal actions.
    {
        TransitionValues pennies{};
        pennies.success.fill(1.0);
        pennies.success[1] = -1.0;
        pennies.failed = -1.0;
        const std::array<std::size_t, 3> drop_indices{0, 1, 59};
        const std::array<std::size_t, 2> check_indices{0, 1};

        const auto certified = try_support(
            pennies,
            drop_indices,
            check_indices,
            backend,
            scratch);
        require(
            certified.has_value(),
            "the matching-pennies support was rejected");
        require(
            certified->certificate.gap <= kSaddleTolerance
                && std::abs(certified->certificate.midpoint) <= numeric_tolerance,
            "the matching-pennies support returned an invalid certificate");
        require(
            std::abs(certified->drop.mass[0] - 0.5) <= numeric_tolerance
                && std::abs(certified->drop.mass[1] - 0.5)
                    <= numeric_tolerance
                && std::abs(certified->check.mass[0] - 0.5)
                    <= numeric_tolerance
                && std::abs(certified->check.mass[1] - 0.5)
                    <= numeric_tolerance
                && certified->drop.mass[59] == 0.0,
            "the matching-pennies masses were embedded at the wrong actions");
    }

    // A genuinely asymmetric all-action game has geometric, oppositely
    // skewed policies. This pins the transpose/orientation of both systems.
    {
        TransitionValues asymmetric{};
        asymmetric.success.fill(0.0);
        asymmetric.success[0] = 0.5;
        asymmetric.failed = 0.025;

        const auto certified = try_support(
            asymmetric,
            full_support,
            full_support,
            backend,
            scratch);
        require(
            certified.has_value(),
            "the asymmetric full-support game was rejected");

        double geometric_sum = 0.0;
        double power = 1.0;
        for (std::size_t action = 0; action < kActions; ++action) {
            geometric_sum += power;
            power *= 0.95;
        }
        const double expected_value = 0.5 / geometric_sum;
        require(
            certified->certificate.gap <= kSaddleTolerance
                && std::abs(
                       certified->certificate.midpoint - expected_value)
                    <= numeric_tolerance,
            "the asymmetric full-support value is wrong");
        require(
            certified->drop.mass.front() < certified->drop.mass.back()
                && certified->check.mass.front()
                    > certified->check.mass.back(),
            "the asymmetric policies have the wrong orientation");
    }

    // The selected matrix is [[0,0],[1,1]]. Its equalizer system is singular
    // and inconsistent with total probability one.
    {
        TransitionValues singular_infeasible{};
        singular_infeasible.success.fill(0.0);
        singular_infeasible.success[1] = 1.0;
        singular_infeasible.failed = 1.0;
        const std::array<std::size_t, 2> drop_indices{0, 1};
        const std::array<std::size_t, 2> check_indices{0, 2};
        require(
            !try_support(
                singular_infeasible,
                drop_indices,
                check_indices,
                backend,
                scratch).has_value(),
            "a singular infeasible support produced a solution");
    }

    // HiGHS may choose any point from a singular feasible equalizer. A
    // constant full matrix makes every such normalized point certifiable.
    {
        TransitionValues singular_feasible{};
        singular_feasible.success.fill(0.125);
        singular_feasible.failed = 0.125;
        const std::array<std::size_t, 3> indices{3, 17, 41};

        const auto certified = try_support(
            singular_feasible,
            indices,
            indices,
            backend,
            scratch);
        require(
            certified.has_value(),
            "a singular feasible support was rejected");
        require(
            certified->certificate.gap <= kSaddleTolerance
                && std::abs(certified->certificate.midpoint - 0.125)
                    <= numeric_tolerance,
            "the singular feasible candidate was not fully certified");
    }

    // A locally feasible singular support is still rejected when an action
    // outside that support exploits it in the complete 60-action matrix.
    {
        TransitionValues locally_feasible{};
        locally_feasible.success.fill(-1.0);
        locally_feasible.success[0] = 0.0;
        locally_feasible.success[1] = 0.0;
        locally_feasible.failed = 0.0;
        const std::array<std::size_t, 2> indices{0, 1};
        require(
            !try_support(
                locally_feasible,
                indices,
                indices,
                backend,
                scratch).has_value(),
            "a locally equalized but globally exploitable support was accepted");
    }

    // Without nonnegative bounds this support's unique equalizer masses would
    // be (-1,2) and (2,-1). HiGHS must therefore report no feasible candidate.
    // Pin the exact normalization boundary independently as well.
    {
        TransitionValues negative_equalizer{};
        negative_equalizer.success.fill(0.0);
        negative_equalizer.success[1] = 0.5;
        negative_equalizer.failed = -1.0;
        const std::array<std::size_t, 2> indices{0, 1};
        require(
            !try_support(
                negative_equalizer,
                indices,
                indices,
                backend,
                scratch).has_value(),
            "a support requiring negative probability was accepted");

        Policy below_limit{};
        below_limit.mass[0] = 1.0;
        below_limit.mass[1] = -1.000001e-10;
        require(
            !normalize_policy(below_limit, 1e-10).has_value(),
            "a mass below the Section 12 negative limit was accepted");
    }

    // A valid 1x1 equalizer is not enough: this deliberately wrong support is
    // exploitable elsewhere and must fail the independent full certificate.
    {
        TransitionValues matching{};
        matching.success.fill(-1.0);
        matching.success[0] = 1.0;
        matching.failed = -1.0;
        const std::array<std::size_t, 1> wrong_support{0};
        require(
            !try_support(
                matching,
                wrong_support,
                wrong_support,
                backend,
                scratch).has_value(),
            "an uncertified wrong support was accepted");
    }

    // Frozen canonical class 0 from complete_full_v1. The reference artifact
    // accepted its nonsingular all-60 equalizer with every mass positive.
    {
        TransitionValues real{};
        real.success = {
            -0.08374241640910982,
            -0.07824378973646026,
            -0.07276852007830821,
            -0.06729992882635064,
            -0.06184921664221913,
            -0.05640541936411085,
            -0.05098028343761779,
            -0.045559470336475213,
            -0.04015874260475749,
            -0.034759552320429335,
            -0.029378314849305195,
            -0.024001318107639155,
            -0.018641909947997463,
            -0.013281830188160992,
            -0.007934307021762906,
            -0.002593390575041668,
            0.002728933415204883,
            0.008051765093601936,
            0.01335782930865187,
            0.0186701704220803,
            0.02396193587599028,
            0.02925442261468289,
            0.034531088801633136,
            0.039808979516920195,
            0.045072871838972337,
            0.05033988974679343,
            0.05559135558415802,
            0.0608458097708978,
            0.0660862869134648,
            0.07134829255576977,
            0.07660831658518162,
            0.0818270846783182,
            0.08703388199092033,
            0.092241335839917,
            0.09743926400715172,
            0.10263756771046761,
            0.10782451871974238,
            0.11301351083332078,
            0.11818912670266057,
            0.12337335903026528,
            0.12854119286066434,
            0.1337083379009194,
            0.13886283026145219,
            0.14402075657608537,
            0.1491758185527881,
            0.15432293315910023,
            0.15945550432607586,
            0.16459207243968957,
            0.16971383928356784,
            0.1748448271592195,
            0.17995740012712844,
            0.18507380095016018,
            0.1901764349117717,
            0.19528156600736862,
            0.20037508980607405,
            0.2054700775030551,
            0.2105520070887648,
            0.2156368702477291,
            0.22070896748955432,
            0.22580849479758416,
        };
        real.failed = 0.15500017346065081;

        const auto certified = try_support(
            real,
            full_support,
            full_support,
            backend,
            scratch);
        require(
            certified.has_value(),
            "the frozen real full-support tuple was rejected");
        require(
            certified->certificate.gap <= kSaddleTolerance,
            "the frozen real tuple returned an uncertified value");
        require(
            std::abs(
                certified->certificate.midpoint - 0.08985007280951046)
                <= kSaddleTolerance,
            "the frozen real tuple disagrees with its stored artifact value");
    }
}

} // namespace

int main(const int argc, char* argv[])
{
    if (argc == 3
        && std::string_view(argv[1]) == "--section7-crash-child") {
        try {
            run_section7_crash_child(argv[2]);
        } catch (const std::exception& error) {
            std::cerr << "Section 7 crash child failure: " << error.what() << '\n';
            return 2;
        }
    }

    try {
        test_constant_products();
        test_default_construction();
        test_solver_kind_mapping();
        test_survives_injection_boundaries();
        test_revival_probability_boundaries();
        test_revival_surface_exhaustively();
        test_scalar_coordinate_validation();
        test_inclusive_lag();
        test_failure_fatal_quotient_exhaustively();
        test_profile_transition_table_exhaustively();
        test_class_encoding_gate();
        test_potential_bucket_dag_gate();
        test_durable_store_gate(argv[0]);
        test_transition_value_assembly_gate();
        test_policy_certification_gate();
        test_pure_saddle_reduction_gate();
        test_square_support_equalizer_gate();

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
