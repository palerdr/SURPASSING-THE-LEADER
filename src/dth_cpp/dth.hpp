#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace dth {

template <typename T>
class MappedArray;

//SECTION 1: TYPE DEFINTIONS AND CONSTANTS
using ProfileId = std::uint32_t;
using ChildId = std::int32_t;
using ClassId = std::uint64_t;
using Potential = std::uint16_t;

inline constexpr std::size_t kActions = 60;
inline constexpr std::size_t kCapacity = 300;
inline constexpr std::size_t kPenalty = 60;
inline constexpr std::size_t kAliveProfiles = 16'711;
inline constexpr std::size_t kDeadProfileBase = 16'711;
inline constexpr std::size_t kCanonicalProfiles = 17'011;
inline constexpr ClassId kCanonicalClasses = 289'374'121ULL;
inline constexpr std::size_t kDeadRho = 301;
inline constexpr std::size_t kMaxProfilePotential = 600;
inline constexpr std::size_t kMaxClassPotential = 1'200;

inline constexpr double kSaddleTolerance = 1e-6;
inline constexpr double kPivotTolerance = 1e-12;
inline constexpr double kPolicyMassFloor = 1e-9;
inline constexpr std::uint8_t kUnsolvedKind = 255;

struct ProfileTable {
    std::size_t profile_count{};
    //one entry per profile, indexed by profile id
    std::vector<std::int16_t>st{};
    std::vector<std::int16_t>ttd{};
    std::vector<Potential>potential{};
    std::vector<double>revival{};

    //Row-major by profile:
    //success_child[profile * kActions + action].
    std::vector<ChildId>success_child{};
    std::vector<ChildId>failure_child{};

    //TTD-major:
    //alive_id[ttd * kCapacity + st]. (300x301 matrix)
    std::vector<ChildId>alive_id{};

    //Potential buckets, 601 variable length lists
    std::array<std::vector<ProfileId>, kMaxProfilePotential + 1> buckets{};

};

struct TransitionValues{
    std::array<double, kActions> success{};
    double failed{};
};

struct Policy {
    std::array<double, kActions> mass{};
};

struct Certificate {
    double lower{};
    double upper{};
    double midpoint{};
    double gap{};
};

enum class SolverKind : std::uint8_t {
    Pure = 0,
    Support = 1,
    LinearProgram = 2,
};

enum class SolverRoute : std::uint8_t {
    Pure = 0,
    WarmSupport = 1,
    FullSupport = 2,
    LinearProgram = 3,
};

struct SolveResult{
    Certificate certificate{};
    Policy drop_policy{};
    Policy check_policy{};
    SolverRoute route{SolverRoute::Pure};
};

[[nodiscard]] constexpr SolverKind solver_kind_for(const SolverRoute route) noexcept
    {
        if (route == SolverRoute::Pure) return SolverKind::Pure;
        if (route == SolverRoute::LinearProgram) {
            return SolverKind::LinearProgram;
        }
        return SolverKind::Support;
    }

struct RouteCounters {
    std::uint64_t pure{};
    std::uint64_t warm_support{};
    std::uint64_t full_support{};
    std::uint64_t linear_program{};
};

//SECTION 2: SCALAR GAME RULES
[[nodiscard]] bool revival_eligibility(int st, int ttd);
[[nodiscard]] double revival_probability(int st, int ttd);

//SECTION 3: ENUMERATE FAILURE-FATAL QUOTIENT
ProfileTable begin_canonical_profile_table();
ProfileId quotient_profile_id(const ProfileTable& table, int st, int ttd);

//SECTION 4: PER-PROFILE TRANSITION TABLE
void finish_profile_table(ProfileTable& table);

//SECTION 5: DEFINE CLASS ENCODING AND BELLMAN ROLE SWAPS
ClassId encode_class(ProfileTable& table, ProfileId checker, ProfileId dropper);
std::pair<ProfileId, ProfileId> decode_class(ProfileTable& table, ClassId class_id);
ClassId swapped_child_class(ProfileTable& table, ProfileId dropper, ProfileId child_profile);
Potential class_potential(ProfileTable& table, ClassId class_id);

//SECTION 6: BUILD POTENTIAL BUCKETS AND PROVE DAG SCHEDULE
void build_buckets(ProfileTable& table);
void validate_profile_edges(ProfileTable& table);
int layer_size(ProfileTable& table, Potential potential);

//SECTTION 8: ASSEMBLE THE 61 CONTINUATION VALUES FOR A CLASS
TransitionValues assemble_transition_values(ProfileTable& table, const MappedArray<double>& values, ProfileId checker, ProfileId dropper);
double matrix_cell(TransitionValues& t, int drop, int check);
} //namespace dth
