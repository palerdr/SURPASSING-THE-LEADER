#include "dth.hpp"

#include <algorithm>
#include <cmath>
#include <ranges>
#include <stdexcept>

//helpers and whatnot
namespace {
    //private validation functions for st,ttd
    void validate_st(int st) {
        if (st < 0 || st >= 300) {
            throw std::out_of_range("st must be between 0 and 299");
        }
    }
    void validate_ttd(int ttd) {
        if (ttd < 0 || ttd > 300) {
            throw std::out_of_range("ttd must be between 0 and 300");
        }
    }
    void validate_profile(int st, int ttd) {
        validate_st(st);
        validate_ttd(ttd);
    }
    void validate_action(int action) {
        if (0 < action && action < 61) {
            return;
        }
        throw std::out_of_range("action must be between 0 and 60");
    }
    std::size_t index_alive_id(int st, int ttd) {
        validate_st(st);
        validate_ttd(ttd);
        return static_cast<std::size_t>(ttd) * dth::kCapacity + static_cast<std::size_t>(st);
    }
    std::size_t index_success_child(std::size_t id, int lag) {
        return id * dth::kActions + static_cast<std::size_t>(lag-1);
    }

}

//SECTION 2
bool dth::revival_eligibility(int st, int ttd) {
    validate_st(st);
    validate_ttd(ttd);
    bool eligible = (st <= 239 ) && (st + ttd <= 240);
    return eligible;
}

double dth::revival_probability(int st, int ttd) {
    validate_st(st);
    validate_ttd(ttd);
    if (!revival_eligibility(st, ttd)) {
        return 0.0;
    }
    double acute = 1.0 - (static_cast<double>(st) / 240.0);
    double chronic = std::pow(0.75, static_cast<double>(ttd) / 60.0);
    double p = 0.95 * acute * chronic;
    if (!std::isfinite(p) || p <= 0.0 || p >= 1.0) {
        throw std::invalid_argument("p must be a finite number between 0.0 and 1.0 (exclusive)");
    }
    return p;
}

int lag (int drop, int check) {
        validate_action(drop);
        validate_action(check);
        return check - drop + 1;
    }

//SECTION 3
dth::ProfileTable dth::begin_canonical_profile_table() {
    dth::ProfileTable table{};
    table.profile_count = dth::kCanonicalProfiles;

    table.st.resize(dth::kCanonicalProfiles);
    table.ttd.resize(dth::kCanonicalProfiles);
    table.potential.resize(dth::kCanonicalProfiles);
    table.revival.resize(dth::kCanonicalProfiles);
    table.failure_child.resize(dth::kCanonicalProfiles);
    table.success_child.resize(
        dth::kCanonicalProfiles * dth::kActions);

    table.alive_id.assign(
        (dth::kCapacity + 1) * dth::kCapacity,
        dth::ChildId{-1});

    constexpr int capacity = static_cast<int>(dth::kCapacity);
    constexpr int penalty = static_cast<int>(dth::kPenalty);

    std::size_t next = 0;

    auto fill_profile = [&table, &next, capacity](const int ttd) {
        for (int st : std::views::iota(0, capacity)) {
            if (!revival_eligibility(st, ttd)) {
                continue;
                }
            table.alive_id[index_alive_id(st, ttd)] = static_cast<ChildId>(next);
                table.st[next] = static_cast<std::int16_t>(st);
                table.ttd[next] = static_cast<std::int16_t>(ttd);
                ++next;
        }
    };

    fill_profile(0);
    for (int ttd : std::views::iota(penalty, capacity + 1)) {
        fill_profile(ttd);
    }

    if (next != dth::kAliveProfiles) {
        throw std::logic_error("canonical alive profile count mismatch");
    }
    //now add the dead profiles
    for (int st : std::views::iota(0, capacity)) {
        std::size_t id = dth::kDeadProfileBase + static_cast<std::size_t>(st);
        table.st[id] = static_cast<std::int16_t>(st);
        table.ttd[id] = -1;
        ++next;
    }
    if (next != dth::kCanonicalProfiles){
        throw std::logic_error("canonical total profile count mismatch");
    }
    return table;
}

dth::ProfileId dth::quotient_profile_id(const ProfileTable& table, int st, int ttd) {
    validate_profile(st, ttd);
    if (!revival_eligibility(st, ttd)) {
        return static_cast<dth::ProfileId>(dth::kDeadProfileBase + static_cast<std::size_t>(st));
    }
    dth::ChildId id = table.alive_id[index_alive_id(st, ttd)];
    if (id == -1) {
        throw std::logic_error("eligible profile has off-domain TTD 1..59");
    }
    if (id <  0) {
        throw std::logic_error("id cannot be negative");
    }
    return static_cast<dth::ProfileId>(id);
}

//SECTION 4
void dth::finish_profile_table(ProfileTable& table) {
    for (std::size_t id: std::views::iota(std::size_t{0}, table.profile_count)) {
            int st = table.st[id];
            int ttd = table.ttd[id];
            bool failure_fatal = (ttd < 0);

            if (!failure_fatal) {
                table.potential[id] = static_cast<dth::Potential>(st + ttd);
                table.revival[id] = revival_probability(st, ttd);
            } else {
                table.potential[id] = static_cast<dth::Potential>(st + 301);
                table.revival[id] = 0.0;
            }
            //60 successful check profiles
            for (int lag: std::views::iota(1, 61)) {
                int new_st = st + lag;
                if (new_st >= 300) {
                    //SUCCESS CASE 1: absolute death child
                    table.success_child[index_success_child(id, lag)] = ChildId{-1};
                } else if (!failure_fatal) {
                    //SUCCESS CASE 2: still able to tank a failure
                    const ProfileId profile = quotient_profile_id(table, new_st, ttd);
                    const ChildId child = static_cast<ChildId>(profile);
                    table.success_child[index_success_child(id, lag)] = child;
                } else {
                    //SUCCESS CASE 3: already failure_fatal so -1 -> -1; (new_st, -1) profile represented as 16,711 + new_st
                    const ChildId child = static_cast<ChildId>(dth::kDeadProfileBase + static_cast<std::size_t>(new_st));
                    table.success_child[index_success_child(id, lag)] = child;
                }
            }
            //1 failed check profile
            if (!failure_fatal){
                //REVIVAL CASE 1: if a failure isn't fatal and I survived the injection, then new child profile is (0, new_ttd)
                int new_ttd = ttd + st + static_cast<int>(dth::kPenalty);
                table.failure_child[id] = static_cast<ChildId>(quotient_profile_id(table, 0, new_ttd));
            } else {
                //REVIVAL CASE 2: if a failure is fatal then the child id is just sentinel
                table.failure_child[id] = ChildId{-1};
            }          
    }
}

//SECTION 5
dth::ClassId dth::encode_class(ProfileTable& table, ProfileId checker, ProfileId dropper) {
    //17,011 x 17,011 matrix of profile indexed by classId
    if (checker >= table.profile_count) {
        throw std::out_of_range("checker ID must be within profile range");
    }
    if (dropper >= table.profile_count) {
        throw std::out_of_range("dropper ID must be within profile range");
    }
    return ClassId{checker} * static_cast<ClassId>(table.profile_count) + ClassId{dropper};
}

std::pair<dth::ProfileId, dth::ProfileId> dth::decode_class(dth::ProfileTable &table, dth::ClassId class_id) {
    if (class_id >= kCanonicalClasses) {
        throw std::out_of_range("class_id must be within the cross product of profile_counts");
    }
    ProfileId checker = static_cast<ProfileId>(class_id / table.profile_count);
    ProfileId dropper = static_cast<ProfileId>(class_id % table.profile_count);
    return std::make_pair(checker, dropper);
}

dth::ClassId dth::swapped_child_class(ProfileTable& table, ProfileId dropper, ProfileId child_profile) {
    //current checker moves -> child_profile post half round, must swap roles for bellman recursion
    return encode_class(table, dropper, child_profile);
}

dth::Potential dth::class_potential(ProfileTable& table, ClassId class_id) {
    auto [checker, dropper] = decode_class(table, class_id);
    return table.potential[static_cast<std::size_t>(checker)] + table.potential[static_cast<std::size_t>(dropper)];
}

//SECTION 6
void dth::build_buckets(ProfileTable& table) {
    for (std::size_t profile : std::views::iota(std::size_t{0}, table.profile_count)) {
        table.buckets[table.potential[profile]].push_back(profile);
    }
}
void dth::validate_profile_edges(ProfileTable& table) {
    int live_success{0};
    int live_failure{0};
    for (std::size_t profile: std::views::iota(std::size_t{0}, table.profile_count)) {
        const Potential parent_phi = table.potential[profile];
        const std::size_t row_begin = profile * kActions;
        const std::size_t row_end = row_begin + kActions;

        for (std::size_t i : std::views::iota(row_begin, row_end)) {
            const ChildId success_child = table.success_child[i];
            //validate child
            if (success_child >= 0) {
                if (table.potential[success_child] <= parent_phi) {
                    throw std::logic_error("successful child potential must be strictly monotonically increasing");
                }
                ++live_success;
            }
        }
        const ChildId fail_child = table.failure_child[profile];
        if (fail_child >= 0) {
            if (table.potential[fail_child] <= parent_phi) {
                throw std::logic_error("failed child potentital must be strictly monotonically increasing");
            }
            ++live_failure;
        }
    }
    if (live_success != 1'018'830) {
        throw std::logic_error("live successes differ from the canonical number");
    }
    if (live_failure != 16'711) {
        throw std::logic_error("live failures differ from the canonical number");
    }
}

int dth::layer_size(ProfileTable& table, Potential potential) {
    int total{0};
    const size_t first = std::size_t{std::max(0, int{potential} - 600)};
    const size_t last = std::size_t{std::min(600, int{potential})};
    for (std::size_t a: std::views::iota(first, last + 1)) {
        total += table.buckets[a].size() * table.buckets[potential - a].size();
    }
    return total;
}

//SECTION 8
