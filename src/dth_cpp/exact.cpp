#include "dth.hpp"

#include <stdexcept>
#include <cmath>
#include <ranges>

//helpers and shit
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
    std::size_t index_success_child(int id, int lag) {
        return static_cast<std::size_t>(id) * dth::kActions + static_cast<std::size_t>(lag-1);
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
    double chronic = pow(0.75, static_cast<double>(ttd) / 60.0);
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

    auto fill_profile = [&](const int ttd) {
        for (int st : std::views::iota(0, capacity)) {
            if (!revival_eligibility(st, ttd)) {
                continue;
                }
            table.alive_id[index_alive_id(st, ttd)] = static_cast<dth::ChildId>(next);
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
    return static_cast<dth::ProfileId>(id);
}

//SECTION 4
void dth::finish_profile_table(ProfileTable& table) {
    for (int id: std::views::iota(0, static_cast<int>(table.profile_count))) {
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
                    table.success_child[index_success_child(id, lag)] = -1;
                } else if (!failure_fatal) {
                    //SUCCESS CASE 2: still able to tank a failure
                    table.success_child[index_success_child(id, lag)] = static_cast<ChildId>(quotient_profile_id(table, new_st, ttd));
                } else {
                    //SUCCESS CASE 3: already failure_fatal so -1 -> -1; (new_st, -1) profile represented as 16,711 + new_st
                    table.success_child[index_success_child(id, lag)] = static_cast<ChildId>(dth::kDeadProfileBase + static_cast<size_t>(new_st));
                }
            }
            //1 failed check profile
            if (!failure_fatal){
                //REVIVAL CASE 1: if a failure isn't fatal and I survived the injection, then new child profile is (0, new_ttd)
                int new_ttd = ttd + st + static_cast<int>(dth::kPenalty);
                table.failure_child[id] = static_cast<ChildId>(quotient_profile_id(table, 0, new_ttd));
            } else {
                //REVIVAL CASE 2: if a failure is fatal then the child id is just sentinel
                table.failure_child[id] = -1;
            }          
    }
}

