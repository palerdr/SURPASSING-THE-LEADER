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
}

bool dth::revival_eligibility(int st, int ttd) {
    validate_st(st);
    validate_ttd(ttd);
    bool eligible = (st <= 239 and st + ttd <= 240);
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
    int ttd = 0;
    for (int st : std::views::iota(0, capacity)) {
        if (revival_eligibility(st, ttd)) {
            table.alive_id[index_alive_id(st, ttd)] =
                static_cast<dth::ChildId>(next);
            table.st[next] = static_cast<std::int16_t>(st);
            table.ttd[next] = static_cast<std::int16_t>(ttd);
            ++next;
        }
    }
    for (int ttd : std::views::iota(penalty, capacity + 1)) {
        for (int st : std::views::iota(0, capacity)) {
        if (revival_eligibility(st, ttd)) {
            table.alive_id[index_alive_id(st, ttd)] = static_cast<dth::ChildId>(next);
            table.st[next] = static_cast<std::int16_t>(st);
            table.ttd[next] = static_cast<std::int16_t>(ttd);
            ++next;
            }
        }
    }
    if (next != dth::kAliveProfiles) {
        throw std::logic_error("canonical alive profile count mismatch");
    }
    for (int st : std::views::iota(0, capacity)) {
    std::size_t id = dth::kDeadProfileBase + static_cast<std::size_t>(st);
    table.st[id] = static_cast<std::int16_t>(st);
    table.ttd[id] = -1;
    }
    if (next != dth::kCanonicalProfiles){
        throw std::logic_error("canonical total profile count mismatch");
    }
    return table;
}

dth::ProfileId dth::quotient_profile_id(const& ProfileTable table, int st, int ttd) {
    validate_st(st);
    validate_ttd(ttd);
    if (!revival_eligibility(st, ttd)) {
        return static_cast<dth::ProfileId>(dth::kDeadProfileBase + static_cast<std::size_t>(st));
    }

}
