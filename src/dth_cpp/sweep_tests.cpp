#include "highs_backend.hpp"
#include "storage/digest.hpp"
#include "sweep.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>

namespace {
void require(bool ok, const char* message) {
    if (!ok)
        throw std::runtime_error(message);
}
template <class F> void rejects(F f) {
    bool rejected = false;
    try {
        f();
    } catch (const std::exception&) {
        rejected = true;
    }
    require(rejected, "expected rejection");
}
dth::ProfileTable synthetic(std::size_t count) {
    dth::ProfileTable t;
    t.profile_count = count;
    t.st.resize(count);
    t.ttd.resize(count);
    t.potential.resize(count);
    t.revival.resize(count);
    t.success_child.resize(count * 60, -1);
    t.failure_child.resize(count, -1);
    for (std::size_t i = 0; i < count; ++i) {
        t.st[i] = static_cast<std::int16_t>(i);
        t.potential[i] = static_cast<dth::Potential>(i);
        if (i + 7 < count) {
            t.revival[i] = 0.85 - 0.6 * static_cast<double>(i) / static_cast<double>(count);
            t.failure_child[i] = static_cast<dth::ChildId>(i + 7);
        }
        for (std::size_t lag = 1; lag <= 60; ++lag)
            if (i + lag < count)
                t.success_child[i * 60 + lag - 1] = static_cast<dth::ChildId>(i + lag);
    }
    dth::build_buckets(t);
    dth::validate_profile_edges(t);
    return t;
}
dth::DurableStores create(const dth::ProfileTable& table, const std::filesystem::path& directory) {
    auto stores = dth::create_stores(directory, table.profile_count,
                                     table.profile_count * table.profile_count);
    dth::atomic_write_text(directory, "build-config.txt", dth::build_config(table));
    return stores;
}
} // namespace
int main() {
    auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    auto temporary =
        std::filesystem::temp_directory_path() / ("dth-recurrence-test-" + std::to_string(stamp));
    try {
        std::filesystem::create_directory(temporary);
        {
            std::ofstream out(temporary / "abc");
            out << "abc";
        }
        require(dth::sha256_file(temporary / "abc") ==
                    "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
                "SHA256 abc mismatch");
        {
            std::ofstream out(temporary / "million");
            for (int i = 0; i < 1000000; ++i)
                out << 'a';
        }
        require(dth::sha256_file(temporary / "million") ==
                    "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0",
                "SHA256 blocks mismatch");
        auto table = synthetic(40);
        auto one = create(table, temporary / "one");
        dth::SweepOptions options;
        options.threads = 1;
        options.checkpoint_every = 7;
        require(dth::sweep(table, one, temporary / "one", options), "sequential incomplete");
        dth::finalize(table, one, temporary / "one");
        for (std::size_t threads : {2u, 12u}) {
            auto path = temporary / std::to_string(threads);
            {
                auto partial = create(table, path);
                options.threads = threads;
                options.stop_after_layers = 1160;
                require(!dth::sweep(table, partial, path, options), "stop did not checkpoint");
            }
            auto resumed = dth::open_resume(path, 40, 1600);
            options.stop_after_layers.reset();
            require(dth::sweep(table, resumed, path, options), "resume incomplete");
            require(dth::sha256_file(path / "values.bin") ==
                        dth::sha256_file(temporary / "one/values.bin"),
                    "worker/resume values differ");
            require(dth::sha256_file(path / "solver_kind.bin") ==
                        dth::sha256_file(temporary / "one/solver_kind.bin"),
                    "worker/resume routes differ");
        }
        {
            std::mt19937 engine(191);
            std::uniform_real_distribution<double> distribution(-1.0, 1.0);
            dth::HighsBackend backend;
            dth::MatrixScratch scratch{};
            bool reached_lp = false;
            for (int i = 0; i < 100; ++i) {
                dth::TransitionValues t{};
                for (auto& value : t.success)
                    value = distribution(engine);
                t.failed = distribution(engine);
                auto candidate = dth::solve_stage(t, backend, scratch);
                auto oracle = dth::try_linear_program(t, backend, scratch);
                require(oracle.has_value(), "random LP rejected");
                require(std::abs(candidate.certificate.midpoint - oracle->certificate.midpoint) <=
                            1e-6,
                        "ladder disagrees with LP");
                reached_lp |= candidate.route == dth::SolverRoute::LinearProgram;
            }
            require(reached_lp, "corpus did not exercise LP fallback");
        }
        // Resolve the complete small game through LP in an independent order.
        auto oracle = create(table, temporary / "oracle");
        std::vector<dth::ClassId> order(1600);
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](auto a, auto b) {
            return dth::class_potential(table, a) > dth::class_potential(table, b);
        });
        dth::HighsBackend backend;
        dth::MatrixScratch scratch{};
        for (auto id : order) {
            auto [c, d] = dth::decode_class(table, id);
            auto t = dth::assemble_transition_values(table, oracle.values, c, d);
            auto solved = dth::try_linear_program(t, backend, scratch);
            require(solved.has_value(), "LP oracle rejected");
            oracle.values[id] = solved->certificate.midpoint;
            require(std::abs(oracle.values[id] - one.values[id]) <= 1e-6,
                    "recurrence differs from independent LP sweep");
        }
        {
            const auto read_only = dth::open_resume(temporary / "one", 40, 1600, true);
            dth::verify(table, read_only, temporary / "one");
        }
        auto original = one.values[123];
        one.values[123] = original + 0.01;
        one.values.flush();
        rejects([&] { dth::verify(table, one, temporary / "one"); });
        one.values[123] = original;
        one.values.flush();
        one.solver_kind[123] = 255;
        rejects([&] { dth::verify(table, one, temporary / "one"); });
        auto broken = table;
        broken.success_child.back() = 0;
        auto bad = create(broken, temporary / "bad");
        rejects([&] { dth::sweep(broken, bad, temporary / "bad", options); });
        std::filesystem::remove_all(temporary);
        std::cout << "Recurrence, LP parity, resume, threads, read-only verification, and "
                     "corruption gates passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        std::filesystem::remove_all(temporary);
        return 1;
    }
}
