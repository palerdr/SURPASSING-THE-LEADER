#include "sweep.hpp"
#include "highs_backend.hpp"
#include "source_digest.hpp"
#include "storage/digest.hpp"

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <thread>

namespace dth {
namespace {
std::uint64_t total(const RouteCounters& c) {
    return c.pure + c.warm_support + c.full_support + c.linear_program;
}
void add(RouteCounters& a, const RouteCounters& b) {
    a.pure += b.pure;
    a.warm_support += b.warm_support;
    a.full_support += b.full_support;
    a.linear_program += b.linear_program;
}
std::string read_text(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in)
        throw std::runtime_error("missing artifact file: " + path.string());
    std::ostringstream out;
    out << in.rdbuf();
    if (in.bad())
        throw std::runtime_error("artifact read failed");
    return out.str();
}
struct Pair {
    ProfileId checker, dropper;
};
std::vector<Pair> layer_pairs(const ProfileTable& table, int potential) {
    std::vector<Pair> pairs;
    pairs.reserve(static_cast<std::size_t>(layer_size(table, static_cast<Potential>(potential))));
    for (int a = std::max(0, potential - 600); a <= std::min(600, potential); ++a)
        for (auto d : table.buckets[static_cast<std::size_t>(potential - a)])
            for (auto c : table.buckets[static_cast<std::size_t>(a)])
                pairs.push_back({c, d});
    return pairs;
}
struct Worker {
    HighsBackend backend;
    MatrixScratch matrix{};
    RouteCounters counters{};
    std::array<ProfileId, 16> checkers{}, droppers{};
};
class Pool {
  public:
    Pool(const ProfileTable& table, DurableStores& stores, std::size_t count)
        : table_(table), stores_(stores) {
        if (!count)
            throw std::invalid_argument("threads must be positive");
        workers_.reserve(count);
        threads_.reserve(count);
        for (std::size_t i = 0; i < count; ++i)
            workers_.push_back(std::make_unique<Worker>());
        try {
            for (std::size_t i = 0; i < count; ++i)
                threads_.emplace_back([this, i] { run(*workers_[i]); });
        } catch (...) {
            shutdown();
            throw;
        }
    }
    ~Pool() {
        shutdown();
    }
    RouteCounters dispatch(const std::vector<Pair>& pairs) {
        std::unique_lock lock(mutex_);
        pairs_ = &pairs;
        next_ = 0;
        chunk_ = pairs.size() < 2048 ? 16 : 2048;
        error_ = nullptr;
        for (auto& w : workers_)
            w->counters = {};
        remaining_ = threads_.size();
        ++generation_;
        start_.notify_all();
        done_.wait(lock, [this] { return remaining_ == 0; });
        if (error_)
            std::rethrow_exception(error_);
        RouteCounters result{};
        for (auto& w : workers_)
            add(result, w->counters);
        if (total(result) != pairs.size())
            throw std::runtime_error("layer route count mismatch");
        return result;
    }

  private:
    void shutdown() {
        {
            std::lock_guard lock(mutex_);
            stop_ = true;
            start_.notify_all();
        }
        for (auto& t : threads_)
            if (t.joinable())
                t.join();
    }
    void process(Worker& worker, std::size_t first, std::size_t last) {
        for (std::size_t base = first; base < last; base += 16) {
            const auto width = std::min(std::size_t{16}, last - base);
            for (std::size_t l = 0; l < width; ++l) {
                auto pair = (*pairs_)[base + l];
                worker.checkers[l] = pair.checker;
                worker.droppers[l] = pair.dropper;
                stores_.solver_kind[encode_class(table_, pair.checker, pair.dropper)] =
                    kUnsolvedKind;
            }
            auto unresolved = solve_recurrence_chunk(
                static_cast<std::int64_t>(width), worker.checkers.data(), worker.droppers.data(),
                stores_.values.data(), stores_.solver_kind.data(),
                static_cast<std::int64_t>(table_.profile_count), table_.success_child.data(),
                table_.failure_child.data(), table_.revival.data());
            if (unresolved < 0)
                throw std::runtime_error("invalid child or floating-point environment");
            for (std::size_t l = 0; l < width; ++l) {
                const auto c = worker.checkers[l], d = worker.droppers[l];
                const auto id = encode_class(table_, c, d);
                if (stores_.solver_kind[id] == kUnsolvedKind) {
                    try {
                        auto t = assemble_transition_values(table_, stores_.values, c, d);
                        auto result = solve_stage(t, worker.backend, worker.matrix);
                        stores_.values[id] = result.certificate.midpoint;
                        stores_.solver_kind[id] =
                            static_cast<std::uint8_t>(solver_kind_for(result.route));
                    } catch (const std::exception& e) {
                        throw std::runtime_error("class " + std::to_string(id) + ": " + e.what());
                    }
                }
                switch (stores_.solver_kind[id]) {
                case 0:
                    ++worker.counters.pure;
                    break;
                case 1:
                    ++worker.counters.full_support;
                    break;
                case 2:
                    ++worker.counters.linear_program;
                    break;
                default:
                    throw std::runtime_error("unresolved class");
                }
            }
        }
    }
    void run(Worker& worker) {
        std::size_t seen = 0;
        for (;;) {
            {
                std::unique_lock lock(mutex_);
                start_.wait(lock, [&] { return stop_ || generation_ != seen; });
                if (stop_)
                    return;
                seen = generation_;
            }
            try {
                for (;;) {
                    auto first = next_.fetch_add(chunk_);
                    if (first >= pairs_->size())
                        break;
                    process(worker, first, std::min(first + chunk_, pairs_->size()));
                }
            } catch (...) {
                std::lock_guard lock(mutex_);
                if (!error_)
                    error_ = std::current_exception();
            }
            {
                std::lock_guard lock(mutex_);
                if (--remaining_ == 0)
                    done_.notify_one();
            }
        }
    }
    const ProfileTable& table_;
    DurableStores& stores_;
    std::vector<std::unique_ptr<Worker>> workers_;
    std::vector<std::thread> threads_;
    std::mutex mutex_;
    std::condition_variable start_, done_;
    std::size_t generation_{}, remaining_{}, chunk_{2048};
    bool stop_{};
    const std::vector<Pair>* pairs_{};
    std::atomic<std::size_t> next_{};
    std::exception_ptr error_{};
};
} // namespace
std::string build_config(const ProfileTable& table) {
    if (std::endian::native != std::endian::little)
        throw std::runtime_error("require little-endian storage");
    Sha256 digest;
    auto feed = [&]<typename T>(const std::vector<T>& data) {
        digest.update(std::as_bytes(std::span(data)));
    };
    feed(table.st);
    feed(table.ttd);
    feed(table.potential);
    feed(table.revival);
    feed(table.success_child);
    feed(table.failure_child);
    std::ostringstream out;
    out << "dth-cpp-recurrence-v1\n"
        << DTH_SOURCE_DIGEST << '\n'
        << table.profile_count << '\n'
        << digest.finish() << '\n';
    return out.str();
}
void validate_build_config(const ProfileTable& table, const std::filesystem::path& directory) {
    if (read_text(directory / "build-config.txt") != build_config(table))
        throw std::runtime_error("artifact source or profile configuration is stale");
}
bool sweep(const ProfileTable& table, DurableStores& stores, const std::filesystem::path& directory,
           const SweepOptions& options) {
    if (!options.checkpoint_every)
        throw std::invalid_argument("checkpoint interval must be positive");
    validate_profile_edges(table);
    std::vector<bool> seen(table.profile_count, false);
    for (std::size_t p = 0; p < table.buckets.size(); ++p)
        for (auto id : table.buckets[p]) {
            if (id >= table.profile_count || seen[id] || table.potential[id] != p)
                throw std::logic_error("profile bucket partition mismatch");
            seen[id] = true;
        }
    if (std::find(seen.begin(), seen.end(), false) != seen.end())
        throw std::logic_error("profile missing from bucket partition");

    validate_build_config(table, directory);
    if (options.stop_after_layers && *options.stop_after_layers == 0)
        throw std::invalid_argument("layer limit must be positive");
    std::uint64_t expected = 0;
    for (int p = stores.checkpoint.completed_potential; p <= 1200; ++p)
        expected += static_cast<std::uint64_t>(layer_size(table, static_cast<Potential>(p)));
    if (expected != total(stores.checkpoint.counters))
        throw std::runtime_error("checkpoint counters disagree with completed layers");
    Pool pool(table, stores, options.threads);
    std::size_t layers = 0;
    auto started = std::chrono::steady_clock::now();
    for (int p = stores.checkpoint.completed_potential - 1; p >= 0; --p) {
        auto pairs = layer_pairs(table, p);
        auto delta = pool.dispatch(pairs);
        auto next = stores.checkpoint;
        next.completed_potential = p;
        add(next.counters, delta);
        ++layers;
        bool stop = (options.stop_after_layers && layers >= *options.stop_after_layers) ||
                    (options.stop_requested && *options.stop_requested);
        if (p == 0 || stop || layers % options.checkpoint_every == 0) {
            stores.values.flush();
            stores.solver_kind.flush();
            atomically_write_checkpoint(directory, next);
        }
        stores.checkpoint = next;
        if (options.progress_every && layers % options.progress_every == 0) {
            auto elapsed =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
            std::cout << "potential=" << p << " solved=" << total(next.counters)
                      << " seconds=" << elapsed << std::endl;
        }
        if (p > 0 && stop)
            return false;
    }
    return true;
}
std::string completion_manifest(const ProfileTable& table, const DurableStores& stores,
                                const std::filesystem::path& directory) {
    validate_build_config(table, directory);
    const auto count = static_cast<ClassId>(table.profile_count) * table.profile_count;
    if (stores.checkpoint.completed_potential != 0 || total(stores.checkpoint.counters) != count ||
        stores.checkpoint.class_count != count)
        throw std::runtime_error("artifact is incomplete");
    RouteCounters counted{};
    for (ClassId id = 0; id < count; ++id) {
        if (!std::isfinite(stores.values[id]) || std::abs(stores.values[id]) > 1.0 + 1e-9)
            throw std::runtime_error("invalid stored value");
        switch (stores.solver_kind[id]) {
        case 0:
            ++counted.pure;
            break;
        case 1:
            ++counted.full_support;
            break;
        case 2:
            ++counted.linear_program;
            break;
        default:
            throw std::runtime_error("invalid stored route");
        }
    }
    const auto& counters = stores.checkpoint.counters;
    if (counted.pure != counters.pure || counted.full_support != counters.full_support ||
        counted.linear_program != counters.linear_program || counters.warm_support)
        throw std::runtime_error("route counts disagree with checkpoint");
    HighsBackend backend;
    MatrixScratch scratch{};
    std::size_t samples = 0;
    double worst = 0;
    for (int p = 1200; p >= 0; --p) {
        auto pairs = layer_pairs(table, p);
        for (std::size_t i = 0; i < std::min(std::size_t{4}, pairs.size()); ++i) {
            auto [c, d] = pairs[i];
            auto t = assemble_transition_values(table, stores.values, c, d);
            auto fresh = solve_stage(t, backend, scratch);
            auto difference =
                std::abs(fresh.certificate.midpoint - stores.values[encode_class(table, c, d)]);
            if (difference > kSaddleTolerance)
                throw std::runtime_error("stored value failed recertification");
            worst = std::max(worst, difference);
            ++samples;
        }
    }
    if (table.profile_count == kCanonicalProfiles) {
        if (std::abs(stores.values[0] - 0.08985) > 0.00061)
            throw std::runtime_error("root anchor mismatch");
        auto dead = quotient_profile_id(table, 240, 0);
        if (std::abs(stores.values[encode_class(table, dead, dead)] - 0.3372132166291093) > 1e-9)
            throw std::runtime_error("dead-band anchor mismatch");
    }
    std::ostringstream out;
    out << std::setprecision(17);
    out << "{\n  \"schema\": \"dth.cpp-complete-tablebase.v1\",\n  \"source_digest\": \""
        << DTH_SOURCE_DIGEST << "\",\n"
        << "  \"config_id\": \"dth-cpp-recurrence-v1\",\n  \"profiles\": " << table.profile_count
        << ",\n  \"classes\": " << count << ",\n"
        << "  \"maximum_potential\": 1200,\n  \"saddle_tolerance\": 1e-6,\n  \"ladder\": "
           "\"pure/recurrence/full-support/highs-covering-v1\",\n"
        << "  \"highs_version\": \"" << backend.version()
        << "\",\n  \"highs_commit\": \"04024d701f79feb8e2f18bc3df0dffc04ef05088\",\n"
        << "  \"highs_options\": {\"solver\": \"simplex\", \"simplex_strategy\": 1, \"parallel\": "
           "\"off\", \"threads\": 1, \"random_seed\": 0, \"presolve\": \"off\", "
           "\"primal_feasibility_tolerance\": 1e-10, \"dual_feasibility_tolerance\": 1e-10, "
           "\"small_matrix_value\": 1e-12, \"simplex_iteration_limit\": 10000},\n"
        << "  \"pure\": " << counted.pure << ",\n  \"support\": " << counted.full_support
        << ",\n  \"lp\": " << counted.linear_program << ",\n"
        << "  \"recertified_samples\": " << samples << ",\n  \"worst_difference\": " << worst
        << ",\n  \"root_class\": 0,\n  \"root_value\": " << stores.values[0] << ",\n"
        << "  \"values\": {\"file\": \"values.bin\", \"type\": \"<f8\", \"bytes\": " << count * 8
        << ", \"sha256\": \"" << sha256_file(directory / "values.bin") << "\"},\n"
        << "  \"routes\": {\"file\": \"solver_kind.bin\", \"type\": \"u1\", \"bytes\": " << count
        << ", \"sha256\": \"" << sha256_file(directory / "solver_kind.bin") << "\"}\n}\n";
    return out.str();
}
void finalize(const ProfileTable& table, const DurableStores& stores,
              const std::filesystem::path& directory) {
    atomic_write_text(directory, "tablebase.json", completion_manifest(table, stores, directory));
}
void verify(const ProfileTable& table, const DurableStores& stores,
            const std::filesystem::path& directory) {
    auto actual = read_text(directory / "tablebase.json");
    if (actual != completion_manifest(table, stores, directory))
        throw std::runtime_error("manifest or data digest mismatch");
}
} // namespace dth
