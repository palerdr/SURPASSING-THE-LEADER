#include "sweep.hpp"
#include <algorithm>
#include <charconv>
#include <csignal>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

namespace {
volatile std::sig_atomic_t stop_requested = 0;
void request_stop(int) {
    stop_requested = 1;
}
std::size_t count(const std::string& text, bool allow_zero = false) {
    std::size_t value{};
    auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size() ||
        (!allow_zero && value == 0))
        throw std::invalid_argument("invalid count: " + text);
    return value;
}
} // namespace
int main(int argc, char** argv) {
    bool artifact_operation = false;
    try {
        std::filesystem::path output;
        dth::SweepOptions options;
        options.threads = std::max(1u, std::thread::hardware_concurrency());
        options.progress_every = 50;
        options.stop_requested = &stop_requested;
        std::string mode;
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            auto value = [&] {
                if (++i >= argc)
                    throw std::invalid_argument("missing value for " + arg);
                return std::string(argv[i]);
            };
            if (arg == "--output")
                output = value();
            else if (arg == "--threads")
                options.threads = count(value());
            else if (arg == "--stop-after-layers")
                options.stop_after_layers = count(value());
            else if (arg == "--checkpoint-every")
                options.checkpoint_every = count(value());
            else if (arg == "--progress-every")
                options.progress_every = count(value(), true);
            else if (arg == "--fresh" || arg == "--resume" || arg == "--verify-only") {
                if (!mode.empty())
                    throw std::invalid_argument("choose one mode");
                mode = arg;
            } else
                throw std::invalid_argument("unknown option: " + arg);
        }
        if (output.empty() || mode.empty())
            throw std::invalid_argument(
                "require --output PATH and one of --fresh, --resume, --verify-only");
        if (mode == "--verify-only" && options.stop_after_layers)
            throw std::invalid_argument("verification has no layer limit");
        if (mode == "--fresh" && std::filesystem::exists(output) &&
            !std::filesystem::is_empty(output))
            throw std::invalid_argument("fresh output must be absent or empty");
        auto table = dth::begin_canonical_profile_table();
        dth::finish_profile_table(table);
        dth::build_buckets(table);
        dth::validate_profile_edges(table);
        artifact_operation = mode != "--fresh";
        if (mode == "--fresh") {
            std::filesystem::create_directories(output);
            dth::atomic_write_text(output, "build-config.txt", dth::build_config(table));
        } else
            dth::validate_build_config(table, output);
        auto stores = mode == "--fresh"
                          ? dth::create_stores(output, table.profile_count, dth::kCanonicalClasses)
                          : dth::open_resume(output, table.profile_count, dth::kCanonicalClasses,
                                             mode == "--verify-only");
        if (mode == "--verify-only" || std::filesystem::exists(output / "tablebase.json")) {
            dth::verify(table, stores, output);
            std::cout << "Verified root=" << stores.values[0] << '\n';
            return 0;
        }
        artifact_operation = false;
        std::signal(SIGINT, request_stop);
        std::signal(SIGTERM, request_stop);
        std::cout << "threads=" << options.threads << std::endl;
        if (!dth::sweep(table, stores, output, options)) {
            std::cout << "Checkpoint potential=" << stores.checkpoint.completed_potential << '\n';
            return 5;
        }
        dth::finalize(table, stores, output);
        std::cout << "Completed root=" << stores.values[0] << '\n';
        return 0;
    } catch (const std::invalid_argument& e) {
        std::cerr << e.what() << '\n';
        return 2;
    } catch (const std::filesystem::filesystem_error& e) {
        std::cerr << e.what() << '\n';
        return 4;
    } catch (const std::system_error& e) {
        std::cerr << e.what() << '\n';
        return 4;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return artifact_operation ? 6 : 3;
    }
}
