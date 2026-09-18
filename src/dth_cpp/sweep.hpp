#pragma once
#include "dth.hpp"
#include "storage/durable_store.hpp"
#include <csignal>
#include <filesystem>
#include <optional>
#include <string>

namespace dth {
struct SweepOptions {
    std::size_t threads{1};
    std::optional<std::size_t> stop_after_layers{};
    std::size_t progress_every{0};
    std::size_t checkpoint_every{50};
    const volatile std::sig_atomic_t* stop_requested{};
};
std::string build_config(const ProfileTable& table);
void validate_build_config(const ProfileTable& table, const std::filesystem::path& directory);
bool sweep(const ProfileTable& table, DurableStores& stores, const std::filesystem::path& directory,
           const SweepOptions& options);
std::string completion_manifest(const ProfileTable& table, const DurableStores& stores,
                                const std::filesystem::path& directory);
void finalize(const ProfileTable& table, const DurableStores& stores,
              const std::filesystem::path& directory);
void verify(const ProfileTable& table, const DurableStores& stores,
            const std::filesystem::path& directory);
} // namespace dth
