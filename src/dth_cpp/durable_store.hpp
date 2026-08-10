#pragma once

#include "dth.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>

namespace dth {

inline constexpr std::int32_t kInitialCompletedPotential = 1'201;

// One open file whose bytes are directly exposed through virtual memory.
class MappedFile {
public:
    static MappedFile create(
        const std::filesystem::path& path,
        std::size_t byte_count);
    static MappedFile open_existing(
        const std::filesystem::path& path,
        std::size_t expected_byte_count);

    ~MappedFile();

    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;
    MappedFile(MappedFile&& other) noexcept;
    MappedFile& operator=(MappedFile&& other) noexcept;

    void* data() noexcept;
    const void* data() const noexcept;
    std::size_t byte_count() const noexcept;
    void flush();

private:
    MappedFile() = default;
    void close() noexcept;

#ifdef _WIN32
    void* file_handle_ = nullptr;
    void* mapping_handle_ = nullptr;
#else
    int file_descriptor_ = -1;
#endif

    void* data_ = nullptr;
    std::size_t byte_count_ = 0;
};

template <typename T>
class MappedArray {
public:
    static MappedArray create(
        const std::filesystem::path& path,
        std::size_t count,
        const T& initial_value);
    static MappedArray open_existing(
        const std::filesystem::path& path,
        std::size_t expected_count);

    MappedArray(const MappedArray&) = delete;
    MappedArray& operator=(const MappedArray&) = delete;
    MappedArray(MappedArray&&) noexcept = default;
    MappedArray& operator=(MappedArray&&) noexcept = default;

    T& operator[](std::size_t index);
    const T& operator[](std::size_t index) const;
    std::size_t size() const noexcept;
    void flush();

private:
    MappedArray(MappedFile file, std::size_t count) noexcept;
    static std::size_t checked_byte_count(std::size_t count);

    MappedFile file_;
    std::size_t count_ = 0;
};

struct CheckpointRecord {
    std::uint64_t profile_count{};
    ClassId class_count{};
    std::int32_t completed_potential{kInitialCompletedPotential};
    RouteCounters counters{};
};

struct DurableStores {
    MappedArray<double> values;
    MappedArray<std::uint8_t> solver_kind;
    CheckpointRecord checkpoint;
};

[[nodiscard]] DurableStores create_stores(
    const std::filesystem::path& output_dir,
    std::uint64_t profile_count,
    ClassId class_count);

void atomically_write_checkpoint(
    const std::filesystem::path& output_dir,
    const CheckpointRecord& record);

[[nodiscard]] DurableStores open_resume(
    const std::filesystem::path& output_dir,
    std::uint64_t expected_profile_count,
    ClassId expected_class_count);

} // namespace dth

#include "mapped_array.tpp"
