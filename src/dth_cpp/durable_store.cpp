#include "durable_store.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cerrno>
#include <cstdio>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
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
#include <fcntl.h>
#include <unistd.h>
#endif

// SECTION 7: CREATE DURABLE DENSE VALUE AND ROUTING STORES
namespace {

constexpr std::array<std::uint8_t, 8> kCheckpointMagic{
    'D', 'T', 'H', 'C', 'P', 'V', '1', 0,
};
constexpr std::uint32_t kCheckpointSchemaVersion = 1;
constexpr std::string_view kCheckpointConfigId = "dth-cpp-complete-v1";
constexpr std::size_t kMaximumCheckpointBytes = 4'096;

std::filesystem::path checkpoint_path(
    const std::filesystem::path& output_dir)
{
    return output_dir / "checkpoint.bin";
}

std::filesystem::path temporary_checkpoint_path(
    const std::filesystem::path& output_dir)
{
    return output_dir / "checkpoint.tmp";
}

void validate_checkpoint_record(const dth::CheckpointRecord& record)
{
    if (record.profile_count == 0) {
        throw std::invalid_argument("checkpoint profile count cannot be zero");
    }
    if (record.class_count == 0) {
        throw std::invalid_argument("checkpoint class count cannot be zero");
    }
    if (record.completed_potential < 0
        || record.completed_potential > dth::kInitialCompletedPotential) {
        throw std::invalid_argument(
            "checkpoint completed potential is outside 0..1201");
    }
}

std::size_t checked_class_count(const dth::ClassId class_count)
{
    if (class_count == 0) {
        throw std::invalid_argument("class count cannot be zero");
    }
    if (class_count
        > static_cast<dth::ClassId>(
            std::numeric_limits<std::size_t>::max())) {
        throw std::length_error("class count does not fit in size_t");
    }
    return static_cast<std::size_t>(class_count);
}

template <typename Unsigned>
void append_little_endian(
    std::vector<std::uint8_t>& bytes,
    Unsigned value)
{
    static_assert(std::is_unsigned_v<Unsigned>);
    for (std::size_t index = 0; index < sizeof(Unsigned); ++index) {
        bytes.push_back(static_cast<std::uint8_t>(value & Unsigned{0xff}));
        value >>= 8;
    }
}

template <typename Unsigned>
Unsigned read_little_endian(
    const std::vector<std::uint8_t>& bytes,
    std::size_t& offset)
{
    static_assert(std::is_unsigned_v<Unsigned>);
    if (bytes.size() - std::min(bytes.size(), offset) < sizeof(Unsigned)) {
        throw std::runtime_error("checkpoint is truncated");
    }

    Unsigned value = 0;
    for (std::size_t index = 0; index < sizeof(Unsigned); ++index) {
        value |= static_cast<Unsigned>(bytes[offset++]) << (index * 8);
    }
    return value;
}

std::vector<std::uint8_t> serialize_checkpoint(
    const dth::CheckpointRecord& record)
{
    validate_checkpoint_record(record);

    std::vector<std::uint8_t> bytes;
    bytes.reserve(128);
    bytes.insert(bytes.end(), kCheckpointMagic.begin(), kCheckpointMagic.end());
    append_little_endian(bytes, kCheckpointSchemaVersion);
    append_little_endian(
        bytes,
        static_cast<std::uint32_t>(kCheckpointConfigId.size()));
    bytes.insert(
        bytes.end(),
        kCheckpointConfigId.begin(),
        kCheckpointConfigId.end());
    append_little_endian(bytes, record.profile_count);
    append_little_endian(bytes, static_cast<std::uint64_t>(record.class_count));
    append_little_endian(
        bytes,
        std::bit_cast<std::uint32_t>(record.completed_potential));
    append_little_endian(bytes, record.counters.pure);
    append_little_endian(bytes, record.counters.warm_support);
    append_little_endian(bytes, record.counters.full_support);
    append_little_endian(bytes, record.counters.linear_program);
    return bytes;
}

std::vector<std::uint8_t> read_checkpoint_bytes(
    const std::filesystem::path& path)
{
    const std::uintmax_t file_size = std::filesystem::file_size(path);
    if (file_size > kMaximumCheckpointBytes) {
        throw std::runtime_error("checkpoint is unexpectedly large");
    }

    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(file_size));
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("could not open checkpoint for reading");
    }
    if (!bytes.empty()) {
        input.read(
            reinterpret_cast<char*>(bytes.data()),
            static_cast<std::streamsize>(bytes.size()));
    }
    if (!input || input.peek() != std::ifstream::traits_type::eof()) {
        throw std::runtime_error("could not read the complete checkpoint");
    }
    return bytes;
}

dth::CheckpointRecord parse_checkpoint(
    const std::vector<std::uint8_t>& bytes)
{
    std::size_t offset = 0;
    if (bytes.size() < kCheckpointMagic.size()) {
        throw std::runtime_error("checkpoint is truncated before its magic");
    }
    for (const std::uint8_t expected : kCheckpointMagic) {
        if (bytes[offset++] != expected) {
            throw std::runtime_error("checkpoint magic does not match");
        }
    }

    const std::uint32_t schema =
        read_little_endian<std::uint32_t>(bytes, offset);
    if (schema != kCheckpointSchemaVersion) {
        throw std::runtime_error("checkpoint schema version does not match");
    }

    const std::uint32_t config_length =
        read_little_endian<std::uint32_t>(bytes, offset);
    if (config_length != kCheckpointConfigId.size()
        || bytes.size() - std::min(bytes.size(), offset) < config_length) {
        throw std::runtime_error("checkpoint config id length does not match");
    }
    for (const char expected : kCheckpointConfigId) {
        if (bytes[offset++] != static_cast<std::uint8_t>(expected)) {
            throw std::runtime_error("checkpoint config id does not match");
        }
    }

    dth::CheckpointRecord record{};
    record.profile_count =
        read_little_endian<std::uint64_t>(bytes, offset);
    record.class_count = static_cast<dth::ClassId>(
        read_little_endian<std::uint64_t>(bytes, offset));
    record.completed_potential = std::bit_cast<std::int32_t>(
        read_little_endian<std::uint32_t>(bytes, offset));
    record.counters.pure =
        read_little_endian<std::uint64_t>(bytes, offset);
    record.counters.warm_support =
        read_little_endian<std::uint64_t>(bytes, offset);
    record.counters.full_support =
        read_little_endian<std::uint64_t>(bytes, offset);
    record.counters.linear_program =
        read_little_endian<std::uint64_t>(bytes, offset);

    if (offset != bytes.size()) {
        throw std::runtime_error("checkpoint contains trailing bytes");
    }
    validate_checkpoint_record(record);
    return record;
}

#ifdef _WIN32

[[noreturn]] void throw_checkpoint_windows_error(
    const char* operation,
    const std::filesystem::path& path,
    const DWORD error)
{
    throw std::system_error(
        static_cast<int>(error),
        std::system_category(),
        std::string(operation) + " '" + path.string() + "'");
}

void write_checkpoint_atomically(
    const std::filesystem::path& output_dir,
    const std::vector<std::uint8_t>& bytes)
{
    const std::filesystem::path temporary =
        temporary_checkpoint_path(output_dir);
    const std::filesystem::path final = checkpoint_path(output_dir);
    HANDLE file = CreateFileW(
        temporary.c_str(),
        GENERIC_WRITE,
        0,
        nullptr,
        CREATE_ALWAYS,
        FILE_ATTRIBUTE_NORMAL,
        nullptr);
    if (file == INVALID_HANDLE_VALUE) {
        throw_checkpoint_windows_error(
            "could not create temporary checkpoint",
            temporary,
            GetLastError());
    }

    std::size_t offset = 0;
    while (offset < bytes.size()) {
        const std::size_t remaining = bytes.size() - offset;
        const DWORD chunk = static_cast<DWORD>(std::min<std::size_t>(
            remaining,
            std::numeric_limits<DWORD>::max()));
        DWORD written = 0;
        if (!WriteFile(file, bytes.data() + offset, chunk, &written, nullptr)
            || written == 0) {
            const DWORD error = GetLastError();
            CloseHandle(file);
            DeleteFileW(temporary.c_str());
            throw_checkpoint_windows_error(
                "could not write temporary checkpoint",
                temporary,
                error);
        }
        offset += written;
    }

    if (!FlushFileBuffers(file)) {
        const DWORD error = GetLastError();
        CloseHandle(file);
        DeleteFileW(temporary.c_str());
        throw_checkpoint_windows_error(
            "could not flush temporary checkpoint",
            temporary,
            error);
    }
    if (!CloseHandle(file)) {
        const DWORD error = GetLastError();
        DeleteFileW(temporary.c_str());
        throw_checkpoint_windows_error(
            "could not close temporary checkpoint",
            temporary,
            error);
    }

    if (!MoveFileExW(
            temporary.c_str(),
            final.c_str(),
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        const DWORD error = GetLastError();
        DeleteFileW(temporary.c_str());
        throw_checkpoint_windows_error(
            "could not replace checkpoint",
            final,
            error);
    }
}

#else

[[noreturn]] void throw_checkpoint_posix_error(
    const char* operation,
    const std::filesystem::path& path,
    const int error)
{
    throw std::system_error(
        error,
        std::generic_category(),
        std::string(operation) + " '" + path.string() + "'");
}

void fsync_retry(
    const int file_descriptor,
    const std::filesystem::path& path,
    const char* operation)
{
    while (::fsync(file_descriptor) == -1) {
        if (errno == EINTR) {
            continue;
        }
        throw_checkpoint_posix_error(operation, path, errno);
    }
}

void write_checkpoint_atomically(
    const std::filesystem::path& output_dir,
    const std::vector<std::uint8_t>& bytes)
{
    const std::filesystem::path temporary =
        temporary_checkpoint_path(output_dir);
    const std::filesystem::path final = checkpoint_path(output_dir);
    const int file = ::open(
        temporary.c_str(),
        O_CREAT | O_TRUNC | O_WRONLY,
        0666);
    if (file == -1) {
        throw_checkpoint_posix_error(
            "could not create temporary checkpoint",
            temporary,
            errno);
    }

    std::size_t offset = 0;
    while (offset < bytes.size()) {
        const ssize_t written = ::write(
            file,
            bytes.data() + offset,
            bytes.size() - offset);
        if (written == -1) {
            if (errno == EINTR) {
                continue;
            }
            const int error = errno;
            ::close(file);
            ::unlink(temporary.c_str());
            throw_checkpoint_posix_error(
                "could not write temporary checkpoint",
                temporary,
                error);
        }
        if (written == 0) {
            ::close(file);
            ::unlink(temporary.c_str());
            throw std::runtime_error(
                "temporary checkpoint write made no progress");
        }
        offset += static_cast<std::size_t>(written);
    }

    try {
        fsync_retry(file, temporary, "could not flush temporary checkpoint");
    } catch (...) {
        ::close(file);
        ::unlink(temporary.c_str());
        throw;
    }
    if (::close(file) == -1) {
        const int error = errno;
        ::unlink(temporary.c_str());
        throw_checkpoint_posix_error(
            "could not close temporary checkpoint",
            temporary,
            error);
    }

    if (::rename(temporary.c_str(), final.c_str()) == -1) {
        const int error = errno;
        ::unlink(temporary.c_str());
        throw_checkpoint_posix_error(
            "could not replace checkpoint",
            final,
            error);
    }

    const int directory = ::open(output_dir.c_str(), O_RDONLY);
    if (directory == -1) {
        throw_checkpoint_posix_error(
            "could not open checkpoint directory",
            output_dir,
            errno);
    }
    try {
        fsync_retry(
            directory,
            output_dir,
            "could not flush checkpoint directory");
    } catch (...) {
        ::close(directory);
        throw;
    }
    if (::close(directory) == -1) {
        throw_checkpoint_posix_error(
            "could not close checkpoint directory",
            output_dir,
            errno);
    }
}

#endif

} // namespace

dth::DurableStores dth::create_stores(
    const std::filesystem::path& output_dir,
    const std::uint64_t profile_count,
    const ClassId class_count)
{
    if (profile_count == 0) {
        throw std::invalid_argument("profile count cannot be zero");
    }
    const std::size_t mapped_count = checked_class_count(class_count);
    std::filesystem::create_directories(output_dir);

    MappedArray<double> values = MappedArray<double>::create(
        output_dir / "values.bin",
        mapped_count,
        std::numeric_limits<double>::quiet_NaN());
    MappedArray<std::uint8_t> solver_kind =
        MappedArray<std::uint8_t>::create(
            output_dir / "solver_kind.bin",
            mapped_count,
            kUnsolvedKind);
    values.flush();
    solver_kind.flush();

    CheckpointRecord checkpoint{};
    checkpoint.profile_count = profile_count;
    checkpoint.class_count = class_count;
    atomically_write_checkpoint(output_dir, checkpoint);

    return DurableStores{
        std::move(values),
        std::move(solver_kind),
        checkpoint,
    };
}

void dth::atomically_write_checkpoint(
    const std::filesystem::path& output_dir,
    const CheckpointRecord& record)
{
    if (!std::filesystem::is_directory(output_dir)) {
        throw std::invalid_argument(
            "checkpoint output directory does not exist");
    }
    write_checkpoint_atomically(output_dir, serialize_checkpoint(record));
}

dth::DurableStores dth::open_resume(
    const std::filesystem::path& output_dir,
    const std::uint64_t expected_profile_count,
    const ClassId expected_class_count)
{
    if (expected_profile_count == 0) {
        throw std::invalid_argument("expected profile count cannot be zero");
    }
    const std::size_t mapped_count = checked_class_count(expected_class_count);
    const CheckpointRecord checkpoint = parse_checkpoint(
        read_checkpoint_bytes(checkpoint_path(output_dir)));

    if (checkpoint.profile_count != expected_profile_count) {
        throw std::runtime_error("checkpoint profile count does not match");
    }
    if (checkpoint.class_count != expected_class_count) {
        throw std::runtime_error("checkpoint class count does not match");
    }

    MappedArray<double> values = MappedArray<double>::open_existing(
        output_dir / "values.bin",
        mapped_count);
    MappedArray<std::uint8_t> solver_kind =
        MappedArray<std::uint8_t>::open_existing(
            output_dir / "solver_kind.bin",
            mapped_count);

    return DurableStores{
        std::move(values),
        std::move(solver_kind),
        checkpoint,
    };
}
