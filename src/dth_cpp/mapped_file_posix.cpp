#include "durable_store.hpp"

#ifndef _WIN32

#include <cerrno>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace dth {
namespace {

[[noreturn]] void throw_posix_error(
    const char* operation,
    const std::filesystem::path& path,
    const int error
) {
    throw std::system_error(
        error,
        std::generic_category(),
        std::string(operation) + " '" + path.string() + "'"
    );
}

void remove_failed_creation(
    const int file_descriptor,
    const std::filesystem::path& path
) noexcept {
    ::close(file_descriptor);
    ::unlink(path.c_str());
}

void validate_mapping_size(const std::size_t byte_count) {
    if (byte_count == 0) {
        throw std::invalid_argument("a mapped file cannot contain zero bytes");
    }

    const auto maximum_file_size =
        static_cast<std::uintmax_t>(std::numeric_limits<off_t>::max());
    if (static_cast<std::uintmax_t>(byte_count) > maximum_file_size) {
        throw std::length_error("mapped file size exceeds the POSIX file-size limit");
    }
}

} // namespace

MappedFile MappedFile::create(
    const std::filesystem::path& path,
    const std::size_t byte_count
) {
    validate_mapping_size(byte_count);

    const int file_descriptor = ::open(
        path.c_str(),
        O_CREAT | O_EXCL | O_RDWR,
        0666
    );
    if (file_descriptor == -1) {
        throw_posix_error("could not create mapped file", path, errno);
    }

    if (::ftruncate(file_descriptor, static_cast<off_t>(byte_count)) == -1) {
        const int error = errno;
        remove_failed_creation(file_descriptor, path);
        throw_posix_error("could not resize mapped file", path, error);
    }

    void* const data = ::mmap(
        nullptr,
        byte_count,
        PROT_READ | PROT_WRITE,
        MAP_SHARED,
        file_descriptor,
        0
    );
    if (data == MAP_FAILED) {
        const int error = errno;
        remove_failed_creation(file_descriptor, path);
        throw_posix_error("could not map new file", path, error);
    }

    MappedFile result;
    result.file_descriptor_ = file_descriptor;
    result.data_ = data;
    result.byte_count_ = byte_count;
    return result;
}

MappedFile MappedFile::open_existing(
    const std::filesystem::path& path,
    const std::size_t expected_byte_count
) {
    validate_mapping_size(expected_byte_count);

    const int file_descriptor = ::open(path.c_str(), O_RDWR);
    if (file_descriptor == -1) {
        throw_posix_error("could not open mapped file", path, errno);
    }

    struct stat file_status {};
    if (::fstat(file_descriptor, &file_status) == -1) {
        const int error = errno;
        ::close(file_descriptor);
        throw_posix_error("could not inspect mapped file", path, error);
    }

    if (file_status.st_size != static_cast<off_t>(expected_byte_count)) {
        ::close(file_descriptor);
        throw std::runtime_error(
            "mapped file has the wrong size: '" + path.string() + "'"
        );
    }

    void* const data = ::mmap(
        nullptr,
        expected_byte_count,
        PROT_READ | PROT_WRITE,
        MAP_SHARED,
        file_descriptor,
        0
    );
    if (data == MAP_FAILED) {
        const int error = errno;
        ::close(file_descriptor);
        throw_posix_error("could not map existing file", path, error);
    }

    MappedFile result;
    result.file_descriptor_ = file_descriptor;
    result.data_ = data;
    result.byte_count_ = expected_byte_count;
    return result;
}

MappedFile::~MappedFile() {
    close();
}

MappedFile::MappedFile(MappedFile&& other) noexcept
    : file_descriptor_(std::exchange(other.file_descriptor_, -1)),
      data_(std::exchange(other.data_, nullptr)),
      byte_count_(std::exchange(other.byte_count_, 0)) {}

MappedFile& MappedFile::operator=(MappedFile&& other) noexcept {
    if (this != &other) {
        close();
        file_descriptor_ = std::exchange(other.file_descriptor_, -1);
        data_ = std::exchange(other.data_, nullptr);
        byte_count_ = std::exchange(other.byte_count_, 0);
    }
    return *this;
}

void* MappedFile::data() noexcept {
    return data_;
}

const void* MappedFile::data() const noexcept {
    return data_;
}

std::size_t MappedFile::byte_count() const noexcept {
    return byte_count_;
}

void MappedFile::flush() {
    if (data_ == nullptr || file_descriptor_ == -1 || byte_count_ == 0) {
        throw std::logic_error("cannot flush an empty or moved-from mapped file");
    }

    if (::msync(data_, byte_count_, MS_SYNC) == -1) {
        throw std::system_error(
            errno,
            std::generic_category(),
            "could not flush mapped memory"
        );
    }

    if (::fsync(file_descriptor_) == -1) {
        throw std::system_error(
            errno,
            std::generic_category(),
            "could not flush mapped file"
        );
    }
}

void MappedFile::close() noexcept {
    if (data_ != nullptr && byte_count_ != 0) {
        ::munmap(data_, byte_count_);
    }
    if (file_descriptor_ != -1) {
        ::close(file_descriptor_);
    }

    file_descriptor_ = -1;
    data_ = nullptr;
    byte_count_ = 0;
}

} // namespace dth

#endif
