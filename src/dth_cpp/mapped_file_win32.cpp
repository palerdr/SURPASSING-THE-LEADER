#include "durable_store.hpp"

#ifdef _WIN32

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <limits>

namespace dth {

namespace {
    [[noreturn]] void throw_win32_error(
        const char* operation,
        const std::filesystem::path& path,
        const int error
    ) {
        throw std::system_error(
            error,
            std::system_category(),
            std::string(operation) + " '" + path.string() + "'"
        );
}

void validate_mapping_size(const std::size_t byte_count) {
    if (byte_count == 0) {
        throw std::invalid_argument("a mapped file cannot contain zero bytes");
    }

    const auto maximum_file_size =
    static_cast<std::uintmax_t>(
        std::numeric_limits<LONGLONG>::max()
    );
    if (static_cast<std::uintmax_t>(byte_count) > maximum_file_size) {
        throw std::length_error(
            "mapped file size exceeds the WIN32 file-size limit"
        );
        }
    }
} //end of anonymous namespace

    MappedFile MappedFile::create(
        const std::filesystem::path& path,
        const std::size_t byte_count
    ) {
        validate_mapping_size(byte_count);
        HANDLE file = CreateFileW(
            path.c_str(),
            GENERIC_READ | GENERIC_WRITE,
            0,
            nullptr,
            CREATE_NEW,
            FILE_ATTRIBUTE_NORMAL,
            nullptr
        );

        if (file == INVALID_HANDLE_VALUE) {
            DWORD error = GetLastError();
            throw_win32_error("could not create mapped file", path, static_cast<int>(error));
        }

        LARGE_INTEGER size{};
        size.QuadPart = static_cast<LONGLONG>(byte_count);

        if (!SetFilePointerEx(file, size, nullptr, FILE_BEGIN) || !SetEndOfFile(file)) {
            DWORD error = GetLastError();
            CloseHandle(file);
            DeleteFileW(path.c_str());
            throw_win32_error("could not set file size", path, static_cast<int>(error));
        }

        HANDLE mapping = CreateFileMappingW(
            file,
            nullptr,
            PAGE_READWRITE,
            0,
            0,
            nullptr
        );

        if (mapping == nullptr) {
            DWORD error = GetLastError();
            CloseHandle(file);
            DeleteFileW(path.c_str());
            throw_win32_error("could not create file mapping", path, static_cast<int>(error));
        }

        void* data = MapViewOfFile(
            mapping,
            FILE_MAP_ALL_ACCESS,
            0,
            0,
            0
        );

        if (data == nullptr) {
            DWORD error = GetLastError();
            CloseHandle(mapping);
            CloseHandle(file);
            DeleteFileW(path.c_str());
            throw_win32_error("could not map view of file", path, static_cast<int>(error));
        }

        //return the mapped file
        MappedFile result;
        result.file_handle_ = file;
        result.mapping_handle_ = mapping;
        result.data_ = data;
        result.byte_count_ = byte_count;
        return result;

    }

    MappedFile MappedFile::open_existing(
        const std::filesystem::path& path,
        const std::size_t expected_byte_count
    ) {
        validate_mapping_size(expected_byte_count);

        HANDLE file = CreateFileW(
            path.c_str(),
            GENERIC_READ | GENERIC_WRITE,
            0,
            nullptr,
            OPEN_EXISTING,
            FILE_ATTRIBUTE_NORMAL,
            nullptr
        );
        if (file == INVALID_HANDLE_VALUE) {
            DWORD error = GetLastError();
            throw_win32_error("could not open existing file", path, static_cast<int>(error));
        }

        LARGE_INTEGER actual_size{};

        if(!GetFileSizeEx(file, &actual_size)) {
            DWORD error = GetLastError();
            CloseHandle(file);
            throw_win32_error("could not query file size", path, static_cast<int>(error));
        }

        if (actual_size.QuadPart != static_cast<LONGLONG>(expected_byte_count)) {
            CloseHandle(file);
            throw std::runtime_error("mapped file has the wrong size");
        }

        HANDLE mapping = CreateFileMappingW(
            file,
            nullptr,
            PAGE_READWRITE,
            0,
            0,
            nullptr
        );

        if (mapping == nullptr) {
            DWORD error = GetLastError();
            CloseHandle(file);
            throw_win32_error("could not create file mapping", path, static_cast<int>(error));
        }

        void* data = MapViewOfFile(
            mapping,
            FILE_MAP_ALL_ACCESS,
            0,
            0,
            0
        );

        if (data == nullptr) {
            DWORD error = GetLastError();
            CloseHandle(mapping);
            CloseHandle(file);
            throw_win32_error("could not map view of file", path, static_cast<int>(error));
        }

        //return the mapped file
        MappedFile result;
        result.file_handle_ = file;
        result.mapping_handle_ = mapping;
        result.data_ = data;
        result.byte_count_ = expected_byte_count;
        return result;
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

    void MappedFile::close() noexcept {
        if (data_ != nullptr) {
            UnmapViewOfFile(data_);
        }

        if (mapping_handle_ != nullptr) {
            CloseHandle(static_cast<HANDLE>(mapping_handle_));
        }

        if (file_handle_ != nullptr) {
            CloseHandle(static_cast<HANDLE>(file_handle_));
        }

        data_ = nullptr;
        mapping_handle_ = nullptr;
        file_handle_ = nullptr;
        byte_count_ = 0;
    }
    //now the destructor can just call the close method
    MappedFile::~MappedFile() {
        close();
    }

    //moving
    MappedFile::MappedFile(MappedFile&& other) noexcept
    : file_handle_(std::exchange(other.file_handle_, nullptr)),
      mapping_handle_(std::exchange(other.mapping_handle_, nullptr)),
      data_(std::exchange(other.data_, nullptr)),
      byte_count_(std::exchange(other.byte_count_, 0)) {}

    MappedFile& MappedFile::operator=(MappedFile&& other) noexcept {
    if (this != &other) {
        close();
        file_handle_ = std::exchange(other.file_handle_, nullptr);
        mapping_handle_ = std::exchange(other.mapping_handle_, nullptr);
        data_ = std::exchange(other.data_, nullptr);
        byte_count_ = std::exchange(other.byte_count_, 0);
        }
    return *this;
    }

    void MappedFile::flush() {
        if (data_ == nullptr || file_handle_ == nullptr || mapping_handle_ == nullptr || byte_count_ == 0) {
            throw std::logic_error("cannot flush an empty or moved-from mapped file");
        }
        auto flush_view = FlushViewOfFile(data_, byte_count_);
        if (flush_view == FALSE) {
            DWORD error = GetLastError();
            throw std::system_error(
            static_cast<int>(error),
            std::system_category(),
            "could not flush mapped memory"
        );
        }
        auto flushbuff = FlushFileBuffers(static_cast<HANDLE>(file_handle_));
        if (flushbuff == FALSE) {
            DWORD error = GetLastError();
            throw std::system_error(
            static_cast<int>(error),
            std::system_category(),
            "could not flush file buffers"
        );
    }
}

} // namespace dth

#endif
