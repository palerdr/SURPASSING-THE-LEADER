#pragma once

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace dth {

template <typename T>
std::size_t MappedArray<T>::checked_byte_count(
    const std::size_t count
) {
    if (count == 0) {
        throw std::invalid_argument(
            "a mapped array cannot contain zero elements"
        );
    }

    if (
        count >
        std::numeric_limits<std::size_t>::max() / sizeof(T)
    ) {
        throw std::length_error(
            "mapped array byte count overflow"
        );
    }

    return count * sizeof(T);
}

template <typename T>
MappedArray<T>::MappedArray(
    MappedFile file,
    const std::size_t count
) noexcept
    : file_(std::move(file)),
      count_(count) {}

template <typename T>
MappedArray<T> MappedArray<T>::create(
    const std::filesystem::path& path,
    const std::size_t count,
    const T& initial_value
) {
    const std::size_t byte_count =
        checked_byte_count(count);

    MappedFile file =
        MappedFile::create(path, byte_count);

    MappedArray result{
        std::move(file),
        count
    };

    T* elements =
        static_cast<T*>(result.file_.data());

    constexpr std::size_t chunk_bytes =
        std::size_t{64} * 1024 * 1024;

    const std::size_t elements_per_chunk =
        std::max<std::size_t>(
            1,
            chunk_bytes / sizeof(T)
        );

    for (std::size_t offset = 0; offset < count;) {
        const std::size_t chunk_count =
            std::min(
                elements_per_chunk,
                count - offset
            );

        std::fill_n(
            elements + offset,
            chunk_count,
            initial_value
        );

        offset += chunk_count;
    }

    result.flush();
    return result;
}

template <typename T>
MappedArray<T> MappedArray<T>::open_existing(
    const std::filesystem::path& path,
    const std::size_t expected_count
) {
    const std::size_t byte_count =
        checked_byte_count(expected_count);

    MappedFile file =
        MappedFile::open_existing(
            path,
            byte_count
        );

    return MappedArray{
        std::move(file),
        expected_count
    };
}

template <typename T>
T& MappedArray<T>::operator[](
    const std::size_t index
) {
#ifndef NDEBUG
    if (index >= count_) {
        throw std::out_of_range(
            "mapped array index out of range"
        );
    }
#endif

    T* elements =
        static_cast<T*>(file_.data());

    return elements[index];
}

template <typename T>
const T& MappedArray<T>::operator[](
    const std::size_t index
) const {
#ifndef NDEBUG
    if (index >= count_) {
        throw std::out_of_range(
            "mapped array index out of range"
        );
    }
#endif

    const T* elements =
        static_cast<const T*>(file_.data());

    return elements[index];
}

template <typename T>
std::size_t MappedArray<T>::size() const noexcept {
    return count_;
}

template <typename T>
void MappedArray<T>::flush() {
    file_.flush();
}

} // namespace dth