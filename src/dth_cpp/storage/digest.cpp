#include "digest.hpp"
#include <algorithm>
#include <bit>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace dth {
void Sha256::block() {
    constexpr std::array<std::uint32_t, 64> k{
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2};
    std::array<std::uint32_t, 64> w{};
    for (std::size_t i = 0; i < 16; ++i)
        for (std::size_t j = 0; j < 4; ++j)
            w[i] = (w[i] << 8) | std::to_integer<std::uint32_t>(buffer_[4 * i + j]);
    for (std::size_t i = 16; i < 64; ++i) {
        auto x = w[i - 15], y = w[i - 2];
        w[i] = w[i - 16] + (std::rotr(x, 7) ^ std::rotr(x, 18) ^ (x >> 3)) + w[i - 7] +
               (std::rotr(y, 17) ^ std::rotr(y, 19) ^ (y >> 10));
    }
    auto [a, b, c, d, e, f, g, h] = state_;
    for (std::size_t i = 0; i < 64; ++i) {
        auto t1 = h + (std::rotr(e, 6) ^ std::rotr(e, 11) ^ std::rotr(e, 25)) +
                  ((e & f) ^ (~e & g)) + k[i] + w[i];
        auto t2 =
            (std::rotr(a, 2) ^ std::rotr(a, 13) ^ std::rotr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }
    const std::array<std::uint32_t, 8> v{a, b, c, d, e, f, g, h};
    for (std::size_t i = 0; i < 8; ++i)
        state_[i] += v[i];
}
void Sha256::update(std::span<const std::byte> bytes) {
    bytes_ += bytes.size();
    while (!bytes.empty()) {
        auto count = std::min(64 - used_, bytes.size());
        std::copy_n(bytes.begin(), count, buffer_.begin() + static_cast<std::ptrdiff_t>(used_));
        used_ += count;
        bytes = bytes.subspan(count);
        if (used_ == 64) {
            block();
            used_ = 0;
        }
    }
}
std::string Sha256::finish() {
    auto bits = bytes_ * 8;
    buffer_[used_++] = std::byte{0x80};
    if (used_ > 56) {
        std::fill(buffer_.begin() + static_cast<std::ptrdiff_t>(used_), buffer_.end(),
                  std::byte{0});
        block();
        used_ = 0;
    }
    std::fill(buffer_.begin() + static_cast<std::ptrdiff_t>(used_), buffer_.begin() + 56,
              std::byte{0});
    for (std::size_t i = 0; i < 8; ++i)
        buffer_[63 - i] = static_cast<std::byte>((bits >> (i * 8)) & 255);
    block();
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (auto x : state_)
        out << std::setw(8) << x;
    return out.str();
}
std::string sha256_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in)
        throw std::runtime_error("cannot hash file: " + path.string());
    Sha256 digest;
    std::array<std::byte, 65536> buffer{};
    while (in) {
        in.read(reinterpret_cast<char*>(buffer.data()), buffer.size());
        digest.update(std::span(buffer).first(static_cast<std::size_t>(in.gcount())));
    }
    if (!in.eof())
        throw std::runtime_error("file hash read failed");
    return digest.finish();
}
} // namespace dth
