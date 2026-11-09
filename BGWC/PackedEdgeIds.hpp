#ifndef BGWC_LARGEFAST_PACKED_EDGE_IDS_HPP
#define BGWC_LARGEFAST_PACKED_EDGE_IDS_HPP

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <omp.h>
#ifdef __linux__
#include <sys/resource.h>
#endif

// Lossless 32/40/64-bit edge-ID storage
class PackedEdgeIds {
public:
    using value_type = std::uint64_t;
    static constexpr value_type max_narrow_id = std::numeric_limits<std::uint32_t>::max();
    static constexpr value_type max_packed_id = (value_type{1} << 40) - 1;

    class Reference {
    public:
        Reference(PackedEdgeIds& owner, std::size_t index) noexcept
            : owner_(&owner), index_(index) {}
        operator value_type() const noexcept { return owner_->get(index_); }
        Reference& operator=(value_type value) { owner_->set(index_, value); return *this; }
        Reference& operator=(const Reference& other) {
            return *this = static_cast<value_type>(other);
        }
    private:
        PackedEdgeIds* owner_;
        std::size_t index_;
    };

    PackedEdgeIds() = default;

    static constexpr unsigned bytes_for_max_id(value_type maximum) noexcept {
        return maximum <= max_narrow_id ? 4u : maximum <= max_packed_id ? 5u : 8u;
    }

    void assign(std::size_t count, value_type value) {
        assign_with_bound(count, value,
            count == 0 ? value_type{0} : static_cast<value_type>(count - 1));
    }

    void assign_with_bound(std::size_t count, value_type value, value_type maximum_id) {
        const auto maximum = std::max(maximum_id, std::max(value,
            count == 0 ? value_type{0} : static_cast<value_type>(count - 1)));
        PackedEdgeIds replacement;
        replacement.size_ = count;
        replacement.maximum_bound_ = maximum;
        replacement.width_ = bytes_for_max_id(maximum);
        if (count > std::numeric_limits<std::size_t>::max() / replacement.width_)
            throw std::length_error("PackedEdgeIds byte-size overflow");
        if (replacement.width_ == 4) {
            replacement.low_.assign(count, static_cast<std::uint32_t>(value));
        } else if (replacement.width_ == 5) {
            replacement.low_.assign(count, static_cast<std::uint32_t>(value));
            replacement.high_.assign(count, static_cast<std::uint8_t>(value >> 32));
        } else {
            replacement.wide_.assign(count, value);
        }
        swap(replacement);
    }

    void resize(std::size_t count) {
        if (count == size_) return;
        if (size_ == 0) {
            assign_with_bound(count, 0, maximum_bound_);
            return;
        }
        const auto maximum = std::max(maximum_bound_,
            count == 0 ? value_type{0} : static_cast<value_type>(count - 1));
        const auto required = bytes_for_max_id(maximum);
        if (count > std::numeric_limits<std::size_t>::max() / required)
            throw std::length_error("PackedEdgeIds byte-size overflow");
        if (required == width_) {
            // Update storage before metadata.
            if (width_ == 5) {
                high_.resize(count, 0);
                try { low_.resize(count, 0); }
                catch (...) { high_.resize(size_); throw; }
            } else if (width_ == 4) {
                low_.resize(count, 0);
            } else {
                wide_.resize(count, 0);
            }
            size_ = count;
            maximum_bound_ = maximum;
            return;
        }
        // Resizing preserves the original ID range and never narrows storage.
        PackedEdgeIds expanded;
        expanded.assign_with_bound(count, 0, maximum);
        for (std::size_t i = 0; i < std::min(size_, count); ++i)
            expanded.set(i, get(i));
        swap(expanded);
    }

    std::size_t size() const noexcept { return size_; }
    bool empty() const noexcept { return size_ == 0; }
    unsigned bytes_per_id() const noexcept { return width_; }
    std::size_t storage_bytes() const noexcept { return size_ * width_; }
    value_type maximum_id_bound() const noexcept { return maximum_bound_; }

    Reference operator[](std::size_t index) noexcept {
        assert(index < size_);
        return Reference(*this, index);
    }
    value_type operator[](std::size_t index) const noexcept {
        assert(index < size_);
        return get(index);
    }

    void swap(PackedEdgeIds& other) noexcept {
        using std::swap;
        swap(size_, other.size_);
        swap(maximum_bound_, other.maximum_bound_);
        swap(width_, other.width_);
        low_.swap(other.low_);
        high_.swap(other.high_);
        wide_.swap(other.wide_);
    }

private:
    value_type get(std::size_t index) const noexcept {
        if (width_ == 4) return low_[index];
        if (width_ == 5)
            return static_cast<value_type>(low_[index]) |
                   (static_cast<value_type>(high_[index]) << 32);
        return wide_[index];
    }
    void set(std::size_t index, value_type value) {
        if (width_ == 4) {
            if (value > max_narrow_id)
                throw std::overflow_error("PackedEdgeIds identity exceeds 32 bits");
            low_[index] = static_cast<std::uint32_t>(value);
        } else if (width_ == 5) {
            if (value > max_packed_id)
                throw std::overflow_error("PackedEdgeIds identity exceeds 40 bits");
            low_[index] = static_cast<std::uint32_t>(value);
            high_[index] = static_cast<std::uint8_t>(value >> 32);
        } else {
            wide_[index] = value;
        }
    }

    std::size_t size_ = 0;
    value_type maximum_bound_ = 0;
    unsigned width_ = 4;
    std::vector<std::uint32_t> low_;
    std::vector<std::uint8_t> high_;
    std::vector<value_type> wide_;
};

namespace packed_storage_detail {

struct CompactionBudget {
    bool known = false;
    std::uint64_t additional_bytes = 0;
};

inline CompactionBudget compaction_budget() {
    CompactionBudget result;
#ifdef __linux__
    struct rlimit limits;
    if (getrlimit(RLIMIT_AS, &limits) != 0) return result;
    std::uint64_t virtual_bytes = 0, available_bytes = 0;
    bool have_virtual = false, have_available = false;
    std::ifstream status("/proc/self/status"), meminfo("/proc/meminfo");
    std::string line;
    unsigned long long kib = 0;
    while (std::getline(status, line)) {
        if (std::sscanf(line.c_str(), "VmSize: %llu kB", &kib) == 1) {
            if (kib > std::numeric_limits<std::uint64_t>::max()/1024) return result;
            virtual_bytes = static_cast<std::uint64_t>(kib)*1024;
            have_virtual = true;
            break;
        }
    }
    while (std::getline(meminfo, line)) {
        if (std::sscanf(line.c_str(), "MemAvailable: %llu kB", &kib) == 1) {
            if (kib > std::numeric_limits<std::uint64_t>::max()/1024) return result;
            available_bytes = static_cast<std::uint64_t>(kib)*1024;
            have_available = true;
            break;
        }
    }
    if (!have_virtual || !have_available) return result;
    constexpr std::uint64_t gib = std::uint64_t{1} << 30;
    std::uint64_t address_headroom = std::numeric_limits<std::uint64_t>::max();
    if (limits.rlim_cur != RLIM_INFINITY) {
        const auto cap = static_cast<std::uint64_t>(limits.rlim_cur);
        if (cap <= virtual_bytes || cap-virtual_bytes <= 2*gib) return {true,0};
        address_headroom = cap-virtual_bytes-2*gib;
    }
    if (available_bytes <= 12*gib) return {true,0};
    result.known = true;
    result.additional_bytes = std::min(address_headroom,available_bytes-12*gib);
#endif
    return result;
}

// Compute block boundaries without overflow.
inline std::size_t compaction_boundary(std::size_t count, std::size_t block,
                                      std::size_t blocks) {
    if (blocks == 0 || block > blocks)
        throw std::invalid_argument("invalid packed-compaction block boundary");
    static_assert(sizeof(std::size_t) <= sizeof(std::uint64_t), "unsupported size_t");
    return static_cast<std::size_t>(
        static_cast<unsigned __int128>(count)*block/blocks);
}

// Compact IDs and targets in stable order using the supplied prefix offsets.
inline void compact_stable_parallel(PackedEdgeIds& ids,
    std::vector<std::uint32_t>& compact_targets,
    const std::vector<std::uint8_t>& mask, std::uint8_t selected_bit,
    std::size_t expected, PackedEdgeIds::value_type maximum_original_id,
    const std::vector<std::size_t>& offsets,
    const CompactionBudget* test_budget = nullptr) {
    const std::size_t count = ids.size();
    if (mask.size() != count || compact_targets.size() != count ||
        selected_bit == 0 || (selected_bit & static_cast<std::uint8_t>(selected_bit-1)) != 0 ||
        expected > count || offsets.size() < 2 || offsets.front() != 0 ||
        offsets.back() != expected || maximum_original_id < ids.maximum_id_bound())
        throw std::invalid_argument("invalid largefast packed-compaction shape/domain");
    const std::size_t blocks = offsets.size()-1;
    for (std::size_t block=0; block<blocks; ++block) {
        if (offsets[block] > offsets[block+1] || offsets[block+1] > expected ||
            offsets[block+1]-offsets[block] >
                compaction_boundary(count,block+1,blocks)-compaction_boundary(count,block,blocks))
            throw std::invalid_argument("invalid largefast packed-compaction offsets");
    }
    const auto maximum = std::max(maximum_original_id,
        expected == 0 ? PackedEdgeIds::value_type{0} : static_cast<PackedEdgeIds::value_type>(expected-1));
    const unsigned width = PackedEdgeIds::bytes_for_max_id(maximum);
    constexpr std::uint64_t margin = 64*1024;
    const auto budget = test_budget ? *test_budget : compaction_budget();
    if (!budget.known)
        throw std::runtime_error("RESOURCE_LIMIT: largefast compaction budget unavailable");
    if (expected > (std::numeric_limits<std::uint64_t>::max()-margin)/(width+4u) ||
        static_cast<std::uint64_t>(expected)*(width+4u)+margin > budget.additional_bytes)
        throw std::runtime_error("RESOURCE_LIMIT: insufficient budget for largefast ID/target compaction");
    try {
        PackedEdgeIds replacement;
        replacement.assign_with_bound(expected,0,maximum_original_id);
        std::vector<std::uint32_t> replacement_targets(expected,0);
        int population_mismatch = 0;
        #pragma omp parallel for schedule(static,1) reduction(|:population_mismatch)
        for (std::size_t block=0; block<blocks; ++block) {
            const auto begin = compaction_boundary(count,block,blocks);
            const auto end = compaction_boundary(count,block+1,blocks);
            std::size_t write = offsets[block];
            for (std::size_t position=begin; position<end; ++position) {
                if ((mask[position]&selected_bit) == 0) continue;
                if (write < offsets[block+1]) {
                    replacement[write] = static_cast<PackedEdgeIds::value_type>(ids[position]);
                    replacement_targets[write] = compact_targets[position];
                } else {
                    population_mismatch = 1;
                }
                ++write;
            }
            if (write != offsets[block+1]) population_mismatch = 1;
        }
        if (population_mismatch)
            throw std::runtime_error("largefast prefix/scatter population mismatch");
        ids.swap(replacement);
        compact_targets.swap(replacement_targets);
    } catch (const std::bad_alloc&) {
        throw std::runtime_error("RESOURCE_LIMIT: largefast packed-compaction allocation failed");
    }
}

} // namespace packed_storage_detail
#endif
