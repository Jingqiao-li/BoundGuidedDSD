#include "PackedEdgeIds.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include <omp.h>

using std::uint32_t;
using std::uint64_t;
// Wide edge and bitmap indices.
using EdgeId = uint64_t;
using WordId = uint64_t;
static_assert(sizeof(size_t) >= 8, "64-bit address space required");
static_assert(sizeof(EdgeId) == 8 && sizeof(WordId) == 8, "wide index required");


struct FrontOptions {
    bool no_candidate_reduction = false;
    bool no_cover_cut = false;
    bool independent_closures = false;
    bool work_count = false;
    bool trace_closures = false;
    static constexpr uint32_t probe_limit = 7; 
};
static FrontOptions front_options;
struct FrontStats {
    uint64_t batches=0, queries=0, max_batch_width=0, closure_rounds=0;
    uint64_t independent_lane_rounds=0, strict_rounds=0;
    uint64_t initial_strict_edges=0, initial_nonstrict_edges=0;
    uint64_t cover_source_cut=0, cover_target_cut=0, cover_estimated_cost=0;
    bool cover_cut_selection_executed=false;
};
static FrontStats front_stats;
struct alignas(64) WorkSlot {
    uint64_t endpoint_reads=0, reverse_index_reads=0, degree_products=0;
    uint64_t prefix_skipped_checks=0;
};
static std::vector<WorkSlot> work_slots;
static bool work_active=false;
static inline uint32_t work_endpoint(uint32_t value) {
#ifdef BGWC_WORK_COUNT
    if(work_active) ++work_slots[static_cast<size_t>(omp_get_thread_num())].endpoint_reads;
#endif
    return value;
}
static inline EdgeId work_reverse_index(EdgeId value) {
#ifdef BGWC_WORK_COUNT
    if(work_active) ++work_slots[static_cast<size_t>(omp_get_thread_num())].reverse_index_reads;
#endif
    return value;
}
static inline void work_product() {
#ifdef BGWC_WORK_COUNT
    if(work_active) ++work_slots[static_cast<size_t>(omp_get_thread_num())].degree_products;
#endif
}
static inline void work_prefix_skip(uint64_t count) {
#ifdef BGWC_WORK_COUNT
    if(work_active) work_slots[static_cast<size_t>(omp_get_thread_num())].prefix_skipped_checks+=count;
#endif
}
static WorkSlot work_totals() {
    WorkSlot total;
    for(const auto& x:work_slots) {
        total.endpoint_reads+=x.endpoint_reads;
        total.reverse_index_reads+=x.reverse_index_reads;
        total.degree_products+=x.degree_products;
        total.prefix_skipped_checks+=x.prefix_skipped_checks;
    }
    return total;
}

struct Timer {
    std::chrono::steady_clock::time_point t0;
    Timer() : t0(std::chrono::steady_clock::now()) {}
    double elapsed() const {
        auto t1 = std::chrono::steady_clock::now();
        return std::chrono::duration<double>(t1 - t0).count();
    }
};

class FastInput {
public:
    explicit FastInput(const std::string& path, std::uint64_t offset = 0, std::size_t buf_sz = (1u << 20))
        : fp_(std::fopen(path.c_str(), "rb")), buf_(buf_sz), idx_(0), size_(0) {
        if (!fp_) throw std::runtime_error("cannot open input file: " + path);
        if (offset != 0 && std::fseek(fp_, static_cast<long>(offset), SEEK_SET) != 0) {
            throw std::runtime_error("cannot seek input file: " + path);
        }
    }

    ~FastInput() {
        if (fp_) std::fclose(fp_);
    }

    bool readUInt64(uint64_t& out) {
        int c = nextNonSpace();
        if (c == EOF) return false;
        if (c < '0' || c > '9') {
            throw std::runtime_error("bad txt input: non-numeric token");
        }
        uint64_t x = 0;
        while (c != EOF && c > ' ') {
            if (c < '0' || c > '9') {
                throw std::runtime_error("bad txt input: non-numeric token");
            }
            x = x * 10ull + static_cast<uint64_t>(c - '0');
            c = nextChar();
        }
        out = x;
        return true;
    }

private:
    FILE* fp_;
    std::vector<char> buf_;
    std::size_t idx_;
    std::size_t size_;

    inline int refill() {
        size_ = std::fread(buf_.data(), 1, buf_.size(), fp_);
        idx_ = 0;
        return size_ ? 0 : EOF;
    }

    inline int nextChar() {
        if (idx_ >= size_) {
            if (refill() == EOF) return EOF;
        }
        return static_cast<unsigned char>(buf_[idx_++]);
    }

    inline void skipLine() {
        int c = nextChar();
        while (c != EOF && c != '\n') c = nextChar();
    }

    inline int nextNonSpace() {
        while (true) {
            int c = nextChar();
            while (c != EOF && c <= ' ') c = nextChar();
            if (c == '#' || c == '%' || c == '/') {
                skipLine();
                continue;
            }
            return c;
        }
    }
};

struct CSRGraph {
    uint32_t n = 0;
    uint64_t m = 0;
    uint64_t min_input_id = 0;
    uint64_t max_input_id = 0;
    uint64_t id_shift = 0;
    std::vector<uint64_t> off;      // size n+1
    std::vector<uint32_t> dst;      // size m
    std::vector<uint32_t> src;      // size m
    std::vector<uint64_t> in_off;   // size n+1
    PackedEdgeIds in_edges; // edge ids grouped by destination
    std::vector<uint32_t> out0;     // initial out-degree
    std::vector<uint32_t> in0;      // initial in-degree
};

static inline bool has_ext(const std::string& s, const std::string& ext) {
    return s.size() >= ext.size() && s.compare(s.size() - ext.size(), ext.size(), ext) == 0;
}

static inline bool is_comment_or_empty(const std::string& line) {
    for (char c : line) {
        if (c == ' ' || c == '\t' || c == '\r' || c == '\n') continue;
        return c == '#' || c == '%' || c == '/';
    }
    return true;
}

static std::vector<uint64_t> parse_uints_from_line(const std::string& line) {
    std::vector<uint64_t> vals;
    const char* s = line.c_str();
    while (*s != '\0') {
        while (*s == ' ' || *s == '\t' || *s == '\r') ++s;
        if (*s == '\0' || *s == '\n' || *s == '#' || *s == '%' || *s == '/') break;
        char* end = nullptr;
        uint64_t x = std::strtoull(s, &end, 10);
        if (end == s) throw std::runtime_error("bad txt header: non-numeric token");
        vals.push_back(x);
        s = end;
    }
    return vals;
}

#ifdef BGWC_MEMORY_DIAGNOSTICS
static void memory_stage(const char* label) {
    std::fprintf(stderr, "[memory-stage] %s\n", label);
    FILE* status = std::fopen("/proc/self/status", "r");
    if (status) {
        char line[256];
        while (std::fgets(line, sizeof(line), status)) {
            if (std::strncmp(line, "VmSize:", 7) == 0 ||
                std::strncmp(line, "VmRSS:", 6) == 0 ||
                std::strncmp(line, "VmPeak:", 7) == 0 ||
                std::strncmp(line, "VmHWM:", 6) == 0)
                std::fputs(line, stderr);
        }
        std::fclose(status);
    }
    std::fflush(stderr);
}
#else
static inline void memory_stage(const char*) {}
#endif

static void build_in_index(CSRGraph& g) {
    if (g.m > std::numeric_limits<size_t>::max() / sizeof(EdgeId)) {
        throw std::runtime_error("edge index allocation exceeds address-space size");
    }

    g.in_off.assign(static_cast<size_t>(g.n) + 1, 0);
    for (uint32_t v : g.dst) {
        ++g.in_off[v + 1];
    }
    for (uint32_t v = 0; v < g.n; ++v) {
        g.in_off[v + 1] += g.in_off[v];
    }

    memory_stage("build_in_index.in_edges.allocate.begin");
    g.in_edges.assign(g.m, 0);
    memory_stage("build_in_index.in_edges.allocate.complete");
    std::vector<uint64_t> cur = g.in_off;
    for (uint64_t e = 0; e < g.m; ++e) {
        uint32_t v = work_endpoint(g.dst[e]);
        g.in_edges[cur[v]++] = static_cast<EdgeId>(e);
    }
}

static CSRGraph load_txt_graph_dense_range(const std::string& path) {
    CSRGraph g;

    struct Edge {
        uint32_t u, v;
        bool operator<(const Edge& other) const {
            if (u != other.u) return u < other.u;
            return v < other.v;
        }
        bool operator==(const Edge& other) const {
            return u == other.u && v == other.v;
        }
    };

    uint64_t declared_n = 0;
    uint64_t declared_m = 0;
    uint64_t data_offset = 0;
    {
        std::ifstream fin(path, std::ios::binary);
        if (!fin) throw std::runtime_error("cannot open input file: " + path);
        std::string line;
        bool got_header = false;
        while (std::getline(fin, line)) {
            std::streampos pos = fin.tellg();
            if (pos == std::streampos(-1)) {
                fin.clear();
                fin.seekg(0, std::ios::end);
                pos = fin.tellg();
            }
            if (is_comment_or_empty(line)) continue;
            std::vector<uint64_t> header = parse_uints_from_line(line);
            if (header.size() < 2) throw std::runtime_error("bad txt input: expected a graph header");
            if (header.size() == 2) {
                declared_n = header[0];
                declared_m = header[1];
            } else {
                // Accept the alternate bipartite header format.
                declared_m = header[0];
                declared_n = std::max(header[1], header[2]);
            }
            data_offset = static_cast<uint64_t>(pos);
            got_header = true;
            break;
        }
        if (!got_header) throw std::runtime_error("bad txt input: missing graph header");
    }

    std::vector<Edge> edges;
    if (declared_m > 0 && declared_m < (1ull << 62)) {
        edges.reserve(static_cast<std::size_t>(declared_m));
    }

    FastInput in(path, data_offset);
    uint64_t uu = 0, vv = 0;
    uint64_t min_vid = std::numeric_limits<uint64_t>::max();
    uint64_t max_vid = 0;
    bool first_edge = true;
    while (in.readUInt64(uu)) {
        if (!in.readUInt64(vv)) break;
        if (uu == vv) continue;
        if (uu > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) ||
            vv > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
            throw std::runtime_error("vertex id exceeds uint32_t range");
        }
        min_vid = std::min(min_vid, std::min(uu, vv));
        max_vid = std::max(max_vid, std::max(uu, vv));
        first_edge = false;
        edges.push_back(Edge{static_cast<uint32_t>(uu), static_cast<uint32_t>(vv)});
    }

    uint64_t shift = (!first_edge && min_vid >= 1 && declared_n > 0 && max_vid <= declared_n) ? 1 : 0;
    for (auto& e : edges) {
        e.u = static_cast<uint32_t>(static_cast<uint64_t>(e.u) - shift);
        e.v = static_cast<uint32_t>(static_cast<uint64_t>(e.v) - shift);
    }

    uint64_t n64 = declared_n;
    if (!first_edge) {
        uint64_t shifted_max = max_vid - shift;
        n64 = std::max(n64, shifted_max + 1);
    }
    if (n64 > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        throw std::runtime_error("vertex count exceeds uint32_t range");
    }

    std::sort(edges.begin(), edges.end());
    edges.erase(std::unique(edges.begin(), edges.end()), edges.end());

    g.n = static_cast<uint32_t>(n64);
    g.m = static_cast<uint64_t>(edges.size());
    g.min_input_id = first_edge ? 0 : min_vid;
    g.max_input_id = first_edge ? (g.n ? static_cast<uint64_t>(g.n - 1) : 0) : max_vid;
    g.id_shift = shift;

    g.out0.assign(g.n, 0);
    g.in0.assign(g.n, 0);
    for (const auto& e : edges) {
        ++g.out0[e.u];
        ++g.in0[e.v];
    }

    g.off.assign(static_cast<size_t>(g.n) + 1, 0);
    for (uint32_t i = 0; i < g.n; ++i) g.off[i + 1] = g.off[i] + g.out0[i];
    g.dst.assign(g.m, 0);
    g.src.assign(g.m, 0);
    std::vector<uint64_t> cur = g.off;
    for (const auto& e : edges) {
        uint64_t pos = cur[e.u]++;
        g.dst[pos] = e.v;
        g.src[pos] = e.u;
    }
    build_in_index(g);
    return g;
}

struct State {
    std::vector<uint8_t> alive;          // 1 if current edge is alive
    std::vector<uint32_t> out_deg;       // only owner thread writes a row's out-degree
    std::unique_ptr<std::atomic<uint32_t>[]> in_deg; // multiple threads update/read
    std::vector<uint32_t> active_sources;
    uint64_t alive_edges = 0;
};

static State copy_state(const State& src, uint32_t n) {
    State dst;
    dst.alive = src.alive;
    dst.out_deg = src.out_deg;
    dst.in_deg.reset(new std::atomic<uint32_t>[n]);
    for (uint32_t i = 0; i < n; ++i) {
        dst.in_deg[i].store(src.in_deg[i].load(std::memory_order_relaxed), std::memory_order_relaxed);
    }
    dst.active_sources = src.active_sources;
    dst.alive_edges = src.alive_edges;
    return dst;
}

static void rebuild_active_sources(State& st) {
    size_t wr = 0;
    for (size_t i = 0; i < st.active_sources.size(); ++i) {
        uint32_t u = st.active_sources[i];
        if (st.out_deg[u] > 0) st.active_sources[wr++] = u;
    }
    st.active_sources.resize(wr);
}

static inline uint64_t edge_weight(uint32_t out_degree, uint32_t in_degree) {
    work_product();
    return static_cast<uint64_t>(out_degree) * static_cast<uint64_t>(in_degree);
}

static inline bool edge_weight_lt(uint32_t out_degree, uint32_t in_degree, uint64_t threshold) {
    return edge_weight(out_degree, in_degree) < threshold;
}

static inline bool edge_weight_eq(uint32_t out_degree, uint32_t in_degree, uint64_t threshold) {
    return edge_weight(out_degree, in_degree) == threshold;
}

static uint64_t apply_sparse_deletions(const CSRGraph& g, State& st, const std::vector<EdgeId>& edges) {
    uint64_t removed = 0;

    #pragma omp parallel for schedule(dynamic, 4096) reduction(+:removed)
    for (size_t i = 0; i < edges.size(); ++i) {
        uint64_t e = edges[i];
        if (!st.alive[e]) continue;
        st.alive[e] = 0;
        ++removed;
        uint32_t u = work_endpoint(g.src[e]);
        uint32_t v = work_endpoint(g.dst[e]);
        #pragma omp atomic update
        st.out_deg[u] -= 1;
        st.in_deg[v].fetch_sub(1, std::memory_order_relaxed);
    }

    if (removed != 0) {
        st.alive_edges -= removed;
        rebuild_active_sources(st);
    }
    return removed;
}

struct XYPairResult {
    bool found = false;
    uint32_t x = 0;
    uint32_t y = 0;
};

static XYPairResult find_xy_from_wstar_induced(const CSRGraph& g, State& st, uint64_t wstar, const std::vector<uint32_t>& candidates) {
    XYPairResult ans;
    int p = omp_get_max_threads();
    std::vector<std::vector<EdgeId>> local_marked(p);
    std::vector<EdgeId> marked_edges;

    for (uint32_t dstar : candidates) {
        if (st.alive_edges == 0) return ans;
        if (dstar == 0 || wstar % dstar != 0) continue;
        uint64_t x64 = wstar / dstar;
        if (x64 == 0 || x64 > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) continue;
        uint32_t xstar = static_cast<uint32_t>(x64);

        while (st.alive_edges != 0) {
            uint64_t marked = 0;
            int candidate_found = 0;
            for (auto& v : local_marked) v.clear();
            #pragma omp parallel reduction(+:marked) reduction(max:candidate_found)
            {
                int tid = omp_get_thread_num();
                auto& out = local_marked[tid];
                #pragma omp for schedule(dynamic, 256) nowait
                for (size_t ai = 0; ai < st.active_sources.size(); ++ai) {
                    uint32_t u = st.active_sources[ai];
                    uint32_t ou = st.out_deg[u];
                    if (ou == 0) continue;
                    for (uint64_t e = g.off[u]; e < g.off[u + 1]; ++e) {
                        if (!st.alive[e]) continue;
                        uint32_t v = work_endpoint(g.dst[e]);
                        uint32_t iv = st.in_deg[v].load(std::memory_order_relaxed);
                        if (edge_weight_lt(ou, iv, wstar)) {
                            out.push_back(static_cast<EdgeId>(e));
                            ++marked;
                        } else if (iv == dstar && edge_weight_eq(ou, dstar, wstar)) {
                            out.push_back(static_cast<EdgeId>(e));
                            ++marked;
                            candidate_found = 1;
                        }
                    }
                }
            }

            if (candidate_found) {
                ans.found = true;
                ans.x = xstar;
                ans.y = dstar;
            }
            if (marked == 0) break;

            size_t total = 0;
            for (auto& v : local_marked) total += v.size();
            marked_edges.clear();
            marked_edges.reserve(total);
            for (auto& v : local_marked) {
                marked_edges.insert(marked_edges.end(), v.begin(), v.end());
            }
            uint64_t removed = apply_sparse_deletions(g, st, marked_edges);
            if (removed == 0) break;
            if (st.alive_edges == 0) {
                ans.found = true;
                ans.x = xstar;
                ans.y = dstar;
                return ans;
            }
        }
    }
    return ans;
}

template<class Index>
static void append_thread_vectors(std::vector<Index>& out, std::vector<std::vector<Index>>& local) {
    // Concatenate thread-local results in order.
    std::vector<size_t> offsets(local.size() + 1, 0);
    for (size_t slot = 0; slot < local.size(); ++slot)
        offsets[slot + 1] = offsets[slot] + local[slot].size();
    out.resize(offsets.back());
    #pragma omp parallel for schedule(static, 1)
    for (size_t slot = 0; slot < local.size(); ++slot) {
        std::copy(local[slot].begin(), local[slot].end(), out.begin() + offsets[slot]);
        local[slot].clear();
    }
}

struct CoreStats {
    uint64_t edges = 0;
    uint32_t S = 0;
    uint32_t T = 0;
    double density = 0.0;
};


static State extract_initial_candidate(
    const CSRGraph& g,
    const State& h,
    uint32_t x,
    uint32_t y) {
    State result;
    result.alive.assign(g.m, 0);
    result.out_deg.assign(g.n, 0);
    result.in_deg.reset(new std::atomic<uint32_t>[g.n]);

    std::vector<uint8_t> target_eligible(g.n, 0);
    #pragma omp parallel for schedule(static)
    for (uint32_t v = 0; v < g.n; ++v) {
        result.in_deg[v].store(0, std::memory_order_relaxed);
        target_eligible[v] =
            h.in_deg[v].load(std::memory_order_relaxed) >= y;
    }

    uint64_t selected_edges = 0;
    #pragma omp parallel for schedule(dynamic, 256) reduction(+:selected_edges)
    for (size_t ai = 0; ai < h.active_sources.size(); ++ai) {
        uint32_t u = h.active_sources[ai];
        uint32_t eligible_count = 0;
        for (uint64_t e = g.off[u]; e < g.off[u + 1]; ++e) {
            if (h.alive[e] && target_eligible[work_endpoint(g.dst[e])]) ++eligible_count;
        }
        if (eligible_count < x) continue;

        result.out_deg[u] = eligible_count;
        selected_edges += eligible_count;
        for (uint64_t e = g.off[u]; e < g.off[u + 1]; ++e) {
            if (!h.alive[e] || !target_eligible[work_endpoint(g.dst[e])]) continue;
            result.alive[e] = 1;
            result.in_deg[work_endpoint(g.dst[e])].fetch_add(1, std::memory_order_relaxed);
        }
    }

    result.alive_edges = selected_edges;
    result.active_sources.reserve(h.active_sources.size());
    for (uint32_t u : h.active_sources) {
        if (result.out_deg[u] != 0) result.active_sources.push_back(u);
    }
    return result;
}

static XYPairResult xy_from_single_candidate(uint64_t wstar, const std::vector<uint32_t>& candidates) {
    XYPairResult xy;
    if (candidates.size() != 1) return xy;
    uint32_t y = candidates[0];
    if (y == 0 || wstar % y != 0) return xy;
    uint64_t x64 = wstar / y;
    if (x64 == 0 || x64 > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) return xy;
    xy.found = true;
    xy.x = static_cast<uint32_t>(x64);
    xy.y = y;
    return xy;
}

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

static inline uint32_t transcript_popcount_range(
    const std::vector<uint64_t>& transcript,
    uint64_t begin,
    uint64_t end) {
    if (begin >= end) return 0;
    const uint64_t first_word = begin >> 6;
    const uint64_t last_word = (end - 1) >> 6;
    uint32_t count = 0;
    for (uint64_t word_index = first_word; word_index <= last_word; ++word_index) {
        uint64_t bits = transcript[word_index];
        if (word_index == first_word) bits &= ~uint64_t{0} << (begin & 63U);
        if (word_index == last_word && (end & 63U) != 0) {
            bits &= (uint64_t{1} << (end & 63U)) - 1;
        }
        count += static_cast<uint32_t>(__builtin_popcountll(bits));
    }
    return count;
}

static inline void atomic_set_transcript_bit(
    std::vector<uint64_t>& transcript,
    uint64_t edge) {
    __atomic_fetch_or(
        &transcript[edge >> 6],
        uint64_t{1} << (edge & 63U),
        __ATOMIC_RELAXED);
}

static inline void merge_source_row_word(
    std::vector<uint64_t>& transcript,
    uint64_t word_index,
    uint64_t mask,
    uint64_t row_begin,
    uint64_t row_end) {
    if (mask == 0) return;
    const uint64_t word_begin = word_index << 6;
    const uint64_t word_end = word_begin + 64;
    if (row_begin <= word_begin && word_end <= row_end) {
        transcript[word_index] = mask;
    } else {
        __atomic_fetch_or(
            &transcript[word_index], mask, __ATOMIC_RELAXED);
    }
}

struct WordPruneStats {
    uint64_t removed = 0;
};

static inline WordPruneStats prune_transcript_word(
    const CSRGraph& g,
    std::vector<uint64_t>& transcript,
    uint64_t word_index,
    uint64_t threshold,
    std::vector<uint32_t>& source_degree,
    const std::vector<uint32_t>& target_compact_index,
    std::vector<uint32_t>& target_degree,
    std::vector<uint32_t>& local_target_decrement) {
    WordPruneStats stats;
    uint64_t bits = transcript[word_index];
    uint64_t kept = bits;
    uint32_t pending_source = std::numeric_limits<uint32_t>::max();
    uint32_t pending_source_removals = 0;
    bool pending_source_is_local = false;
    auto flush_source = [&]() {
        if (pending_source_removals != 0) {
            if (pending_source_is_local) {
                source_degree[pending_source] -= pending_source_removals;
            } else {
                __atomic_fetch_sub(
                    &source_degree[pending_source],
                    pending_source_removals,
                    __ATOMIC_RELAXED);
            }
        }
        pending_source_removals = 0;
    };
    while (bits != 0) {
        const unsigned bit = static_cast<unsigned>(__builtin_ctzll(bits));
        bits &= bits - 1;
        const uint64_t edge = (word_index << 6) + bit;
        const uint32_t u = work_endpoint(g.src[edge]);
        const uint32_t v = work_endpoint(g.dst[edge]);
        if (u != pending_source) {
            flush_source();
            pending_source = u;
            const uint64_t word_begin = word_index << 6;
            pending_source_is_local = g.off[u] >= word_begin &&
                g.off[static_cast<size_t>(u) + 1] <= word_begin + 64;
        }
        const uint32_t current_source_degree =
            (pending_source_is_local ? source_degree[u] :
                __atomic_load_n(&source_degree[u], __ATOMIC_RELAXED)) -
            pending_source_removals;
        const uint32_t compact_target = target_compact_index[v];
        const uint32_t current_target_degree =
            __atomic_load_n(&target_degree[v], __ATOMIC_RELAXED) -
            local_target_decrement[compact_target];
        if (edge_weight(current_source_degree, current_target_degree) > threshold) {
            continue;
        }
        kept &= ~(uint64_t{1} << bit);
        ++pending_source_removals;
        ++local_target_decrement[compact_target];
        ++stats.removed;
    }
    flush_source();
    transcript[word_index] = kept;
    return stats;
}

// Sparse closure with word ownership and atomic source updates.
static inline WordPruneStats prune_sparse_reconstruction_word(
    const CSRGraph& g,
    std::vector<uint64_t>& transcript,
    WordId word_index,
    uint64_t threshold,
    std::vector<uint32_t>& source_degree,
    const std::vector<uint32_t>& target_index,
    const std::vector<uint32_t>& compact_target_degree,
    std::vector<uint32_t>& local_target_decrement,
    std::vector<uint32_t>& local_touched_targets) {
    WordPruneStats stats;
    uint64_t bits = transcript[word_index];
    uint64_t kept = bits;
    uint32_t pending_source = std::numeric_limits<uint32_t>::max();
    uint32_t pending_removals = 0;
    auto flush_source = [&]() {
        if (pending_removals != 0) {
            __atomic_fetch_sub(&source_degree[pending_source], pending_removals,
                               __ATOMIC_RELAXED);
        }
        pending_removals = 0;
    };
    while (bits != 0) {
        const unsigned bit = static_cast<unsigned>(__builtin_ctzll(bits));
        bits &= bits - 1;
        const uint64_t edge = (static_cast<uint64_t>(word_index) << 6) + bit;
        const uint32_t source = work_endpoint(g.src[edge]);
        const uint32_t target = target_index[work_endpoint(g.dst[edge])];
        if (source != pending_source) {
            flush_source();
            pending_source = source;
        }
        const uint32_t source_value =
            __atomic_load_n(&source_degree[source], __ATOMIC_RELAXED) -
            pending_removals;
        const uint32_t target_value = compact_target_degree[target] -
            local_target_decrement[target];
        if (edge_weight(source_value, target_value) >= threshold) continue;
        kept &= ~(uint64_t{1} << bit);
        ++pending_removals;
        if (local_target_decrement[target]++ == 0) {
            local_touched_targets.push_back(target);
        }
        ++stats.removed;
    }
    flush_source();
    transcript[word_index] = kept;
    return stats;
}

struct ParallelReconstructionResult {
    uint64_t edges = 0;
    std::vector<uint32_t> sources;
    std::vector<uint32_t> targets;
};

static ParallelReconstructionResult parallel_sparse_reconstruction(
    const CSRGraph& g,
    std::vector<uint64_t>& transcript,
    std::vector<uint32_t>& source_labels,
    std::vector<uint32_t>& target_labels,
    uint64_t threshold,
    uint32_t maximum_out_degree) {
    ParallelReconstructionResult result;
    const int maximum_workers = omp_get_max_threads();

    // Filter targets by their degree upper bounds.
    std::vector<std::vector<uint32_t>> local_vertices(
        static_cast<size_t>(maximum_workers));
    #pragma omp parallel
    {
        auto& local = local_vertices[static_cast<size_t>(omp_get_thread_num())];
        #pragma omp for schedule(static) nowait
        for (uint32_t target = 0; target < g.n; ++target) {
            if (g.in0[target] != 0 &&
                edge_weight(maximum_out_degree, g.in0[target]) >= threshold) {
                local.push_back(target);
            }
        }
    }
    std::vector<uint32_t> target_ids;
    append_thread_vectors(target_ids, local_vertices);
    const size_t target_count = target_ids.size();
    const int workers = maximum_workers;
    local_vertices.resize(static_cast<size_t>(workers));
    for (auto& local : local_vertices) local.clear();
    #pragma omp parallel for num_threads(workers) schedule(static)
    for (size_t index = 0; index < target_count; ++index) {
        target_labels[target_ids[index]] = static_cast<uint32_t>(index + 1);
    }

    std::vector<std::vector<uint32_t>> target_deltas(static_cast<size_t>(workers));
    std::vector<std::vector<WordId>> local_words(static_cast<size_t>(workers));
    uint64_t selected_edges = 0;
    #pragma omp parallel num_threads(workers) reduction(+:selected_edges)
    {
        const size_t worker = static_cast<size_t>(omp_get_thread_num());
        auto& counts = target_deltas[worker];
        counts.assign(target_count, 0);
        auto& sources = local_vertices[worker];
        auto& words = local_words[worker];
        #pragma omp for schedule(static) nowait
        for (uint64_t word = 0; word < transcript.size(); ++word) {
            uint64_t bits = transcript[word];
            uint64_t kept = bits;
            uint32_t pending_source = std::numeric_limits<uint32_t>::max();
            uint32_t pending_count = 0;
            auto flush_source = [&]() {
                if (pending_count != 0) {
                    const uint32_t old = __atomic_fetch_add(
                        &source_labels[pending_source], pending_count, __ATOMIC_RELAXED);
                    if (old == 0) sources.push_back(pending_source);
                }
                pending_count = 0;
            };
            while (bits != 0) {
                const unsigned bit = static_cast<unsigned>(__builtin_ctzll(bits));
                bits &= bits - 1;
                const uint64_t edge = (word << 6) + bit;
                const uint32_t slot = target_labels[work_endpoint(g.dst[edge])];
                if (slot == 0) {
                    kept &= ~(uint64_t{1} << bit);
                    continue;
                }
                const uint32_t source = work_endpoint(g.src[edge]);
                if (source != pending_source) {
                    flush_source();
                    pending_source = source;
                }
                ++pending_count;
                ++counts[slot - 1];
                ++selected_edges;
            }
            flush_source();
            transcript[word] = kept;
            if (kept != 0) words.push_back(static_cast<WordId>(word));
        }
    }
    for (auto& counts : target_deltas) {
        if (counts.size() != target_count) counts.assign(target_count, 0);
    }
    std::vector<uint32_t> source_ids;
    std::vector<WordId> active_words;
    append_thread_vectors(source_ids, local_vertices);
    append_thread_vectors(active_words, local_words);
    std::vector<uint32_t> compact_target_degree(target_count);
    #pragma omp parallel for num_threads(workers) schedule(static)
    for (size_t target = 0; target < target_count; ++target) {
        uint32_t degree = 0;
        for (auto& counts : target_deltas) {
            degree += counts[target];
            counts[target] = 0;
        }
        compact_target_degree[target] = degree;
        target_labels[target_ids[target]] = static_cast<uint32_t>(target);
    }
    result.edges = selected_edges;

    std::vector<std::vector<uint32_t>> touched_targets(static_cast<size_t>(workers));
    std::vector<WordId> next_words;
    while (result.edges != 0) {
        uint64_t removed = 0;
        #pragma omp parallel num_threads(workers) reduction(+:removed)
        {
            const size_t worker = static_cast<size_t>(omp_get_thread_num());
            auto& words = local_words[worker];
            auto& delta = target_deltas[worker];
            auto& touched = touched_targets[worker];
            words.clear();
            touched.clear();
            #pragma omp for schedule(static) nowait
            for (size_t index = 0; index < active_words.size(); ++index) {
                const WordId word = active_words[index];
                const WordPruneStats stats = prune_sparse_reconstruction_word(
                    g, transcript, word, threshold, source_labels, target_labels,
                    compact_target_degree, delta, touched);
                removed += stats.removed;
                if (transcript[word] != 0) words.push_back(word);
            }
        }
        #pragma omp parallel for num_threads(workers) schedule(static, 1)
        for (int owner = 0; owner < workers; ++owner) {
            auto& delta = target_deltas[static_cast<size_t>(owner)];
            for (uint32_t target : touched_targets[static_cast<size_t>(owner)]) {
                __atomic_fetch_sub(&compact_target_degree[target], delta[target],
                                   __ATOMIC_RELAXED);
                delta[target] = 0;
            }
        }
        append_thread_vectors(next_words, local_words);
        active_words.swap(next_words);
        result.edges -= removed;
        if (removed == 0) break;
    }
    result.sources.reserve(source_ids.size());
    for (uint32_t source : source_ids) {
        if (source_labels[source] != 0) result.sources.push_back(source);
    }
    result.targets.reserve(target_ids.size());
    for (size_t target = 0; target < target_count; ++target) {
        const uint32_t degree = compact_target_degree[target];
        const uint32_t vertex = target_ids[target];
        target_labels[vertex] = degree;
        if (degree != 0) result.targets.push_back(vertex);
    }
    return result;
}

static CSRGraph load_bin32_graph_mmap(const std::string& path) {
    struct Pair32 { uint32_t u, v; };

    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("cannot open binary input file: " + path);
    struct stat file_stat {};
    if (::fstat(fd, &file_stat) != 0 || file_stat.st_size <= 0 ||
        file_stat.st_size % static_cast<off_t>(sizeof(Pair32)) != 0) {
        ::close(fd);
        throw std::runtime_error("invalid raw bin32 input size");
    }
    const size_t byte_count = static_cast<size_t>(file_stat.st_size);
    void* mapped = ::mmap(nullptr, byte_count, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapped == MAP_FAILED) {
        ::close(fd);
        throw std::runtime_error("mmap failed for binary input file");
    }
    (void)::madvise(mapped, byte_count, MADV_SEQUENTIAL);

    const Pair32* pairs = static_cast<const Pair32*>(mapped);
    const uint64_t raw_m = byte_count / sizeof(Pair32);
    uint64_t min_id = std::numeric_limits<uint64_t>::max();
    uint64_t max_id = 0;
    uint64_t m = 0;
    for (uint64_t i = 0; i < raw_m; ++i) {
        const Pair32 pair = pairs[i];
        if (pair.u == pair.v) continue;
        min_id = std::min<uint64_t>(min_id, std::min(pair.u, pair.v));
        max_id = std::max<uint64_t>(max_id, std::max(pair.u, pair.v));
        ++m;
    }
    if (m == 0 || max_id - min_id + 1 > std::numeric_limits<uint32_t>::max() ||
        m > std::numeric_limits<size_t>::max() / sizeof(EdgeId)) {
        ::munmap(mapped, byte_count);
        ::close(fd);
        throw std::runtime_error("bin32 graph exceeds probe representation or is empty");
    }

    CSRGraph g;
    g.n = static_cast<uint32_t>(max_id - min_id + 1);
    g.m = m;
    g.min_input_id = min_id;
    g.max_input_id = max_id;
    g.id_shift = min_id;
    g.out0.assign(g.n, 0);
    g.in0.assign(g.n, 0);
    for (uint64_t i = 0; i < raw_m; ++i) {
        const Pair32 pair = pairs[i];
        if (pair.u == pair.v) continue;
        const uint32_t u = static_cast<uint32_t>(pair.u - min_id);
        const uint32_t v = static_cast<uint32_t>(pair.v - min_id);
        ++g.out0[u];
        ++g.in0[v];
    }

    g.off.assign(static_cast<size_t>(g.n) + 1, 0);
    for (uint32_t u = 0; u < g.n; ++u) g.off[u + 1] = g.off[u] + g.out0[u];
    g.dst.assign(g.m, 0);
    g.src.assign(g.m, 0);
    std::vector<uint64_t> cursor = g.off;
    for (uint64_t i = 0; i < raw_m; ++i) {
        const Pair32 pair = pairs[i];
        if (pair.u == pair.v) continue;
        const uint32_t u = static_cast<uint32_t>(pair.u - min_id);
        const uint32_t v = static_cast<uint32_t>(pair.v - min_id);
        const uint64_t edge = cursor[u]++;
        g.src[edge] = u;
        g.dst[edge] = v;
    }
    ::munmap(mapped, byte_count);
    ::close(fd);
    build_in_index(g);
    return g;
}

static std::vector<uint32_t> collect_transcript_candidates(
    const CSRGraph& g,
    const std::vector<uint64_t>& transcript,
    const std::vector<uint32_t>& active_sources,
    const std::vector<uint32_t>& out_degree,
    const std::vector<uint32_t>& in_degree,
    uint64_t w_star) {
    std::vector<std::vector<uint32_t>> local(
        static_cast<size_t>(omp_get_max_threads()));
    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        auto& candidates = local[static_cast<size_t>(tid)];
        #pragma omp for schedule(dynamic, 256) nowait
        for (size_t index = 0; index < active_sources.size(); ++index) {
            const uint32_t u = active_sources[index];
            const uint32_t source_degree = out_degree[u];
            if (source_degree == 0 || w_star % source_degree != 0) continue;
            const uint64_t required = w_star / source_degree;
            if (required > std::numeric_limits<uint32_t>::max()) continue;
            // Test the target threshold implied by this source degree.
            bool found = false;
            const uint64_t begin = g.off[u], end = g.off[u + 1];
            for (uint64_t word = begin >> 6; word < (end + 63) / 64 && !found; ++word) {
                uint64_t bits = transcript[word];
                if (word == (begin >> 6)) bits &= (~uint64_t{0} << (begin & 63));
                if (word == (end >> 6) && (end & 63)) bits &= ((uint64_t{1} << (end & 63)) - 1);
                while (bits) {
                    const unsigned bit = __builtin_ctzll(bits);
                    bits &= bits - 1;
                    if (in_degree[work_endpoint(g.dst[(word << 6) + bit])] == required) {
                        candidates.push_back(static_cast<uint32_t>(required));
                        found = true;
                        break;
                    }
                }
            }
        }
    }
    std::vector<uint32_t> candidates;
    size_t total = 0;
    for (const auto& values : local) total += values.size();
    candidates.reserve(total);
    for (const auto& values : local) {
        candidates.insert(candidates.end(), values.begin(), values.end());
    }
    std::sort(candidates.begin(), candidates.end());
    candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());
    return candidates;
}

static bool transcript_satisfies_xy(
    const std::vector<uint32_t>& active_sources,
    const std::vector<uint32_t>& active_targets,
    const std::vector<uint32_t>& out_degree,
    const std::vector<uint32_t>& in_degree,
    uint32_t x,
    uint32_t y) {
    for (uint32_t u : active_sources) {
        if (out_degree[u] > 0 && out_degree[u] < x) return false;
    }
    for (uint32_t v : active_targets) {
        if (in_degree[v] > 0 && in_degree[v] < y) return false;
    }
    return true;
}

static State materialize_transcript_state(
    const CSRGraph& g,
    const std::vector<uint64_t>& transcript,
    const std::vector<uint32_t>& active_sources,
    const std::vector<uint32_t>& out_degree,
    const std::vector<uint32_t>& in_degree,
    uint64_t alive_edges) {
    State state;
    state.alive.assign(g.m, 0);
    #pragma omp parallel for schedule(static)
    for (uint64_t word_index = 0; word_index < transcript.size(); ++word_index) {
        uint64_t bits = transcript[word_index];
        while (bits != 0) {
            const unsigned bit = static_cast<unsigned>(__builtin_ctzll(bits));
            bits &= bits - 1;
            state.alive[(word_index << 6) + bit] = 1;
        }
    }
    state.out_deg = out_degree;
    state.in_deg.reset(new std::atomic<uint32_t>[g.n]);
    for (uint32_t v = 0; v < g.n; ++v) {
        state.in_deg[v].store(in_degree[v], std::memory_order_relaxed);
    }
    state.active_sources = active_sources;
    state.alive_edges = alive_edges;
    return state;
}

struct ParametricSearchResult {
    uint64_t w_star = 0;
    std::vector<EdgeId> h_edge_ids;
    // Degrees of the maximum induced-number core.
    std::vector<uint32_t> h_source_ids;
    std::vector<uint32_t> h_target_ids;
    std::vector<uint32_t> h_source_degrees;
    std::vector<uint32_t> h_target_degrees;
};


static void restore_bitmap_word_owned(
    std::vector<uint64_t>& transcript,
    const std::vector<WordId>& old_words,
    const std::vector<EdgeId>& edge_ids) {
    #pragma omp parallel
    {
        #pragma omp for schedule(static)
        for (size_t index = 0; index < old_words.size(); ++index)
            transcript[old_words[index]] = 0;
        // Restore after clearing completes.
        const size_t worker = static_cast<size_t>(omp_get_thread_num());
        const size_t team = static_cast<size_t>(omp_get_num_threads());
        const size_t count = edge_ids.size();
        auto word_boundary = [&](size_t position) {
            while (position > 0 && position < count &&
                   (edge_ids[position - 1] >> 6) == (edge_ids[position] >> 6))
                ++position;
            return position;
        };
        const size_t begin = word_boundary(count * worker / team);
        const size_t end = word_boundary(count * (worker + 1) / team);
        size_t position = begin;
        while (position < end) {
            const WordId word = edge_ids[position] >> 6;
            uint64_t mask = 0;
            do {
                const EdgeId edge = edge_ids[position++];
                mask |= uint64_t{1} << (edge & 63U);
            } while (position < end && (edge_ids[position] >> 6) == word);
            transcript[word] = mask;
        }
    }
}

static ParametricSearchResult exact_w_by_parametric_closure(
    const CSRGraph& g,
    const std::vector<uint64_t>& transcript,
    const std::vector<WordId>& active_words,
    const std::vector<uint32_t>& candidate_sources,
    const std::vector<uint32_t>& candidate_targets,
    std::vector<uint32_t>& source_degree,
    std::vector<uint32_t>& target_degree,
    uint64_t feasible_threshold,
    uint64_t alive_edges) {
    std::vector<uint32_t> sources;
    std::vector<uint32_t> targets;
    for (uint32_t u : candidate_sources) {
        if (source_degree[u] != 0) sources.push_back(u);
    }
    for (uint32_t v : candidate_targets) {
        if (target_degree[v] != 0) targets.push_back(v);
    }
    if (sources.empty() || targets.empty() || alive_edges == 0) return {};

    std::vector<uint32_t> parent_source_degree(sources.size());
    std::vector<uint32_t> parent_target_degree(targets.size());
    for (size_t index = 0; index < sources.size(); ++index) {
        parent_source_degree[index] = source_degree[sources[index]];
        source_degree[sources[index]] = static_cast<uint32_t>(index + 1);
    }
    for (size_t index = 0; index < targets.size(); ++index) {
        parent_target_degree[index] = target_degree[targets[index]];
        target_degree[targets[index]] = static_cast<uint32_t>(index + 1);
    }

#ifdef BGWC_MEMORY_DIAGNOSTICS
    std::fprintf(stderr, "[memory-allocation] compact_candidate edges=%llu bytes=%llu\n",
                 static_cast<unsigned long long>(alive_edges),
                 static_cast<unsigned long long>(5ULL * alive_edges));
#endif
    memory_stage("exact_search.candidate_reserve.begin");
    PackedEdgeIds retained_edges;
    retained_edges.assign_with_bound(static_cast<size_t>(alive_edges), 0,
                                    g.m == 0 ? EdgeId{0} : g.m - 1);
    std::vector<uint32_t> compact_targets(static_cast<size_t>(alive_edges));
    size_t materialized_edges = 0;
    memory_stage("exact_search.candidate_reserve.complete");
    uint64_t upper = feasible_threshold + 1;
    for (WordId word_index : active_words) {
        uint64_t bits = transcript[word_index];
        while (bits != 0) {
            const unsigned bit = static_cast<unsigned>(__builtin_ctzll(bits));
            bits &= bits - 1;
            const uint64_t edge = (static_cast<uint64_t>(word_index) << 6) + bit;
            const uint32_t source = source_degree[work_endpoint(g.src[edge])] - 1;
            const uint32_t target = target_degree[work_endpoint(g.dst[edge])] - 1;
            retained_edges[materialized_edges] = static_cast<EdgeId>(edge);
            compact_targets[materialized_edges++] = target;
            upper = std::max(
                upper,
                edge_weight(
                    parent_source_degree[source],
                    parent_target_degree[target]));
        }
    }
    for (size_t index = 0; index < sources.size(); ++index) {
        source_degree[sources[index]] = parent_source_degree[index];
    }
    if (materialized_edges != alive_edges) {
        throw std::runtime_error("parametric closure edge materialization mismatch");
    }

    struct ProbeResult {
        bool feasible = false;
        std::vector<uint32_t> source_degree;
        std::vector<uint32_t> target_degree;
    };

    ParametricSearchResult result;
    uint64_t low = feasible_threshold;
    uint64_t high = upper;
    std::vector<uint32_t> retained_source_degree = parent_source_degree;
    std::vector<uint32_t> retained_target_degree = parent_target_degree;
    while (low + 1 < high) {
        const uint64_t width = high - low;
        const uint64_t interior = width - 1;
        const size_t probe_count = static_cast<size_t>(
            std::min<uint64_t>(front_options.probe_limit, interior));
        ++front_stats.batches;
        front_stats.queries+=probe_count;
        front_stats.max_batch_width=std::max<uint64_t>(front_stats.max_batch_width,probe_count);
        std::vector<uint64_t> thresholds(probe_count);
        const uint64_t pieces = probe_count + 1;
        const uint64_t whole = width / pieces;
        const uint64_t remainder = width % pieces;
        for (uint64_t index = 1; index <= probe_count; ++index) {
            // Split intervals without overflowing.
            thresholds[index - 1] = low + whole * index + remainder * index / pieces;
        }

        std::vector<ProbeResult> probe_results(probe_count);
        const size_t source_count = retained_source_degree.size();
        const size_t target_count = retained_target_degree.size();
        const size_t retained_count = retained_edges.size();
        std::vector<uint64_t> source_off(source_count + 1, 0);
        for (size_t source = 0; source < source_count; ++source)
            source_off[source + 1] = source_off[source] + retained_source_degree[source];
        if (source_off.back() != retained_count)
            throw std::runtime_error("CSR degree/edge certificate mismatch");

        std::vector<uint32_t> lane_source_degree(
            probe_count * source_count);
        std::vector<uint32_t> lane_target_degree(
            probe_count * target_count);
        for (size_t lane = 0; lane < probe_count; ++lane) {
            std::copy(
                retained_source_degree.begin(),
                retained_source_degree.end(),
                lane_source_degree.begin() + lane * source_count);
            std::copy(
                retained_target_degree.begin(),
                retained_target_degree.end(),
                lane_target_degree.begin() + lane * target_count);
        }

        const uint8_t full_lane_mask = static_cast<uint8_t>(
            (uint32_t{1} << probe_count) - 1);
        std::vector<uint8_t> lane_mask(retained_count, full_lane_mask);
        const size_t worker_count = static_cast<size_t>(omp_get_max_threads());
        if (target_count != 0 && probe_count > std::numeric_limits<size_t>::max() / target_count)
            throw std::length_error("batch target state overflow");
        const size_t target_state_size = probe_count * target_count;
        if (target_state_size != 0 && worker_count > std::numeric_limits<size_t>::max() / target_state_size)
            throw std::length_error("private batch target state overflow");
        const size_t private_state_size = worker_count * target_state_size;
        // Preallocate thread-local counters and touched keys.
        constexpr uint64_t allocation_margin = 16ULL * 1024 * 1024;
        if (private_state_size > (std::numeric_limits<uint64_t>::max() - allocation_margin) / 12ULL)
            throw std::length_error("private batch scratch byte overflow");
        const uint64_t private_scratch_bytes = 12ULL * private_state_size + allocation_margin;
        const auto scratch_budget = packed_storage_detail::compaction_budget();
        if (!scratch_budget.known || private_scratch_bytes > scratch_budget.additional_bytes)
            throw std::runtime_error("RESOURCE_LIMIT: insufficient memory budget for private batch counters and touched keys");
        memory_stage("exact_search.private_batch_scratch.allocate.begin");
        std::unique_ptr<uint32_t[]> thread_target_decrement;
        std::vector<std::vector<uint64_t>> touched_targets;
        try {
            thread_target_decrement.reset(new uint32_t[private_state_size]);
            touched_targets.resize(worker_count);
            for (auto& touched : touched_targets) touched.reserve(target_state_size);
        } catch (const std::bad_alloc&) {
            throw std::runtime_error("RESOURCE_LIMIT: private batch scratch allocation failed");
        }
        #pragma omp parallel for schedule(static, 1)
        for (size_t owner = 0; owner < worker_count; ++owner) {
            uint32_t* const begin = thread_target_decrement.get() + owner * target_state_size;
            std::fill(begin, begin + target_state_size, 0);
        }
        memory_stage("exact_search.private_batch_scratch.allocate.complete");
        auto clear_touched = [&]() {
            for (auto& touched : touched_targets) touched.clear();
        };
        auto commit_private_decrements = [&]() {
            // Combine updates after the source pass.
            #pragma omp parallel for schedule(static, 1)
            for (size_t owner = 0; owner < worker_count; ++owner) {
                uint32_t* const decrement = thread_target_decrement.get() + owner * target_state_size;
                for (uint64_t key64 : touched_targets[owner]) {
                    const size_t key = static_cast<size_t>(key64);
                    __atomic_fetch_sub(&lane_target_degree[key], decrement[key], __ATOMIC_RELAXED);
                    decrement[key] = 0;
                }
            }
        };

        if (front_options.independent_closures) {
            for (size_t lane = 0; lane < probe_count; ++lane) {
                const uint8_t bit=static_cast<uint8_t>(uint32_t{1} << lane);
                while (true) {
                    ++front_stats.closure_rounds;
                    ++front_stats.independent_lane_rounds;
                    clear_touched();
                    uint64_t removed_lanes=0;
                    #pragma omp parallel for reduction(+:removed_lanes) schedule(dynamic, 256)
                    for (size_t source=0; source<source_count; ++source) {
                        const size_t worker=static_cast<size_t>(omp_get_thread_num());
                        auto& local_touched=touched_targets[worker];
                        uint32_t* const decrement=thread_target_decrement.get()+worker*target_state_size;
                        for(uint64_t offset=source_off[source]; offset<source_off[source+1]; ++offset) {
                            const EdgeId position=static_cast<EdgeId>(offset);
                            if((lane_mask[position] & bit)==0) continue;
                            const uint32_t target=work_endpoint(compact_targets[position]);
                            const size_t target_slot=lane*target_count+target;
                            if(edge_weight(lane_source_degree[lane*source_count+source],
                                           lane_target_degree[target_slot]-decrement[target_slot])>thresholds[lane])
                                continue;
                            lane_mask[position]=static_cast<uint8_t>(lane_mask[position] & static_cast<uint8_t>(~bit));
                            --lane_source_degree[lane*source_count+source];
                            if(decrement[target_slot]++==0) local_touched.push_back(target_slot);
                            ++removed_lanes;
                        }
                    }
                    commit_private_decrements();
                    if(removed_lanes==0) break;
                }
            }
            // Check that independently computed closures remain nested.
#ifndef NDEBUG
            for(uint8_t mask:lane_mask)
                if(mask!=0 && (mask & static_cast<uint8_t>(mask+1))!=0)
                    throw std::runtime_error("independent complete closures lost nesting");
#endif
        } else {
        while (true) {
            ++front_stats.closure_rounds;
            clear_touched();
            uint64_t removed_lanes = 0;
            #pragma omp parallel for reduction(+:removed_lanes) schedule(dynamic, 256)
            for (size_t source = 0; source < source_count; ++source) {
                const size_t worker = static_cast<size_t>(omp_get_thread_num());
                auto& local_touched = touched_targets[worker];
                uint32_t* const local_target_decrement =
                    thread_target_decrement.get() + worker * target_state_size;
                for (uint64_t offset = source_off[source];
                     offset < source_off[source + 1];
                     ++offset) {
                    const EdgeId position = static_cast<EdgeId>(offset);
                    const uint32_t target = work_endpoint(compact_targets[position]);
                    const uint8_t old_mask = lane_mask[position];
#ifndef NDEBUG
                    if (old_mask != 0 &&
                        (old_mask & static_cast<uint8_t>(old_mask + 1)) != 0) {
                        throw std::runtime_error(
                            "prefix-round closure lost lane nesting");
                    }
#endif
                    size_t surviving_lanes = static_cast<size_t>(
                        __builtin_popcount(static_cast<unsigned>(old_mask)));
                    while (surviving_lanes != 0) {
                        const size_t lane = surviving_lanes - 1;
                        if (edge_weight(
                                lane_source_degree[
                                    lane * source_count + source],
                                lane_target_degree[lane * target_count + target] -
                                    local_target_decrement[lane * target_count + target]) >
                            thresholds[lane]) {
                            work_prefix_skip(surviving_lanes - 1);
                            break;
                        }
                        --surviving_lanes;
                    }
                    const uint8_t new_mask = static_cast<uint8_t>(
                        (uint32_t{1} << surviving_lanes) - 1);
                    const uint8_t removed = static_cast<uint8_t>(
                        old_mask & static_cast<uint8_t>(~new_mask));
                    lane_mask[position] = new_mask;
                    removed_lanes += static_cast<uint64_t>(
                        __builtin_popcount(static_cast<unsigned>(removed)));
                    uint8_t pending_removed = removed;
                    while (pending_removed != 0) {
                        const unsigned lane = static_cast<unsigned>(
                            __builtin_ctz(static_cast<unsigned>(pending_removed)));
                        // Update the owned source row.
                        --lane_source_degree[lane * source_count + source];
                        const uint64_t key =
                            static_cast<uint64_t>(lane) * target_count + target;
                        if (local_target_decrement[static_cast<size_t>(key)]++ == 0)
                            local_touched.push_back(key);
                        pending_removed = static_cast<uint8_t>(
                            pending_removed & (pending_removed - 1));
                    }
                }
            }
            commit_private_decrements();
            if (removed_lanes == 0) break;
        }

        }

        // Recover lane populations from prefix-mask counts.
        struct alignas(64) PrefixLengthCounts {
            uint64_t counts[8] = {};
        };
        std::vector<PrefixLengthCounts> prefix_counts(worker_count);
        int prefix_workers = 1;
        #pragma omp parallel
        {
            const size_t worker = static_cast<size_t>(omp_get_thread_num());
            const size_t team = static_cast<size_t>(omp_get_num_threads());
            #pragma omp single
            prefix_workers = static_cast<int>(team);
            auto& counts = prefix_counts[worker].counts;
            const size_t begin = packed_storage_detail::compaction_boundary(retained_count, worker, team);
            const size_t end = packed_storage_detail::compaction_boundary(retained_count, worker + 1, team);
            for (size_t position = begin; position < end; ++position) {
                const unsigned length = static_cast<unsigned>(
                    __builtin_popcount(static_cast<unsigned>(lane_mask[position])));
                ++counts[length];
            }
        }
        uint64_t prefix_population[8] = {};
        for (int worker = 0; worker < prefix_workers; ++worker) {
            for (size_t length = 0; length <= probe_count; ++length)
                prefix_population[length] += prefix_counts[static_cast<size_t>(worker)].counts[length];
        }
        std::vector<uint64_t> lane_edges(probe_count, 0);
        uint64_t suffix_population = 0;
        for (size_t length = probe_count; length != 0; --length) {
            suffix_population += prefix_population[length];
            lane_edges[length - 1] = suffix_population;
        }
        if(front_options.trace_closures || front_options.work_count) {
            for(size_t lane=0; lane<probe_count; ++lane) {
                uint64_t fingerprint=0;
                const uint8_t bit=static_cast<uint8_t>(uint32_t{1} << lane);
                for(size_t position=0; position<retained_count; ++position) {
                    if((lane_mask[position] & bit)==0) continue;
                    uint64_t z=uint64_t(retained_edges[position])+0x9e3779b97f4a7c15ULL;
                    z=(z^(z>>30))*0xbf58476d1ce4e5b9ULL;
                    z=(z^(z>>27))*0x94d049bb133111ebULL;
                    fingerprint^=z^(z>>31);
                }
                std::cerr << "front_closure_trace: batch=" << front_stats.batches
                          << " threshold=" << thresholds[lane]
                          << " edges=" << lane_edges[lane]
                          << " membership_hash=" << fingerprint << "\n";
            }
        }
        size_t selected_lane = probe_count;
        for (size_t lane = 0; lane < probe_count; ++lane) {
            ProbeResult& lane_result = probe_results[lane];
            lane_result.feasible = lane_edges[lane] != 0;
            if (lane_result.feasible) selected_lane = lane;
        }
        if (selected_lane != probe_count) {
            ProbeResult& selected = probe_results[selected_lane];
            // Keep the selected closure's degree certificate.
            selected.source_degree.assign(
                lane_source_degree.begin() + selected_lane * source_count,
                lane_source_degree.begin() +
                    (selected_lane + 1) * source_count);
            selected.target_degree.assign(
                lane_target_degree.begin() + selected_lane * target_count,
                lane_target_degree.begin() +
                    (selected_lane + 1) * target_count);
            const size_t selected_edge_count = static_cast<size_t>(lane_edges[selected_lane]);
            const uint8_t selected_bit = static_cast<uint8_t>(uint32_t{1} << selected_lane);
            // Reuse per-thread histograms for stable packing.
            std::vector<size_t> pack_offsets(static_cast<size_t>(prefix_workers) + 1, 0);
            for (int worker = 0; worker < prefix_workers; ++worker) {
                uint64_t count = 0;
                for (size_t length = selected_lane + 1; length <= probe_count; ++length)
                    count += prefix_counts[static_cast<size_t>(worker)].counts[length];
                if (count > std::numeric_limits<size_t>::max() - pack_offsets[static_cast<size_t>(worker)])
                    throw std::length_error("histogram pack offset overflow");
                pack_offsets[static_cast<size_t>(worker) + 1] =
                    pack_offsets[static_cast<size_t>(worker)] + static_cast<size_t>(count);
            }
            if (pack_offsets.back() != selected_edge_count)
                throw std::runtime_error("prefix histogram/pack population mismatch");
            std::vector<uint32_t>().swap(lane_source_degree);
            std::vector<uint32_t>().swap(lane_target_degree);
            thread_target_decrement.reset();
            decltype(touched_targets)().swap(touched_targets);
            std::vector<uint64_t>().swap(source_off);
            decltype(prefix_counts)().swap(prefix_counts);
            std::vector<uint64_t>().swap(lane_edges);
            std::vector<uint32_t>().swap(retained_source_degree);
            std::vector<uint32_t>().swap(retained_target_degree);
            memory_stage("exact_search.pack_scratch_released");
            // Keep inputs until stable scatter completes.
            packed_storage_detail::compact_stable_parallel(
                retained_edges, compact_targets, lane_mask, selected_bit, selected_edge_count,
                g.m == 0 ? EdgeId{0} : g.m - 1, pack_offsets);
#ifdef BGWC_MEMORY_DIAGNOSTICS
            std::fprintf(stderr, "[memory-pack] mode=unified-parallel-scatter input_edges=%llu output_edges=%llu\n",
                static_cast<unsigned long long>(retained_count),
                static_cast<unsigned long long>(selected_edge_count));
#endif
        }

        uint64_t next_low = low;
        uint64_t next_high = high;
        bool saw_infeasible = false;
        size_t retained_index = probe_count;
        for (size_t index = 0; index < probe_count; ++index) {
            if (probe_results[index].feasible) {
                if (saw_infeasible) {
                    throw std::runtime_error(
                        "parametric closure predicate lost monotonicity");
                }
                next_low = thresholds[index];
                retained_index = index;
            } else if (!saw_infeasible) {
                next_high = thresholds[index];
                saw_infeasible = true;
            }
        }
        if (retained_index != probe_count) {
            retained_source_degree = std::move(
                probe_results[retained_index].source_degree);
            retained_target_degree = std::move(
                probe_results[retained_index].target_degree);
            if (retained_edges.empty()) {
                throw std::runtime_error(
                    "parametric closure retained an empty witness");
            }
        }
        low = next_low;
        high = next_high;
    }
    result.w_star = high;
    if (low + 1 != high || retained_edges.empty()) {
        throw std::runtime_error("parametric H witness is empty");
    }
    result.h_edge_ids.resize(retained_edges.size());
    #pragma omp parallel for schedule(static)
    for (size_t position = 0; position < retained_edges.size(); ++position) {
        result.h_edge_ids[position] = retained_edges[position];
    }
    // The final interval identifies the nonstrict core.
    for (size_t index = 0; index < targets.size(); ++index) {
        target_degree[targets[index]] = parent_target_degree[index];
    }
    result.h_source_ids = std::move(sources);
    result.h_target_ids = std::move(targets);
    result.h_source_degrees = std::move(retained_source_degree);
    result.h_target_degrees = std::move(retained_target_degree);
    return result;
}


struct QualityOriginalGraphStats {
    uint32_t max_out;
    uint32_t max_in;
    uint32_t nonzero_sources;
    uint32_t nonzero_targets;
};

// Certificate for a selected subgraph of the complete maximum product core.
struct CertifiedInitialCandidate {
    std::vector<uint8_t> source_membership;
    std::vector<uint8_t> target_membership;
    CoreStats stats;

    CertifiedInitialCandidate(const CertifiedInitialCandidate&) = delete;
    CertifiedInitialCandidate& operator=(const CertifiedInitialCandidate&) = delete;
    CertifiedInitialCandidate(CertifiedInitialCandidate&&) = default;
    CertifiedInitialCandidate& operator=(CertifiedInitialCandidate&&) = default;

    // Use roles and degrees certified by the complete-core producer.
    static CertifiedInitialCandidate from_maximum_h_subset(
        const CSRGraph& g, uint64_t w_star, const XYPairResult& xy,
        std::vector<uint8_t>&& sources, std::vector<uint8_t>&& targets,
        const CoreStats& exact_stats) {
        if (sources.size() != g.n || targets.size() != g.n)
            throw std::runtime_error("certified initial_candidate membership dimensions disagree");
        if (exact_stats.edges > g.m || exact_stats.S > g.n || exact_stats.T > g.n ||
            exact_stats.S > exact_stats.edges || exact_stats.T > exact_stats.edges ||
            ((exact_stats.edges == 0) != (exact_stats.S == 0 || exact_stats.T == 0)))
            throw std::runtime_error("certified initial_candidate statistics violate support bounds");
        if (g.m == 0) {
            if (w_star != 0 || exact_stats.edges != 0 || exact_stats.S != 0 || exact_stats.T != 0)
                throw std::runtime_error("empty graph certified initial_candidate is not empty");
        } else if (!xy.found || xy.x == 0 || xy.y == 0 ||
                   edge_weight(xy.x, xy.y) != w_star) {
            throw std::runtime_error("certified initial_candidate lacks an exact H threshold pair");
        }
        return CertifiedInitialCandidate(g, w_star, std::move(sources),
                                    std::move(targets), exact_stats);
    }

    void require_original_graph(const CSRGraph& g, uint64_t w_star) const {
        if (original_graph != &g || exact_w_star != w_star ||
            source_membership.size() != g.n || target_membership.size() != g.n)
            throw std::runtime_error("certified initial_candidate graph or w_star provenance mismatch");
    }

private:
    const CSRGraph* original_graph;
    uint64_t exact_w_star;

    CertifiedInitialCandidate(const CSRGraph& g, uint64_t w_star,
                         std::vector<uint8_t>&& sources,
                         std::vector<uint8_t>&& targets, const CoreStats& exact_stats)
        : source_membership(std::move(sources)), target_membership(std::move(targets)),
          stats(exact_stats), original_graph(&g), exact_w_star(w_star) {}
};

struct QualityCoreResult {
    std::vector<uint8_t> source_membership;
    std::vector<uint8_t> target_membership;
    CoreStats stats;
    uint64_t mu_upper_bound = 0;
    bool certified = false;
    uint64_t probe_calls = 0;
    uint64_t probe_rounds = 0;
    uint64_t probe_edge_scans = 0;
    uint64_t induced_edge_scans = 0;
    uint64_t branches = 0;
    uint64_t initial_candidate_evaluations = 0;
    bool natural_initial_certified = false;
    bool early_stop_disabled = false;
    bool state_reuse_disabled = false, trace_bound_disabled = false;
    bool safe_candidate_disabled = false;
#ifdef BGWC_PROFILE
    double profile_initial_candidate_selection_time=0, profile_phase2_search_time=0;
#endif
    uint64_t fallback_empty_queries = 0, fallback_trace_bound_candidates = 0;
    uint64_t fallback_trace_inherited_branches = 0, fallback_trace_pruned_branches = 0;
    uint64_t fallback_candidate_threshold = 0;
    uint64_t fallback_candidate_vertices = 0, fallback_candidate_edges = 0;
    uint64_t fallback_initializations = 0, fallback_inherited_advances = 0;
    uint64_t fallback_role_scans = 0, fallback_deleted_edges = 0;
    uint64_t fallback_candidate_edge_scans = 0, fallback_empty_cap_updates = 0;
    bool fallback_simple_verified = false;
    double fallback_candidate_time = 0.0;
};

namespace quality_core_detail {
using Wide = unsigned __int128;

static bool density_covers_product(const CoreStats& stats, uint64_t product) {
    if (stats.edges == 0 || stats.S == 0 || stats.T == 0) return product == 0;
    return Wide(stats.edges) * stats.edges >= Wide(product) * stats.S * stats.T;
}

static bool density_better(const CoreStats& candidate, const CoreStats& incumbent) {
    if (candidate.edges == 0) return false;
    if (incumbent.edges == 0) return true;
    return Wide(candidate.edges) * candidate.edges * incumbent.S * incumbent.T >
           Wide(incumbent.edges) * incumbent.edges * candidate.S * candidate.T;
}

static uint64_t first_product_improvement(const CoreStats& stats, uint32_t fixed) {
    const Wide denominator = Wide(fixed) * stats.S * stats.T;
    if (denominator == 0) throw std::runtime_error("quality search has empty incumbent");
    return static_cast<uint64_t>(Wide(stats.edges) * stats.edges / denominator) + 1;
}

static uint32_t floor_sqrt_u64(uint64_t value) {
    uint64_t low = 0, high = uint64_t{1} << 32;
    while (low + 1 < high) {
        const uint64_t middle = low + (high - low) / 2;
        if (Wide(middle) * middle <= value) low = middle;
        else high = middle;
    }
    return static_cast<uint32_t>(low);
}

static void finish_density(CoreStats& stats) {
    stats.density = (stats.S != 0 && stats.T != 0)
        ? static_cast<double>(stats.edges) /
          std::sqrt(static_cast<double>(stats.S) * static_cast<double>(stats.T))
        : 0.0;
}

// Build the search candidate graph from the original graph.
struct CandidateDomain {
    CSRGraph graph;
    std::vector<uint32_t> original;
    bool simple_verified = true;
    const CSRGraph* borrowed_original = nullptr;
    uint64_t guard_row_checks = 0;
    const CSRGraph& view() const { return borrowed_original ? *borrowed_original : graph; }
    uint32_t original_vertex(uint32_t v) const { return borrowed_original ? v : original[v]; }
};

// Reuse the original adjacency and validate simple-graph bounds.
static CandidateDomain whole_original_domain(const CSRGraph& g) {
    CandidateDomain domain;
    domain.borrowed_original = &g;
    int unordered_row = 0;
    uint64_t row_checks=0;
    #pragma omp parallel for reduction(max:unordered_row) reduction(+:row_checks) schedule(dynamic, 256)
    for (uint64_t u=0; u<g.n; ++u) {
        for (uint64_t e=g.off[u]+1; e<g.off[u+1]; ++e) {
            ++row_checks;
            if (work_endpoint(g.dst[e]) <= work_endpoint(g.dst[e-1])) unordered_row=1;
        }
    }
    domain.simple_verified = unordered_row == 0;
    domain.guard_row_checks=row_checks;
    return domain;
}

static CandidateDomain build_candidate_domain(const CSRGraph& g, uint64_t q) {
    CandidateDomain domain;
    std::vector<uint32_t> out(g.n, 0), in(g.n, 0), map(g.n, 0);
    #pragma omp parallel for schedule(dynamic, 256)
    for (uint64_t u = 0; u < g.n; ++u) {
        uint32_t degree = 0;
        for (uint64_t e = g.off[u]; e < g.off[u + 1]; ++e) {
            const uint32_t v = work_endpoint(g.dst[e]);
            if (edge_weight(g.out0[u],g.in0[v]) < q) continue;
            ++degree;
            __atomic_fetch_add(&in[v], uint32_t{1}, __ATOMIC_RELAXED);
        }
        out[u] = degree;
    }
    CSRGraph& c = domain.graph;
    for (uint64_t v = 0; v < g.n; ++v) if (out[v] || in[v]) {
        map[v] = static_cast<uint32_t>(domain.original.size());
        domain.original.push_back(static_cast<uint32_t>(v));
    }
    c.n = static_cast<uint32_t>(domain.original.size());
    c.off.resize(size_t(c.n) + 1); c.in_off.resize(size_t(c.n) + 1);
    c.out0.resize(c.n); c.in0.resize(c.n);
    for (uint64_t v = 0; v < c.n; ++v) {
        c.out0[v] = out[domain.original[v]];
        c.in0[v] = in[domain.original[v]];
        c.off[v + 1] = c.off[v] + c.out0[v];
        c.in_off[v + 1] = c.in_off[v] + c.in0[v];
    }
    c.m = c.off[c.n];
    c.src.resize(c.m); c.dst.resize(c.m); c.in_edges.resize(c.m);
    std::vector<uint64_t> cursor = c.in_off;
    int unordered_row = 0;
    #pragma omp parallel for reduction(max:unordered_row) schedule(dynamic, 256)
    for (uint64_t u = 0; u < c.n; ++u) {
        const uint32_t original_u = domain.original[u];
        uint64_t write = c.off[u];
        uint32_t previous = 0;
        bool have_previous = false;
        for (uint64_t e = g.off[original_u]; e < g.off[size_t(original_u) + 1]; ++e) {
            const uint32_t original_v = work_endpoint(g.dst[e]);
            if (edge_weight(g.out0[original_u],g.in0[original_v]) < q) continue;
            const uint32_t v = map[original_v];
            if (have_previous && v <= previous) unordered_row = 1;
            previous = v; have_previous = true;
            c.src[write] = static_cast<uint32_t>(u); c.dst[write] = v;
            const uint64_t slot = __atomic_fetch_add(&cursor[v], uint64_t{1}, __ATOMIC_RELAXED);
            c.in_edges[slot] = static_cast<EdgeId>(write++);
        }
    }
    domain.simple_verified = unordered_row == 0;
    return domain;
}

struct ProbeScratch {
    std::unique_ptr<uint8_t[]> alive;
    std::unique_ptr<uint32_t[]> out_degree, in_degree, source_queue, target_queue;
    std::unique_ptr<uint64_t[]> source_deleted, target_deleted;
    uint64_t remaining = 0, epoch = 0;
    uint32_t previous_x = 0, previous_y = 0;
    uint32_t maximum_removed_out = 0, maximum_removed_in = 0;

    explicit ProbeScratch(const CSRGraph& g)
        : alive(new uint8_t[size_t(g.m)]), out_degree(new uint32_t[g.n]),
          in_degree(new uint32_t[g.n]), source_queue(new uint32_t[g.n]),
          target_queue(new uint32_t[g.n]), source_deleted(new uint64_t[g.n]),
          target_deleted(new uint64_t[g.n]) {}

    void reset(const CSRGraph& g, QualityCoreResult& counters) {
        ++counters.fallback_initializations;
        remaining = g.m; epoch = 0; previous_x = previous_y = 0;
        maximum_removed_out = maximum_removed_in = 0;
        #pragma omp parallel if(g.m >= 32768)
        {
            #pragma omp for schedule(static) nowait
            for (uint64_t v = 0; v < g.n; ++v) {
                out_degree[v] = g.out0[v]; in_degree[v] = g.in0[v];
                source_deleted[v] = g.out0[v] ? UINT64_MAX : 0;
                target_deleted[v] = g.in0[v] ? UINT64_MAX : 0;
            }
            #pragma omp for schedule(static)
            for (uint64_t e = 0; e < g.m; ++e) alive[e] = 1;
        }
    }

    CoreStats probe(const CSRGraph& g, uint32_t x, uint32_t y, QualityCoreResult& counters) {
        if (!x || !y || x < previous_x || y < previous_y)
            throw std::runtime_error("nonmonotone incremental core query");
        if (epoch) ++counters.fallback_inherited_advances;
        previous_x = x; previous_y = y; ++epoch; ++counters.probe_calls;
        counters.fallback_role_scans += uint64_t(4) * g.n;
        uint64_t source_tail = 0, target_tail = 0;
        uint64_t source_begin = 0, source_end = 0, target_begin = 0, target_end = 0;
        uint64_t removed_edges = 0, scanned_edges = 0, rounds = 0;
        uint32_t surviving_sources = 0, surviving_targets = 0;
        uint32_t removed_out = 0, removed_in = 0;
        bool done = false;
        #pragma omp parallel if(g.m >= 32768) reduction(+:removed_edges,scanned_edges) reduction(max:removed_out,removed_in)
        {
            #pragma omp for schedule(static)
            for (uint64_t v = 0; v < g.n; ++v) {
                if (out_degree[v] && out_degree[v] < x) {
                    const uint64_t slot = __atomic_fetch_add(&source_tail, uint64_t{1}, __ATOMIC_RELAXED);
                    source_queue[slot] = static_cast<uint32_t>(v);
                }
                if (in_degree[v] && in_degree[v] < y) {
                    const uint64_t slot = __atomic_fetch_add(&target_tail, uint64_t{1}, __ATOMIC_RELAXED);
                    target_queue[slot] = static_cast<uint32_t>(v);
                }
            }
            #pragma omp single
            {
                source_end = source_tail; target_end = target_tail;
                done = (source_end == 0 && target_end == 0);
            }
            while (!done) {
                auto remove_edge = [&](EdgeId e) {
                    ++scanned_edges;
                    if (__atomic_exchange_n(&alive[e], uint8_t{0}, __ATOMIC_RELAXED) == 0) return;
                    ++removed_edges;
                    const uint32_t u = work_endpoint(g.src[e]), v = work_endpoint(g.dst[e]);
                    const uint32_t old_out = __atomic_fetch_sub(&out_degree[u], uint32_t{1}, __ATOMIC_RELAXED);
                    const uint32_t old_in = __atomic_fetch_sub(&in_degree[v], uint32_t{1}, __ATOMIC_RELAXED);
                    if (old_out == 1) source_deleted[u] = epoch;
                    if (old_in == 1) target_deleted[v] = epoch;
                    if (old_out == x) {
                        const uint64_t slot = __atomic_fetch_add(&source_tail, uint64_t{1}, __ATOMIC_RELAXED);
                        source_queue[slot] = u;
                    }
                    if (old_in == y) {
                        const uint64_t slot = __atomic_fetch_add(&target_tail, uint64_t{1}, __ATOMIC_RELAXED);
                        target_queue[slot] = v;
                    }
                };
                #pragma omp for schedule(dynamic, 1)
                for (uint64_t i = source_begin; i < source_end; ++i) {
                    const uint32_t u = source_queue[i];
                    const uint32_t degree = __atomic_load_n(&out_degree[u], __ATOMIC_RELAXED);
                    removed_out = std::max(removed_out, degree);
                    if (!degree) continue;
                    for (uint64_t e = g.off[u]; e < g.off[size_t(u) + 1]; ++e)
                        remove_edge(static_cast<EdgeId>(e));
                }
                // Track deletion degrees across source and target waves.
                #pragma omp for schedule(dynamic, 1)
                for (uint64_t i = target_begin; i < target_end; ++i) {
                    const uint32_t v = target_queue[i];
                    const uint32_t degree = __atomic_load_n(&in_degree[v], __ATOMIC_RELAXED);
                    removed_in = std::max(removed_in, degree);
                    if (!degree) continue;
                    for (uint64_t slot = g.in_off[v]; slot < g.in_off[size_t(v) + 1]; ++slot)
                        remove_edge(work_reverse_index(g.in_edges[slot]));
                }
                #pragma omp single
                {
                    ++rounds;
                    source_begin = source_end; target_begin = target_end;
                    source_end = source_tail; target_end = target_tail;
                    done = source_begin == source_end && target_begin == target_end;
                }
            }
            #pragma omp for reduction(+:surviving_sources,surviving_targets) schedule(static)
            for (uint64_t v = 0; v < g.n; ++v) {
                surviving_sources += out_degree[v] != 0;
                surviving_targets += in_degree[v] != 0;
            }
        }
        maximum_removed_out = std::max(maximum_removed_out, removed_out);
        maximum_removed_in = std::max(maximum_removed_in, removed_in);
        counters.probe_rounds += rounds; counters.probe_edge_scans += scanned_edges;
        counters.fallback_deleted_edges += removed_edges;
        if (removed_edges > remaining) throw std::runtime_error("incremental edge count underflow");
        remaining -= removed_edges;
        CoreStats answer;
        answer.edges = remaining; answer.S = surviving_sources; answer.T = surviving_targets;
        finish_density(answer); return answer;
    }

    void capture(const CSRGraph& original, const CandidateDomain& domain,
                 QualityCoreResult& output, uint64_t best_epoch) const {
        #pragma omp parallel for schedule(static)
        for (uint64_t v = 0; v < original.n; ++v) {
            output.source_membership[v] = 0; output.target_membership[v] = 0;
        }
        #pragma omp parallel for schedule(static)
        for (uint64_t v = 0; v < domain.view().n; ++v) {
            const uint32_t u = domain.original_vertex(static_cast<uint32_t>(v));
            output.source_membership[u] = source_deleted[v] > best_epoch;
            output.target_membership[u] = target_deleted[v] > best_epoch;
        }
    }
};
} // namespace quality_core_detail

static QualityCoreResult quality_certify_dds(
    const CSRGraph& g,
    uint64_t w_star,
    CertifiedInitialCandidate initial_candidate,
    const QualityOriginalGraphStats& original_stats, bool disable_early_stop,
    bool disable_state_reuse, bool disable_trace_bound, bool disable_safe_candidate) {
    using namespace quality_core_detail;
#ifdef BGWC_PROFILE
    Timer initial_candidate_selection_timer;
#endif
    initial_candidate.require_original_graph(g, w_star);
    if (g.m > std::numeric_limits<size_t>::max() / sizeof(EdgeId))
        throw std::runtime_error("quality operator edge allocation exceeds address space");
    QualityCoreResult result;
    result.early_stop_disabled = disable_early_stop;
    result.state_reuse_disabled = disable_state_reuse;
    result.trace_bound_disabled = disable_trace_bound;
    result.safe_candidate_disabled = disable_safe_candidate;
    result.source_membership = std::move(initial_candidate.source_membership);
    result.target_membership = std::move(initial_candidate.target_membership);
    const uint32_t maximum_out = original_stats.max_out;
    const uint32_t maximum_in = original_stats.max_in;
    result.stats.edges = g.m;
    result.stats.S = original_stats.nonzero_sources;
    result.stats.T = original_stats.nonzero_targets;
    finish_density(result.stats);
    ++result.initial_candidate_evaluations;
    if (g.m == 0) {
        result.certified = true;
#ifdef BGWC_PROFILE
        result.profile_initial_candidate_selection_time=initial_candidate_selection_timer.elapsed();
#endif
        return result;
    }
    if (w_star < std::max(maximum_out, maximum_in))
        throw std::runtime_error("quality w_star bound is below a feasible star product");

    CoreStats initial_candidate_stats = initial_candidate.stats;
    finish_density(initial_candidate_stats);
    ++result.initial_candidate_evaluations;
    bool whole_membership_pending = true;
    if (density_better(initial_candidate_stats, result.stats)) {
        result.stats = initial_candidate_stats;
        whole_membership_pending = false;
    }
    auto materialize_whole_membership = [&]() {
        if (!whole_membership_pending) return;
        #pragma omp parallel for schedule(static)
        for (uint64_t vertex = 0; vertex < g.n; ++vertex) {
            result.source_membership[vertex] = g.out0[vertex] != 0;
            result.target_membership[vertex] = g.in0[vertex] != 0;
        }
        whole_membership_pending = false;
    };
    result.natural_initial_certified = density_covers_product(result.stats, w_star);
    if (result.natural_initial_certified && !disable_early_stop) {
        result.mu_upper_bound = w_star;
        result.certified = true;
        materialize_whole_membership();
#ifdef BGWC_PROFILE
        result.profile_initial_candidate_selection_time=initial_candidate_selection_timer.elapsed();
#endif
        return result;
    }
#ifdef BGWC_PROFILE
    result.profile_initial_candidate_selection_time=initial_candidate_selection_timer.elapsed();
#endif
    // Forced search disables global density early returns.
    const uint64_t initial_floor = static_cast<uint64_t>(
        Wide(result.stats.edges) * result.stats.edges / (Wide(result.stats.S) * result.stats.T));
    result.fallback_candidate_threshold = initial_floor + 1;
    Timer candidate_timer;
    CandidateDomain domain = disable_safe_candidate ? whole_original_domain(g)
        : build_candidate_domain(g, result.fallback_candidate_threshold);
    result.fallback_candidate_time = candidate_timer.elapsed();
#ifdef BGWC_PROFILE
    Timer phase2_search_timer;
#endif
    const CSRGraph& q = domain.view();
    result.fallback_candidate_vertices = q.n; result.fallback_candidate_edges = q.m;
    result.fallback_candidate_edge_scans = disable_safe_candidate ? domain.guard_row_checks : g.m;
    if (!disable_safe_candidate)
        for (uint32_t v : domain.original) result.fallback_candidate_edge_scans += g.out0[v];
    result.fallback_simple_verified = domain.simple_verified;
    ProbeScratch scratch(q);
    const uint32_t root_w_star = floor_sqrt_u64(w_star);
    uint64_t completed_upper = initial_floor;
    std::vector<uint32_t> ranked_out = q.out0, ranked_in = q.in0;
    std::sort(ranked_out.begin(), ranked_out.end(), std::greater<uint32_t>());
    std::sort(ranked_in.begin(), ranked_in.end(), std::greater<uint32_t>());
    const uint32_t q_max_out = ranked_out.empty() ? 0 : ranked_out.front();
    const uint32_t q_max_in = ranked_in.empty() ? 0 : ranked_in.front();
    for (unsigned orientation = 0; orientation != 2; ++orientation) {
        const auto& ranked_fixed = orientation == 0 ? ranked_out : ranked_in;
        const auto& ranked_opposite = orientation == 0 ? ranked_in : ranked_out;
        const uint32_t maximum_fixed = orientation == 0 ? q_max_out : q_max_in;
        const uint32_t maximum_opposite = orientation == 0 ? q_max_in : q_max_out;
        const uint64_t fixed_limit = std::min(root_w_star, maximum_fixed);
        uint64_t inherited_opposite_upper = maximum_opposite;
        size_t qualifying_fixed = ranked_fixed.size();
        for (uint64_t fixed64 = 1; fixed64 <= fixed_limit; ++fixed64) {
            ++result.branches;
            const uint32_t fixed = static_cast<uint32_t>(fixed64);
            uint64_t upper = std::min(inherited_opposite_upper, w_star / fixed);
            uint64_t upper_without_trace = std::min<uint64_t>(maximum_opposite, w_star / fixed);
            if (domain.simple_verified) {
                while (qualifying_fixed && ranked_fixed[qualifying_fixed - 1] < fixed)
                    --qualifying_fixed;
                const uint64_t rank_bound = fixed64 <= ranked_opposite.size()
                    ? ranked_opposite[fixed64 - 1] : 0;
                const uint64_t structural_upper = std::min<uint64_t>(qualifying_fixed, rank_bound);
                upper = std::min(upper, structural_upper);
                upper_without_trace = std::min(upper_without_trace, structural_upper);
            }
            if (upper < upper_without_trace) ++result.fallback_trace_inherited_branches;
            if (density_covers_product(result.stats, fixed64 * upper)) {
                if (!density_covers_product(result.stats, fixed64 * upper_without_trace))
                    ++result.fallback_trace_pruned_branches;
                completed_upper = std::max(completed_upper, fixed64 * upper);
                continue;
            }
            if (!disable_state_reuse) scratch.reset(q, result);
            uint64_t best_epoch = 0;
            while (true) {
                const uint64_t opposite = first_product_improvement(result.stats, fixed);
                if (opposite > upper) {
                    completed_upper = std::max(completed_upper, fixed64 * upper);
                    break;
                }
                const uint32_t x = orientation == 0 ? fixed : static_cast<uint32_t>(opposite);
                const uint32_t y = orientation == 0 ? static_cast<uint32_t>(opposite) : fixed;
                // Reset peeling state for independent queries.
                if (disable_state_reuse) scratch.reset(q, result);
                const CoreStats found = scratch.probe(q, x, y, result);
                if (!found.edges) {
                    ++result.fallback_empty_queries;
                    // Use deletion traces to bound later parameter branches.
                    const uint64_t trace_upper = orientation == 0
                        ? scratch.maximum_removed_in : scratch.maximum_removed_out;
                    if (trace_upper >= opposite)
                        throw std::runtime_error("invalid empty-query degree trace");
                    if (trace_upper < maximum_opposite)
                        ++result.fallback_trace_bound_candidates;
                    // Keep the current branch bound without cross-branch reuse.
                    if (!disable_trace_bound && trace_upper < inherited_opposite_upper) {
                        inherited_opposite_upper = trace_upper;
                        ++result.fallback_empty_cap_updates;
                    }
                    completed_upper = std::max(completed_upper, fixed64 * trace_upper);
                    break;
                }
                if (!density_covers_product(found, fixed64 * opposite))
                    throw std::runtime_error("incremental core density below threshold product");
                if (density_better(found, result.stats)) {
                    result.stats = found; best_epoch = scratch.epoch;
                    // Save improved memberships before resetting query state.
                    if (disable_state_reuse) {
                        scratch.capture(g, domain, result, best_epoch);
                        whole_membership_pending = false;
                    }
                } else throw std::runtime_error("incremental density failed to increase");
                if (!disable_early_stop && density_covers_product(result.stats, w_star)) {
                    if (!disable_state_reuse) scratch.capture(g, domain, result, best_epoch);
                    result.mu_upper_bound = w_star; result.certified = true;
#ifdef BGWC_PROFILE
                    result.profile_phase2_search_time=phase2_search_timer.elapsed();
#endif
                    return result;
                }
            }
            if (best_epoch && !disable_state_reuse) {
                scratch.capture(g, domain, result, best_epoch);
                whole_membership_pending = false;
            }
        }
    }
    result.mu_upper_bound = std::min(w_star, completed_upper);
    if (!density_covers_product(result.stats, result.mu_upper_bound))
        throw std::runtime_error("quality search ended without a complete density certificate");
    result.certified = true;
    materialize_whole_membership();
#ifdef BGWC_PROFILE
    result.profile_phase2_search_time=phase2_search_timer.elapsed();
#endif
    return result;
}


static void save_quality_output(
    const CSRGraph& g, const QualityCoreResult& result, const std::string& prefix) {
    std::ofstream fs(prefix + ".S.txt");
    std::ofstream ft(prefix + ".T.txt");
    std::ofstream fe(prefix + ".edges.txt");
    if (!fs || !ft || !fe) {
        throw std::runtime_error("failed to create quality output files: " + prefix);
    }
    for (uint32_t v = 0; v < g.n; ++v) {
        if (result.source_membership[v]) fs << (uint64_t(v) + g.id_shift) << '\n';
        if (result.target_membership[v]) ft << (uint64_t(v) + g.id_shift) << '\n';
    }
    for (uint64_t e = 0; e < g.m; ++e) {
        if (result.source_membership[work_endpoint(g.src[e])] && result.target_membership[work_endpoint(g.dst[e])]) {
            fe << (uint64_t(work_endpoint(g.src[e])) + g.id_shift) << ' '
               << (uint64_t(work_endpoint(g.dst[e])) + g.id_shift) << '\n';
        }
    }
    if (!fs || !ft || !fe) throw std::runtime_error("failed while saving quality output");
}


static constexpr size_t kCachedDegreeBuckets = 256;
struct alignas(64) DegreeCoverWorkspace {
    uint64_t out_bins[kCachedDegreeBuckets];
    uint64_t in_bins[kCachedDegreeBuckets];
    uint64_t out_total, in_total, out_offset, in_offset;
    uint64_t best_cost;
    uint32_t best_source_cut, best_target_cut;
    size_t target_count, target_offset;
    // Initialize thread-local buckets in parallel.
    DegreeCoverWorkspace() noexcept {}
};

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " GRAPH THREADS [--verify-output] [--save PREFIX]\n"
                  << "\nRegular options:\n"
                  << "  --verify-output  Verify selected memberships and induced-edge counts.\n"
                  << "  --save PREFIX    Write PREFIX.S.txt, PREFIX.T.txt, and PREFIX.edges.txt.\n"
                  << "\nAdvanced ablation/debug options:\n"
                  << "  --no-early-stop          Disable density-certificate early returns.\n"
                  << "  --no-candidate-reduction Use all edges in maximum induced-number computation.\n"
                  << "  --no-cover-cut           Scan all source rows during initial edge filtering.\n"
                  << "  --independent-closures   Evaluate threshold closures independently.\n"
                  << "  --no-safe-candidate      Use the whole graph for subsequent core search.\n"
                  << "  --no-state-reuse         Reset peeling state for every core query.\n"
                  << "  --no-trace-bound         Disable inherited deletion-degree bounds.\n"
                  << "  --trace-closures         Write executed closure-query traces to stderr.\n"
                  << "  --work-count             Enable internal counters (requires BGWC_WORK_COUNT).\n"
                  << "\nThe batch query budget is fixed at 7.\n";
        return 64;
    }
    std::string save_prefix;
    bool disable_early_stop = false;
    bool disable_state_reuse = false, disable_trace_bound = false;
    bool disable_safe_candidate = false;
    bool verify_output_membership = false;
    if (const char* value = std::getenv("BGWC_DISABLE_EARLY"))
        disable_early_stop = std::string(value) == "1";
    for (int i = 3; i < argc; ++i) {
        const std::string option = argv[i];
        if (option == "--no-early-stop") disable_early_stop = true;
        else if (option == "--no-cover-cut") front_options.no_cover_cut=true;
        else if (option == "--no-safe-candidate") disable_safe_candidate=true;
        else if (option == "--no-state-reuse") disable_state_reuse = true;
        else if (option == "--no-trace-bound") disable_trace_bound = true;
        else if (option == "--verify-output") verify_output_membership = true;
        else if (option == "--save" && i + 1 < argc) save_prefix = argv[++i];
        else if (option == "--no-candidate-reduction") front_options.no_candidate_reduction=true;
        else if (option == "--independent-closures") front_options.independent_closures=true;
        else if (option == "--work-count") front_options.work_count=true;
        else if (option == "--trace-closures") front_options.trace_closures=true;
        else { std::cerr << "Unknown or incomplete option: " << option << "\n"; return 64; }
    }
#ifndef BGWC_WORK_COUNT
    if(front_options.work_count) {
        std::cerr << "--work-count requires a separate -DBGWC_WORK_COUNT build\n"; return 64;
    }
#endif
    omp_set_dynamic(0);
    const int threads = std::max(1, std::atoi(argv[2]));
    omp_set_num_threads(threads);
    Timer read_timer;
    const std::string graph_path = argv[1];
    memory_stage("load_graph.begin");
    CSRGraph g = (has_ext(graph_path, ".bin") || has_ext(graph_path, ".bin32"))
        ? load_bin32_graph_mmap(graph_path)
        : load_txt_graph_dense_range(graph_path);
    const double read_time = read_timer.elapsed();
#ifdef BGWC_MEMORY_DIAGNOSTICS
    std::fprintf(stderr, "[memory-shape] n=%llu m=%llu threads=%d\n",
                 static_cast<unsigned long long>(g.n),
                 static_cast<unsigned long long>(g.m), threads);
#endif
    memory_stage("load_graph.complete");

    work_slots.resize(static_cast<size_t>(omp_get_max_threads()));
    work_active=front_options.work_count;
    Timer full_algorithm_timer;
    uint32_t maximum_out_degree = 0;
    uint32_t maximum_in_degree = 0;
    int observed_team_threads = 0;
    uint32_t original_nonzero_sources = 0, original_nonzero_targets = 0;
    // Count original source and target roles.
    #pragma omp parallel reduction(max:maximum_out_degree,maximum_in_degree,observed_team_threads) reduction(+:original_nonzero_sources,original_nonzero_targets)
    {
        observed_team_threads = omp_get_num_threads();
        #pragma omp for schedule(static) nowait
        for (uint32_t vertex = 0; vertex < g.n; ++vertex) {
            const uint32_t out_degree = g.out0[vertex];
            const uint32_t in_degree = g.in0[vertex];
            maximum_out_degree = std::max(maximum_out_degree, out_degree);
            maximum_in_degree = std::max(maximum_in_degree, in_degree);
            original_nonzero_sources += out_degree != 0;
            original_nonzero_targets += in_degree != 0;
        }
    }
    const QualityOriginalGraphStats quality_original_stats{
        maximum_out_degree, maximum_in_degree,
        original_nonzero_sources, original_nonzero_targets};
    uint64_t lower = std::max(maximum_out_degree, maximum_in_degree);
    // Initialize the density lower bound from a star.

    const uint64_t transcript_words = (g.m + 63) / 64;
    std::vector<uint64_t> transcript(transcript_words, 0);
    std::vector<uint64_t> initial_nonstrict_transcript;
    std::vector<uint32_t> source_labels(g.n, 0);
    std::vector<uint32_t> target_labels(g.n, 0);
    uint64_t candidate_edges=0;
#ifdef BGWC_PROFILE
    double profile_batch_threshold_time=0.0;
#endif
    if(front_options.no_candidate_reduction) {
        std::fill(transcript.begin(),transcript.end(),~uint64_t{0});
        if(g.m % 64) transcript.back()=(uint64_t{1} << (g.m % 64))-1;
        initial_nonstrict_transcript=transcript;
        source_labels=g.out0;
        target_labels=g.in0;
        candidate_edges=g.m;
        front_stats.initial_nonstrict_edges=g.m;
    } else if (front_options.no_cover_cut) {
        // Scan every source row with the same degree filters.
        std::vector<std::vector<EdgeId>> equality_edges(omp_get_max_threads());
        #pragma omp parallel for reduction(+:candidate_edges) schedule(dynamic, 256)
        for (uint32_t u=0; u<g.n; ++u) {
            uint32_t strict_degree=0;
            uint64_t current_word=std::numeric_limits<uint64_t>::max(), mask=0;
            for (uint64_t e=g.off[u]; e<g.off[u+1]; ++e) {
                const uint32_t v=work_endpoint(g.dst[e]);
                const uint64_t product=edge_weight(g.out0[u],g.in0[v]);
                if (product == lower) equality_edges[omp_get_thread_num()].push_back(static_cast<EdgeId>(e));
                if (product <= lower) continue;
                const uint64_t word=e >> 6;
                if (word != current_word) {
                    if (current_word != std::numeric_limits<uint64_t>::max())
                        merge_source_row_word(transcript,current_word,mask,g.off[u],g.off[u+1]);
                    current_word=word; mask=0;
                }
                mask |= uint64_t{1} << (e & 63U);
                ++strict_degree; ++candidate_edges;
                #pragma omp atomic update
                target_labels[v] += 1;
            }
            if (current_word != std::numeric_limits<uint64_t>::max())
                merge_source_row_word(transcript,current_word,mask,g.off[u],g.off[u+1]);
            source_labels[u]=strict_degree;
        }
        initial_nonstrict_transcript=transcript;
        front_stats.initial_nonstrict_edges=candidate_edges;
        for (const auto& edges:equality_edges) {
            front_stats.initial_nonstrict_edges+=edges.size();
            for (EdgeId e:edges) initial_nonstrict_transcript[e >> 6] |= uint64_t{1} << (e & 63U);
        }
        front_stats.cover_source_cut=0;
        front_stats.cover_target_cut=maximum_in_degree;
        front_stats.cover_estimated_cost=g.m;
    } else {
    front_stats.cover_cut_selection_executed=true;
    const size_t out_volume_size = static_cast<size_t>(maximum_out_degree) + 2;
    const size_t in_volume_size = static_cast<size_t>(maximum_in_degree) + 2;
    std::unique_ptr<uint64_t[]> out_volume(new uint64_t[out_volume_size]);
    std::unique_ptr<uint64_t[]> in_volume(new uint64_t[in_volume_size]);
    std::vector<DegreeCoverWorkspace> cover_workspace(
        static_cast<size_t>(omp_get_max_threads()));
    uint32_t source_cut = 0;
    uint32_t target_cut = maximum_in_degree;
    uint64_t cover_estimated_cost = g.m;
    std::vector<uint32_t> cover_targets;
    std::vector<std::vector<EdgeId>> local_equality_edges(
        static_cast<size_t>(omp_get_max_threads()));
    // Publish stage counts at their worksharing barriers.
    uint64_t source_phase_edges = 0;
    uint64_t target_phase_edges = 0;
    #pragma omp parallel
    {
        const int cover_tid = omp_get_thread_num();
        const int cover_team = omp_get_num_threads();
        DegreeCoverWorkspace& cover_local = cover_workspace[cover_tid];
        std::fill_n(cover_local.out_bins, kCachedDegreeBuckets, uint64_t{0});
        std::fill_n(cover_local.in_bins, kCachedDegreeBuckets, uint64_t{0});
        // Initialize shared bins before atomic updates.
        #pragma omp for schedule(static)
        for (size_t index = 0; index < out_volume_size + in_volume_size; ++index) {
            if (index < out_volume_size) out_volume[index] = 0;
            else in_volume[index - out_volume_size] = 0;
        }
        #pragma omp for schedule(static)
        for (uint32_t vertex = 0; vertex < g.n; ++vertex) {
            const uint32_t out_degree = g.out0[vertex];
            const uint32_t in_degree = g.in0[vertex];
            if (out_degree < kCachedDegreeBuckets) {
                cover_local.out_bins[out_degree] += out_degree;
            } else {
                #pragma omp atomic update
                out_volume[out_degree] += out_degree;
            }
            if (in_degree < kCachedDegreeBuckets) {
                cover_local.in_bins[in_degree] += in_degree;
            } else {
                #pragma omp atomic update
                in_volume[in_degree] += in_degree;
            }
        }
        // Reduce bins from the active team.
        #pragma omp for schedule(static)
        for (size_t bucket = 0; bucket < kCachedDegreeBuckets; ++bucket) {
            uint64_t out_sum = 0, in_sum = 0;
            for (int worker = 0; worker < cover_team; ++worker) {
                out_sum += cover_workspace[worker].out_bins[bucket];
                in_sum += cover_workspace[worker].in_bins[bucket];
            }
            if (bucket <= maximum_out_degree) out_volume[bucket] = out_sum;
            if (bucket <= maximum_in_degree) in_volume[bucket] = in_sum;
        }
        const size_t out_begin = out_volume_size * static_cast<size_t>(cover_tid) / cover_team;
        const size_t out_end = out_volume_size * static_cast<size_t>(cover_tid + 1) / cover_team;
        const size_t in_begin = in_volume_size * static_cast<size_t>(cover_tid) / cover_team;
        const size_t in_end = in_volume_size * static_cast<size_t>(cover_tid + 1) / cover_team;
        uint64_t out_sum = 0, in_sum = 0;
        for (size_t index = out_end; index > out_begin;) {
            --index;
            out_sum += out_volume[index];
            out_volume[index] = out_sum;
        }
        for (size_t index = in_end; index > in_begin;) {
            --index;
            in_sum += in_volume[index];
            in_volume[index] = in_sum;
        }
        cover_local.out_total = out_sum;
        cover_local.in_total = in_sum;
        #pragma omp barrier
        #pragma omp single
        {
            uint64_t out_carry = 0, in_carry = 0;
            for (int worker = cover_team; worker-- > 0;) {
                cover_workspace[worker].out_offset = out_carry;
                cover_workspace[worker].in_offset = in_carry;
                out_carry += cover_workspace[worker].out_total;
                in_carry += cover_workspace[worker].in_total;
            }
        }
        for (size_t index = out_begin; index < out_end; ++index)
            out_volume[index] += cover_local.out_offset;
        for (size_t index = in_begin; index < in_end; ++index)
            in_volume[index] += cover_local.in_offset;
        // Publish suffix sums before evaluating cuts.
        #pragma omp barrier
        cover_local.best_cost = g.m;
        cover_local.best_source_cut = 0;
        cover_local.best_target_cut = maximum_in_degree;
        const uint64_t cut_begin = 1 + static_cast<uint64_t>(maximum_out_degree) * cover_tid / cover_team;
        const uint64_t cut_end = 1 + static_cast<uint64_t>(maximum_out_degree) * (cover_tid + 1) / cover_team;
        for (uint64_t candidate_cut = cut_begin; candidate_cut < cut_end; ++candidate_cut) {
            const uint64_t candidate_target_cut = (lower - 1) / candidate_cut;
            const uint64_t row_checks = out_volume[candidate_cut + 1];
            const uint64_t column_checks = candidate_target_cut < maximum_in_degree
                ? in_volume[static_cast<size_t>(candidate_target_cut) + 1] : 0;
            const uint64_t candidate_cost = row_checks + column_checks;
            if (candidate_cost < cover_local.best_cost ||
                (candidate_cost == cover_local.best_cost && candidate_cut < cover_local.best_source_cut)) {
                cover_local.best_cost = candidate_cost;
                cover_local.best_source_cut = static_cast<uint32_t>(candidate_cut);
                cover_local.best_target_cut = static_cast<uint32_t>(
                    std::min<uint64_t>(candidate_target_cut, maximum_in_degree));
            }
        }
        #pragma omp barrier
        #pragma omp single
        {
            // Break equal-cost ties by source threshold.
            for (int worker = 0; worker < cover_team; ++worker) {
                const auto& candidate = cover_workspace[worker];
                if (candidate.best_cost < cover_estimated_cost ||
                    (candidate.best_cost == cover_estimated_cost &&
                     candidate.best_source_cut < source_cut)) {
                    cover_estimated_cost = candidate.best_cost;
                    source_cut = candidate.best_source_cut;
                    target_cut = candidate.best_target_cut;
                }
            }
        }
        const uint64_t vertex_begin = static_cast<uint64_t>(g.n) * cover_tid / cover_team;
        const uint64_t vertex_end = static_cast<uint64_t>(g.n) * (cover_tid + 1) / cover_team;
        cover_local.target_count = 0;
        for (uint64_t vertex = vertex_begin; vertex < vertex_end; ++vertex)
            cover_local.target_count += g.in0[vertex] > target_cut;
        #pragma omp barrier
        #pragma omp single
        {
            size_t total = 0;
            for (int worker = 0; worker < cover_team; ++worker) {
                cover_workspace[worker].target_offset = total;
                total += cover_workspace[worker].target_count;
            }
            cover_targets.resize(total);
        }
        size_t target_position = cover_local.target_offset;
        for (uint64_t vertex = vertex_begin; vertex < vertex_end; ++vertex) {
            if (g.in0[vertex] > target_cut)
                cover_targets[target_position++] = static_cast<uint32_t>(vertex);
        }
        // Build ordered, disjoint target slices.
        #pragma omp for reduction(+:source_phase_edges) schedule(dynamic, 256)
        for (uint32_t u = 0; u < g.n; ++u) {
            if (g.out0[u] <= source_cut) continue;
            uint32_t strict_source_degree = 0;
            uint64_t current_word = std::numeric_limits<uint64_t>::max();
            uint64_t strict_mask = 0;
            for (uint64_t e = g.off[u]; e < g.off[u + 1]; ++e) {
                const uint32_t target = work_endpoint(g.dst[e]);
                const uint64_t product = edge_weight(g.out0[u], g.in0[target]);
                if (product == lower) {
                    local_equality_edges[static_cast<size_t>(omp_get_thread_num())]
                        .push_back(static_cast<EdgeId>(e));
                }
                if (product > lower) {
                    const uint64_t word_index = e >> 6;
                    if (word_index != current_word) {
                        if (current_word != std::numeric_limits<uint64_t>::max()) {
                            merge_source_row_word(
                                transcript, current_word, strict_mask,
                                g.off[u], g.off[u + 1]);
                        }
                        current_word = word_index;
                        strict_mask = 0;
                    }
                    strict_mask |= uint64_t{1} << (e & 63U);
                    ++strict_source_degree;
                    ++source_phase_edges;
                    #pragma omp atomic update
                    target_labels[target] += 1;
                }
            }
            if (current_word != std::numeric_limits<uint64_t>::max()) {
                merge_source_row_word(
                    transcript, current_word, strict_mask,
                    g.off[u], g.off[u + 1]);
            }
            source_labels[u] = strict_source_degree;
        }
        // Finish atomic source updates before the target phase.
        #pragma omp for reduction(+:target_phase_edges) schedule(dynamic, 1)
        for (size_t target_index = 0;
             target_index < cover_targets.size();
             ++target_index) {
            const uint32_t v = cover_targets[target_index];
            uint32_t strict_target_add = 0;
            for (uint64_t pos = g.in_off[v]; pos < g.in_off[v + 1]; ++pos) {
                const EdgeId edge = work_reverse_index(g.in_edges[pos]);
                const uint32_t u = work_endpoint(g.src[edge]);
                if (g.out0[u] > source_cut) continue;
                const uint64_t product = edge_weight(g.out0[u], g.in0[v]);
                if (product == lower) {
                    local_equality_edges[
                        static_cast<size_t>(omp_get_thread_num())].push_back(edge);
                }
                if (product > lower) {
                    atomic_set_transcript_bit(transcript, edge);
                    ++strict_target_add;
                    ++target_phase_edges;
                }
            }
            target_labels[v] += strict_target_add;
        }
        // Finish target updates before recounting.
        #pragma omp for schedule(dynamic, 256) nowait
        for (uint32_t u = 0; u < g.n; ++u) {
            // Recount rows affected by target coverage.
            if (g.out0[u] <= source_cut && target_phase_edges != 0)
                source_labels[u] = transcript_popcount_range(transcript, g.off[u], g.off[u + 1]);
        }
    }
    front_stats.cover_source_cut=source_cut;
    front_stats.cover_target_cut=target_cut;
    front_stats.cover_estimated_cost=cover_estimated_cost;
    candidate_edges = source_phase_edges + target_phase_edges;
    front_stats.initial_nonstrict_edges=candidate_edges;
    initial_nonstrict_transcript = transcript;
    for (const auto& edges : local_equality_edges) {
        front_stats.initial_nonstrict_edges+=edges.size();
        for (EdgeId edge : edges) {
            initial_nonstrict_transcript[edge >> 6] |=
                uint64_t{1} << (edge & 63U);
        }
    }

    }
    front_stats.initial_strict_edges=candidate_edges;

    std::vector<uint32_t> active_sources;
    std::vector<uint32_t> active_targets;
    active_sources.reserve(g.n);
    active_targets.reserve(g.n);
    for (uint32_t v = 0; v < g.n; ++v) {
        if (source_labels[v] != 0) active_sources.push_back(v);
        if (target_labels[v] != 0) active_targets.push_back(v);
    }
    uint32_t lifts = 0;
    ParametricSearchResult first_parametric_search;
    bool parametric_certified = false;
    {
        std::vector<uint32_t> direct_index_targets = active_targets;
        std::vector<uint32_t> direct_target_index(
            g.n, std::numeric_limits<uint32_t>::max());
        for (size_t index = 0; index < direct_index_targets.size(); ++index) {
            direct_target_index[direct_index_targets[index]] =
                static_cast<uint32_t>(index);
        }
#ifdef BGWC_MEMORY_DIAGNOSTICS
        std::fprintf(stderr, "[memory-allocation] direct_decrements P=%d T=%llu bytes=%llu\n",
                     omp_get_max_threads(),
                     static_cast<unsigned long long>(direct_index_targets.size()),
                     static_cast<unsigned long long>(4ULL * omp_get_max_threads() * direct_index_targets.size()));
#endif
        memory_stage("direct_decrements.allocate.begin");
        std::vector<std::vector<uint32_t>> local_target_decrements(
            static_cast<size_t>(omp_get_max_threads()),
            std::vector<uint32_t>(direct_index_targets.size(), 0));
        memory_stage("direct_decrements.allocate.complete");
        auto apply_target_decrements = [&]() {
            #pragma omp parallel for schedule(static)
            for (size_t index = 0; index < direct_index_targets.size(); ++index) {
                uint32_t decrement = 0;
                for (const auto& local : local_target_decrements) {
                    decrement += local[index];
                }
                target_labels[direct_index_targets[index]] -= decrement;
            }
        };
        uint64_t direct_alive_edges = candidate_edges;
        std::vector<WordId> active_words;
        std::vector<WordId> next_active_words;
        std::vector<std::vector<WordId>> local_active_words(
            static_cast<size_t>(omp_get_max_threads()));
        while (true) {
            if (lifts != 0)
                throw std::runtime_error("unexpected second outer strict closure");
            // Compact the active-word frontier after each strict peeling round.
            if (lifts == 0) {
                #pragma omp parallel
                {
                    auto& kept_words = local_active_words[
                        static_cast<size_t>(omp_get_thread_num())];
                    kept_words.clear();
                    #pragma omp for schedule(static) nowait
                    for (uint64_t word_index = 0;
                         word_index < transcript_words;
                         ++word_index) {
                        if (transcript[word_index] != 0) {
                            kept_words.push_back(static_cast<WordId>(word_index));
                        }
                    }
                }
                append_thread_vectors(active_words, local_active_words);
            }
            {
                while (true) {
                    ++front_stats.strict_rounds;
                    uint64_t removed = 0;
                    #pragma omp parallel reduction(+:removed)
                    {
                        const int tid = omp_get_thread_num();
                        auto& target_decrement =
                            local_target_decrements[static_cast<size_t>(tid)];
                        std::fill(
                            target_decrement.begin(), target_decrement.end(), 0);
                        auto& kept_words =
                            local_active_words[static_cast<size_t>(tid)];
                        kept_words.clear();
                        #pragma omp for schedule(static) nowait
                        for (size_t index = 0; index < active_words.size(); ++index) {
                            const uint64_t word_index = active_words[index];
                            const WordPruneStats stats = prune_transcript_word(
                                g, transcript, word_index, lower,
                                source_labels, direct_target_index,
                                target_labels, target_decrement);
                            removed += stats.removed;
                            if (transcript[word_index] != 0) {
                                kept_words.push_back(static_cast<WordId>(word_index));
                            }
                        }
                    }
                    apply_target_decrements();
                    append_thread_vectors(next_active_words, local_active_words);
                    active_words.swap(next_active_words);
                    direct_alive_edges -= removed;
                    if (removed == 0 || direct_alive_edges == 0) break;
                }
            }

            if (direct_alive_edges == 0) {
                candidate_edges = 0;
                break;
            }
#ifdef BGWC_PROFILE
            Timer batch_threshold_timer;
#endif
            // Release closure scratch while retaining the active-word certificate.
            memory_stage("strict_closure.complete.before_scratch_release");
            std::vector<std::vector<uint32_t>>().swap(local_target_decrements);
            std::vector<uint32_t>().swap(direct_target_index);
            std::vector<uint32_t>().swap(direct_index_targets);
            std::vector<std::vector<WordId>>().swap(local_active_words);
            std::vector<WordId>().swap(next_active_words);
            memory_stage("strict_closure.scratch_released.before_exact_search");
            first_parametric_search = exact_w_by_parametric_closure(
                g, transcript, active_words,
                active_sources, active_targets,
                source_labels, target_labels,
                lower, direct_alive_edges);
#ifdef BGWC_PROFILE
            profile_batch_threshold_time+=batch_threshold_timer.elapsed();
#endif
            if (first_parametric_search.w_star <= lower ||
                first_parametric_search.h_edge_ids.empty())
                throw std::runtime_error("nonempty strict closure lacks a valid exact PMPC lift");
            lower = first_parametric_search.w_star;
            parametric_certified = true;
            ++lifts;


            size_t source_write = 0;
            for (uint32_t vertex : active_sources) {
                if (__atomic_load_n(&source_labels[vertex], __ATOMIC_RELAXED) != 0) {
                    active_sources[source_write++] = vertex;
                }
            }
            active_sources.resize(source_write);
            size_t target_write = 0;
            for (uint32_t vertex : active_targets) {
                if (__atomic_load_n(&target_labels[vertex], __ATOMIC_RELAXED) != 0) {
                    active_targets[target_write++] = vertex;
                }
            }
            active_targets.resize(target_write);

            restore_bitmap_word_owned(
                transcript, active_words, first_parametric_search.h_edge_ids);
            candidate_edges = first_parametric_search.h_edge_ids.size();
            #pragma omp parallel for schedule(static)
            for (size_t index = 0; index < active_sources.size(); ++index) {
                source_labels[active_sources[index]] = 0;
            }
            #pragma omp parallel for schedule(static)
            for (size_t index = 0; index < active_targets.size(); ++index) {
                target_labels[active_targets[index]] = 0;
            }
            break;
        }
    }

    uint32_t positive_sources = 0;
    uint32_t positive_targets = 0;
    for (uint32_t vertex : active_sources) {
        positive_sources += source_labels[vertex] != 0;
    }
    for (uint32_t vertex : active_targets) {
        positive_targets += target_labels[vertex] != 0;
    }

    if (lifts == 0) {
        transcript.swap(initial_nonstrict_transcript);
    } else {
        std::vector<uint64_t>().swap(initial_nonstrict_transcript);
    }
    if (positive_sources != 0 || positive_targets != 0) {
        throw std::runtime_error("strict certificate ended with positive labels");
    }

    uint64_t h_edges = 0;
    std::vector<uint32_t>& h_out_degree = source_labels;
    std::vector<uint32_t>& h_in_degree = target_labels;
    std::vector<uint32_t> h_active_sources;
    std::vector<uint32_t> h_active_targets;
    if (parametric_certified) {
        // Import the certified degrees.
        const auto& certificate = first_parametric_search;
        if (certificate.h_source_ids.size() != certificate.h_source_degrees.size() ||
            certificate.h_target_ids.size() != certificate.h_target_degrees.size()) {
            throw std::runtime_error("certified H degree dimensions disagree");
        }
        h_edges = certificate.h_edge_ids.size();
        h_active_sources.reserve(certificate.h_source_ids.size());
        h_active_targets.reserve(certificate.h_target_ids.size());
        for (size_t index = 0; index < certificate.h_source_ids.size(); ++index) {
            const uint32_t vertex = certificate.h_source_ids[index];
            const uint32_t degree = certificate.h_source_degrees[index];
            source_labels[vertex] = degree;
            if (degree != 0) h_active_sources.push_back(vertex);
        }
        for (size_t index = 0; index < certificate.h_target_ids.size(); ++index) {
            const uint32_t vertex = certificate.h_target_ids[index];
            const uint32_t degree = certificate.h_target_degrees[index];
            target_labels[vertex] = degree;
            if (degree != 0) h_active_targets.push_back(vertex);
        }
    } else {
        ParallelReconstructionResult parallel_rebuild =
            parallel_sparse_reconstruction(
                g, transcript, source_labels, target_labels, lower,
                maximum_out_degree);
        h_edges = parallel_rebuild.edges;
        h_active_sources = std::move(parallel_rebuild.sources);
        h_active_targets = std::move(parallel_rebuild.targets);
    }

    const double maximum_induced_phase_time = full_algorithm_timer.elapsed();
    const WorkSlot front_work_snapshot=work_totals();
    Timer initialization_and_core_search_timer;
#ifdef BGWC_PROFILE
    Timer initial_candidate_prepare_timer;
#endif
    // Find a positive threshold pair with product w_star.
    if (g.m != 0 && h_edges == 0)
        throw std::runtime_error("nonempty graph has an empty exact H certificate");
    std::vector<uint32_t> reconstructed_candidates =
        collect_transcript_candidates(
            g, transcript, h_active_sources, h_out_degree, h_in_degree, lower);
    XYPairResult reconstructed_xy =
        xy_from_single_candidate(lower, reconstructed_candidates);
    std::vector<uint8_t> quality_initial_candidate_sources(g.n, 0);
    std::vector<uint8_t> quality_initial_candidate_targets(g.n, 0);
    CoreStats certified_initial_candidate_stats;
    uint32_t initial_source_count = 0, initial_target_count = 0;
    int ineligible_initial_role = 0;
    if (reconstructed_xy.found && transcript_satisfies_xy(
            h_active_sources, h_active_targets, h_out_degree, h_in_degree,
            reconstructed_xy.x, reconstructed_xy.y)) {
        if (reconstructed_xy.x == 0 || reconstructed_xy.y == 0 ||
            edge_weight(reconstructed_xy.x, reconstructed_xy.y) != lower)
            throw std::runtime_error("H initial_candidate pair does not multiply to w_star");
        certified_initial_candidate_stats.edges = h_edges;
        #pragma omp parallel for reduction(+:initial_source_count,initial_target_count) reduction(max:ineligible_initial_role) schedule(static)
        for (uint32_t vertex = 0; vertex < g.n; ++vertex) {
            const bool source = h_out_degree[vertex] != 0;
            const bool target = h_in_degree[vertex] != 0;
            quality_initial_candidate_sources[vertex] = source;
            quality_initial_candidate_targets[vertex] = target;
            initial_source_count += source;
            initial_target_count += target;
            if ((source && h_out_degree[vertex] < reconstructed_xy.x) ||
                (target && h_in_degree[vertex] < reconstructed_xy.y))
                ineligible_initial_role = 1;
        }
    } else {
        State initial_induced_state = materialize_transcript_state(
            g, transcript, h_active_sources, h_out_degree, h_in_degree, h_edges);
        if (!reconstructed_xy.found) {
            State h_work = copy_state(initial_induced_state, g.n);
            reconstructed_xy = find_xy_from_wstar_induced(
                g, h_work, lower, reconstructed_candidates);
        }
        if (g.m != 0 && (!reconstructed_xy.found || reconstructed_xy.x == 0 ||
            reconstructed_xy.y == 0 || edge_weight(reconstructed_xy.x, reconstructed_xy.y) != lower))
            throw std::runtime_error("nonempty exact H has no certified initial_candidate pair");
        State extracted_initial_state;
        const State* selected_initial_state = &initial_induced_state;
        if (reconstructed_xy.found) {
            extracted_initial_state = extract_initial_candidate(
                g, initial_induced_state, reconstructed_xy.x, reconstructed_xy.y);
            selected_initial_state = &extracted_initial_state;
        }
        // Extract the induced subgraph on the eligible source and target roles.
        certified_initial_candidate_stats.edges = selected_initial_state->alive_edges;
        #pragma omp parallel for reduction(+:initial_source_count,initial_target_count) reduction(max:ineligible_initial_role) schedule(static)
        for (uint32_t vertex = 0; vertex < g.n; ++vertex) {
            const bool source = selected_initial_state->out_deg[vertex] != 0;
            const bool target =
                selected_initial_state->in_deg[vertex].load(std::memory_order_relaxed) != 0;
            quality_initial_candidate_sources[vertex] = source;
            quality_initial_candidate_targets[vertex] = target;
            initial_source_count += source;
            initial_target_count += target;
            if ((source && h_out_degree[vertex] < reconstructed_xy.x) ||
                (target && h_in_degree[vertex] < reconstructed_xy.y))
                ineligible_initial_role = 1;
        }
    }
    if (ineligible_initial_role)
        throw std::runtime_error("certified initial_candidate role is ineligible under exact H degrees");
    certified_initial_candidate_stats.S = initial_source_count;
    certified_initial_candidate_stats.T = initial_target_count;
    CertifiedInitialCandidate certified_initial_candidate = CertifiedInitialCandidate::from_maximum_h_subset(
        g, lower, reconstructed_xy, std::move(quality_initial_candidate_sources),
        std::move(quality_initial_candidate_targets), certified_initial_candidate_stats);
#ifdef BGWC_PROFILE
    const double profile_initial_candidate_prepare_time=initial_candidate_prepare_timer.elapsed();
#endif
    Timer quality_timer;
    QualityCoreResult quality_result = quality_certify_dds(
        g, lower, std::move(certified_initial_candidate), quality_original_stats, disable_early_stop,
        disable_state_reuse, disable_trace_bound, disable_safe_candidate);
    const double quality_time = quality_timer.elapsed();
    if (!quality_result.certified) {
        throw std::runtime_error("DDS quality certificate was not established");
    }
    const CoreStats& reconstructed_stats = quality_result.stats;
    const double initialization_and_core_search_time = initialization_and_core_search_timer.elapsed();
    const double end_to_end_algorithm_time =
        full_algorithm_timer.elapsed();

    work_active=false; // Exclude export and reporting from counters.
    const WorkSlot complete_work_snapshot=work_totals();
    if (verify_output_membership) {
        uint64_t actual_sources=0,actual_targets=0,actual_edges=0;
        #pragma omp parallel for reduction(+:actual_sources,actual_targets) schedule(static)
        for (uint64_t v=0;v<g.n;++v) {
            actual_sources+=quality_result.source_membership[v]!=0;
            actual_targets+=quality_result.target_membership[v]!=0;
        }
        #pragma omp parallel for reduction(+:actual_edges) schedule(static)
        for (uint64_t e=0;e<g.m;++e)
            actual_edges+=quality_result.source_membership[g.src[e]] && quality_result.target_membership[g.dst[e]];
        if (actual_sources!=reconstructed_stats.S || actual_targets!=reconstructed_stats.T || actual_edges!=reconstructed_stats.edges)
            throw std::runtime_error("output membership does not reproduce the original induced-subgraph statistics");
    }
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "algorithm_time: " << end_to_end_algorithm_time << "\n";
    std::cout << "io_time: " << read_time << "\n";
    std::cout << "w_star: " << lower << "\n";
    std::cout << "density: " << reconstructed_stats.density << "\n";
    std::cout << "|S|: " << reconstructed_stats.S << "\n";
    std::cout << "|T|: " << reconstructed_stats.T << "\n";

    if (!save_prefix.empty()) {
        save_quality_output(g, quality_result, save_prefix);
    }
    return 0;
}
