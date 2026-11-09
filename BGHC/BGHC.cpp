// BGHC: maximum core number and complete highest-order core

#if defined(__AVX512F__) && defined(__AVX512BW__) && !defined(BGHC_SCALAR)
#define BGHC_PLAN_GROUP_AVX512
#define BGHC_RAW64_AVX512BW
#include <immintrin.h>
#endif
#if defined(__SSE2__) && !defined(BGHC_SCALAR)
#include <emmintrin.h>
#endif
#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <omp.h>

using Vertex = std::uint32_t;
using Edge = std::uint64_t;

// Wide arithmetic for initial bounds.
__extension__ using WideInteger = unsigned __int128;
// Shared types and synchronization
struct TopTwoStats {
    Vertex maximum=0,second=0;
    Edge count=0;
    void add(Vertex value) {
        if(value>maximum) {second=maximum;maximum=value;count=1;}
        else if(value==maximum) ++count;
        else second=std::max(second,value);
    }
    void merge(const TopTwoStats& other) {
        if(!other.count) return;
        if(!count) {*this=other;return;}
        if(other.maximum>maximum) {
            second=std::max(maximum,other.second);
            maximum=other.maximum;count=other.count;
        } else if(other.maximum==maximum) {
            second=std::max(second,other.second);count+=other.count;
        } else second=std::max(second,other.maximum);
    }
};
struct AtomicLabelView {
    const Vertex* values;
    Vertex operator[](Vertex v) const {
        return __atomic_load_n(values+v,__ATOMIC_RELAXED);
    }
};
// SIMD path selected at compile time.
[[maybe_unused]] static const char* raw64_compiled_path() {
#if defined(BGHC_RAW64_AVX512BW)
    return "AVX512BW_UNSIGNED_MASK64";
#elif defined(__SSE2__) && !defined(BGHC_SCALAR)
    return "SSE2_FOUR_MASK16";
#else
    return "SCALAR_MASK64";
#endif
}
struct Timer {
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    double seconds() const {
        return std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    }
};
// Graph representation and input
struct Graph {
    Vertex n = 0;
    Edge m = 0;
    std::vector<Edge> xadj;
    std::vector<Vertex> adj;
};
static bool read_exact(std::ifstream& input, void* destination, std::size_t bytes) {
    input.read(static_cast<char*>(destination), static_cast<std::streamsize>(bytes));
    return static_cast<std::size_t>(input.gcount()) == bytes;
}

template <class Value>
static bool read_scalar(std::ifstream& input, int bytes, Value& output) {
    if (bytes == 4) {
        std::uint32_t value = 0;
        if (!read_exact(input, &value, sizeof(value))) return false;
        output = static_cast<Value>(value);
        return true;
    }
    if (bytes == 8) {
        std::uint64_t value = 0;
        if (!read_exact(input, &value, sizeof(value))) return false;
        output = static_cast<Value>(value);
        return true;
    }
    return false;
}

static Graph read_bghc_binary_graph(const char* path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error(std::string("cannot open graph: ") + path);

    int vertex_bytes = 0;
    int edge_bytes = 0;
    if (!read_exact(input, &vertex_bytes, sizeof(vertex_bytes)) ||
        !read_exact(input, &edge_bytes, sizeof(edge_bytes))) {
        throw std::runtime_error("truncated BGHC binary header");
    }
    if (vertex_bytes != 4 || (edge_bytes != 4 && edge_bytes != 8)) {
        throw std::runtime_error("unsupported BGHC binary integer widths");
    }

    Graph graph;
    if (!read_scalar(input, vertex_bytes, graph.n) ||
        !read_scalar(input, edge_bytes, graph.m)) {
        throw std::runtime_error("truncated BGHC binary graph sizes");
    }
    graph.xadj.assign(static_cast<std::size_t>(graph.n) + 1, 0);
    for (Vertex v = 0; v < graph.n; ++v) {
        Edge degree = 0;
        if (!read_scalar(input, edge_bytes, degree)) {
            throw std::runtime_error("truncated BGHC binary degree sequence");
        }
        if (degree > std::numeric_limits<Edge>::max() - graph.xadj[v]) {
            throw std::runtime_error("BGHC binary adjacency volume overflow");
        }
        graph.xadj[static_cast<std::size_t>(v) + 1] = graph.xadj[v] + degree;
    }
    const Edge entries = graph.xadj.back();
    if ((entries & 1ULL) != 0) {
        throw std::runtime_error("BGHC binary adjacency volume is not an undirected edge volume");
    }
    // Derive the edge count from the adjacency data.
    graph.m = entries / 2;
    if (entries > static_cast<Edge>(std::numeric_limits<std::size_t>::max())) {
        throw std::runtime_error("BGHC binary adjacency array is too large");
    }
    graph.adj.resize(static_cast<std::size_t>(entries));
    if (!graph.adj.empty() &&
        !read_exact(
            input, graph.adj.data(),
            graph.adj.size() * sizeof(Vertex))) {
        throw std::runtime_error("truncated BGHC binary adjacency array");
    }
    for (Vertex v = 0; v < graph.n; ++v) {
        Vertex previous = 0;
        bool first = true;
        for (Edge e = graph.xadj[v];
             e < graph.xadj[static_cast<std::size_t>(v) + 1]; ++e) {
            const Vertex u = graph.adj[static_cast<std::size_t>(e)];
            if (u >= graph.n) {
                throw std::runtime_error("BGHC binary neighbor is out of range");
            }
            if (!first && u <= previous) {
                throw std::runtime_error(
                    "BGHC requires strictly sorted adjacency rows");
            }
            previous = u;
            first = false;
        }
    }
    return graph;
}

struct TextGraphHeader {
    Edge declared_vertices = 0;
    Edge declared_edges = 0;
    std::size_t edge_start_line = 0;
    bool force_one_indexed = false;
};

static bool ascii_space(char value) {
    return value == ' ' || value == '\t' || value == '\r' || value == '\n';
}

static std::string_view text_line_view(
    const std::string& line, bool first_line) {
    std::string_view view(line);
    if (first_line && view.size() >= 3 &&
        static_cast<unsigned char>(view[0]) == 0xef &&
        static_cast<unsigned char>(view[1]) == 0xbb &&
        static_cast<unsigned char>(view[2]) == 0xbf) {
        view.remove_prefix(3);
    }
    while (!view.empty() && ascii_space(view.front())) view.remove_prefix(1);
    return view;
}

static bool take_unsigned(std::string_view& view, Edge& value) {
    while (!view.empty() && ascii_space(view.front())) view.remove_prefix(1);
    if (view.empty() || view.front() == '-') return false;
    const char* begin = view.data();
    const char* end = begin + view.size();
    const auto parsed = std::from_chars(begin, end, value);
    if (parsed.ec != std::errc{} || parsed.ptr == begin) return false;
    view.remove_prefix(static_cast<std::size_t>(parsed.ptr - begin));
    return true;
}

static bool parse_two_unsigned(
    std::string_view view, Edge& first, Edge& second) {
    return take_unsigned(view, first) && take_unsigned(view, second);
}

static bool parse_three_unsigned(
    std::string_view view, Edge& first, Edge& second, Edge& third) {
    return take_unsigned(view, first) && take_unsigned(view, second) &&
        take_unsigned(view, third);
}

static bool parse_labeled_unsigned(
    std::string_view view, std::string_view upper_label,
    std::string_view lower_label, Edge& value) {
    std::size_t position = view.find(upper_label);
    std::size_t label_size = upper_label.size();
    if (position == std::string_view::npos) {
        position = view.find(lower_label);
        label_size = lower_label.size();
    }
    if (position == std::string_view::npos) return false;
    view.remove_prefix(position + label_size);
    return take_unsigned(view, value);
}

[[noreturn]] static void throw_text_error(
    const char* path, std::size_t line, const std::string& message) {
    std::string full(path);
    if (line != 0) full += ":" + std::to_string(line);
    full += ": " + message;
    throw std::runtime_error(full);
}

static void validate_text_header(
    const char* path, std::size_t line, const TextGraphHeader& header) {
    if (header.declared_vertices == 0) {
        throw_text_error(path, line, "TXT vertex count must be positive");
    }
    if (header.declared_vertices >
        static_cast<Edge>(std::numeric_limits<Vertex>::max())) {
        throw_text_error(path, line, "TXT vertex count exceeds uint32_t");
    }
}

static TextGraphHeader inspect_text_header(const char* path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error(std::string("cannot open graph: ") + path);
    }

    bool matrix_market = false;
    bool snap_vertices_found = false;
    bool snap_edges_found = false;
    Edge snap_vertices = 0;
    Edge snap_edges = 0;
    std::string line;
    std::size_t line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        std::string_view view = text_line_view(line, line_number == 1);
        if (view.empty()) continue;
        if (view.substr(0, 14) == "%%MatrixMarket") {
            matrix_market = true;
            continue;
        }
        if (view.front() == '#' || view.front() == '%') {
            Edge value = 0;
            if (parse_labeled_unsigned(view, "Nodes:", "nodes:", value)) {
                snap_vertices = value;
                snap_vertices_found = true;
            }
            if (parse_labeled_unsigned(view, "Edges:", "edges:", value)) {
                snap_edges = value;
                snap_edges_found = true;
            }
            continue;
        }

        TextGraphHeader header;
        if (matrix_market) {
            Edge rows = 0;
            Edge columns = 0;
            if (!parse_three_unsigned(view, rows, columns, header.declared_edges)) {
                throw_text_error(path, line_number,
                                 "invalid MatrixMarket size line");
            }
            if (rows != columns) {
                throw_text_error(path, line_number,
                                 "MatrixMarket graph must be square");
            }
            header.declared_vertices = rows;
            header.edge_start_line = line_number + 1;
            header.force_one_indexed = true;

        } else if (snap_vertices_found && snap_edges_found) {
            header.declared_vertices = snap_vertices;
            header.declared_edges = snap_edges;
            header.edge_start_line = line_number;

        } else {
            if (!parse_two_unsigned(
                    view, header.declared_vertices, header.declared_edges)) {
                throw_text_error(path, line_number,
                                 "expected TXT header: <vertices> <edges>");
            }
            header.edge_start_line = line_number + 1;

        }
        validate_text_header(path, line_number, header);
        return header;
    }

    if (snap_vertices_found && snap_edges_found && snap_edges == 0) {
        TextGraphHeader header;
        header.declared_vertices = snap_vertices;
        header.declared_edges = 0;
        header.edge_start_line = line_number + 1;

        validate_text_header(path, line_number, header);
        return header;
    }
    throw_text_error(path, 0, "TXT graph has no usable header");
}

template <class Callback>
static Edge scan_text_edges(
    const char* path, const TextGraphHeader& header, Callback&& callback) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error(std::string("cannot open graph: ") + path);
    }
    std::string line;
    std::size_t line_number = 0;
    Edge parsed_edges = 0;
    while (std::getline(input, line)) {
        ++line_number;
        if (line_number < header.edge_start_line) continue;
        std::string_view view = text_line_view(line, line_number == 1);
        if (view.empty() || view.front() == '#' || view.front() == '%') {
            continue;
        }
        Edge first = 0;
        Edge second = 0;
        if (!parse_two_unsigned(view, first, second)) {
            throw_text_error(path, line_number,
                             "malformed TXT edge; expected two vertex IDs");
        }
        if (parsed_edges == std::numeric_limits<Edge>::max()) {
            throw_text_error(path, line_number, "TXT edge count overflow");
        }
        ++parsed_edges;
        callback(first, second, line_number);
    }
    if (parsed_edges != header.declared_edges) {
        throw_text_error(
            path, 0,
            "TXT edge count mismatch: header=" +
                std::to_string(header.declared_edges) + " parsed=" +
                std::to_string(parsed_edges));
    }
    return parsed_edges;
}

static Graph read_text_graph(const char* path) {
    const TextGraphHeader header = inspect_text_header(path);
    const std::size_t raw_vertex_slots =
        static_cast<std::size_t>(header.declared_vertices) + 1;
    std::vector<Vertex> raw_degree(raw_vertex_slots, 0);
    bool saw_edge = false;
    bool saw_zero = false;
    Edge maximum_id = 0;
    Edge self_loops = 0;

    const auto increment_degree = [&](Edge raw, std::size_t line) {
        Vertex& degree = raw_degree[static_cast<std::size_t>(raw)];
        if (degree == std::numeric_limits<Vertex>::max()) {
            throw_text_error(path, line, "TXT vertex degree exceeds uint32_t");
        }
        ++degree;
    };

    scan_text_edges(
        path, header,
        [&](Edge first, Edge second, std::size_t line) {
            if (first > header.declared_vertices ||
                second > header.declared_vertices) {
                throw_text_error(path, line,
                                 "TXT endpoint exceeds declared vertex range");
            }
            saw_edge = true;
            saw_zero = saw_zero || first == 0 || second == 0;
            maximum_id = std::max(maximum_id, std::max(first, second));
            if (first == second) {
                ++self_loops;
                return;
            }
            increment_degree(first, line);
            increment_degree(second, line);
        });

    const bool one_indexed = header.force_one_indexed ||
        (!saw_zero && saw_edge && maximum_id == header.declared_vertices);
    if (one_indexed && saw_zero) {
        throw_text_error(path, 0, "one-based TXT graph contains vertex 0");
    }
    const Edge actual_vertices = one_indexed
        ? header.declared_vertices
        : std::max(
              header.declared_vertices,
              saw_edge ? maximum_id + 1 : header.declared_vertices);
    if (actual_vertices >
        static_cast<Edge>(std::numeric_limits<Vertex>::max())) {
        throw_text_error(path, 0, "normalized TXT vertex count exceeds uint32_t");
    }

    Graph graph;
    graph.n = static_cast<Vertex>(actual_vertices);

    std::vector<Vertex> degree(static_cast<std::size_t>(graph.n), 0);
    if (one_indexed) {
        for (Edge raw = 1; raw <= header.declared_vertices; ++raw) {
            degree[static_cast<std::size_t>(raw - 1)] =
                raw_degree[static_cast<std::size_t>(raw)];
        }
    } else {
        for (Edge raw = 0; raw < actual_vertices; ++raw) {
            degree[static_cast<std::size_t>(raw)] =
                raw_degree[static_cast<std::size_t>(raw)];
        }
    }

    graph.xadj.assign(static_cast<std::size_t>(graph.n) + 1, 0);
    for (Vertex v = 0; v < graph.n; ++v) {
        const Edge previous = graph.xadj[static_cast<std::size_t>(v)];
        const Edge value = degree[static_cast<std::size_t>(v)];
        if (value > std::numeric_limits<Edge>::max() - previous) {
            throw_text_error(path, 0, "TXT adjacency volume overflow");
        }
        graph.xadj[static_cast<std::size_t>(v) + 1] = previous + value;
    }
    if (graph.xadj.back() >
        static_cast<Edge>(std::numeric_limits<std::size_t>::max())) {
        throw_text_error(path, 0, "TXT adjacency array is too large");
    }
    graph.adj.resize(static_cast<std::size_t>(graph.xadj.back()));
    std::vector<Edge> cursor = graph.xadj;

    scan_text_edges(
        path, header,
        [&](Edge first, Edge second, std::size_t line) {
            if (first > header.declared_vertices ||
                second > header.declared_vertices) {
                throw_text_error(path, line,
                                 "TXT endpoint exceeds declared vertex range");
            }
            if (first == second) return;
            if (one_indexed) {
                if (first == 0 || second == 0) {
                    throw_text_error(path, line,
                                     "one-based TXT graph contains vertex 0");
                }
                --first;
                --second;
            }
            if (first >= graph.n || second >= graph.n) {
                throw_text_error(path, line,
                                 "normalized TXT endpoint is out of range");
            }
            graph.adj[static_cast<std::size_t>(
                cursor[static_cast<std::size_t>(first)]++)] =
                static_cast<Vertex>(second);
            graph.adj[static_cast<std::size_t>(
                cursor[static_cast<std::size_t>(second)]++)] =
                static_cast<Vertex>(first);
        });

    std::vector<Vertex> unique_degree(static_cast<std::size_t>(graph.n), 0);
    for (Vertex v = 0; v < graph.n; ++v) {
        const Edge begin = graph.xadj[static_cast<std::size_t>(v)];
        const Edge end = graph.xadj[static_cast<std::size_t>(v) + 1];
        if (begin == end) continue;
        Vertex* row = graph.adj.data() + static_cast<std::size_t>(begin);
        const std::size_t length = static_cast<std::size_t>(end - begin);
        std::sort(row, row + length);
        Vertex* unique_end = std::unique(row, row + length);
        unique_degree[static_cast<std::size_t>(v)] =
            static_cast<Vertex>(unique_end - row);
    }

    std::vector<Edge> normalized_xadj(
        static_cast<std::size_t>(graph.n) + 1, 0);
    for (Vertex v = 0; v < graph.n; ++v) {
        normalized_xadj[static_cast<std::size_t>(v) + 1] =
            normalized_xadj[static_cast<std::size_t>(v)] +
            unique_degree[static_cast<std::size_t>(v)];
    }
    for (Vertex v = 0; v < graph.n; ++v) {
        const Edge source = graph.xadj[static_cast<std::size_t>(v)];
        const Edge destination =
            normalized_xadj[static_cast<std::size_t>(v)];
        const std::size_t count = unique_degree[static_cast<std::size_t>(v)];
        if (count != 0 && source != destination) {
            std::memmove(
                graph.adj.data() + static_cast<std::size_t>(destination),
                graph.adj.data() + static_cast<std::size_t>(source),
                count * sizeof(Vertex));
        }
    }
    graph.xadj.swap(normalized_xadj);
    graph.adj.resize(static_cast<std::size_t>(graph.xadj.back()));
    if ((graph.xadj.back() & 1ULL) != 0) {
        throw_text_error(path, 0,
                         "TXT normalization produced asymmetric adjacency");
    }
    graph.m = graph.xadj.back() / 2;
    const Edge non_loop_records = header.declared_edges - self_loops;
    if (graph.m > non_loop_records) {
        throw_text_error(path, 0, "TXT normalization edge accounting failed");
    }
    return graph;
}

static bool has_case_insensitive_suffix(
    const std::string& path, const std::string& suffix) {
    if (path.size() < suffix.size()) return false;
    const std::size_t offset = path.size() - suffix.size();
    for (std::size_t index = 0; index < suffix.size(); ++index) {
        char left = path[offset + index];
        char right = suffix[index];
        if (left >= 'A' && left <= 'Z') left = static_cast<char>(left - 'A' + 'a');
        if (right >= 'A' && right <= 'Z') right = static_cast<char>(right - 'A' + 'a');
        if (left != right) return false;
    }
    return true;
}

static Graph read_graph(const char* path) {
    const std::string name(path);
    if (has_case_insensitive_suffix(name, ".txt") ||
        has_case_insensitive_suffix(name, ".mtx") ||
        has_case_insensitive_suffix(name, ".edges")) {
        return read_text_graph(path);
    }
    return read_bghc_binary_graph(path);
}

static inline Vertex graph_degree(const Graph& graph, Vertex v) {
    return static_cast<Vertex>(
        graph.xadj[static_cast<std::size_t>(v) + 1] - graph.xadj[v]);
}

// Initial bounds and statistics
static Vertex clique_edge_upper(Vertex n, Edge m) {
    if (n <= 1 || m == 0) return 0;
    const long double root = std::sqrt(1.0L + 8.0L * static_cast<long double>(m));
    Edge cap = static_cast<Edge>((root - 1.0L) / 2.0L);
    cap = std::min<Edge>(cap, static_cast<Edge>(n - 1));
    while ((cap + 1) <= static_cast<Edge>(n - 1) &&
           static_cast<WideInteger>(cap + 1) * static_cast<WideInteger>(cap + 2) <=
               static_cast<WideInteger>(2) * m) {
        ++cap;
    }
    while (static_cast<WideInteger>(cap) * static_cast<WideInteger>(cap + 1) >
           static_cast<WideInteger>(2) * m) {
        --cap;
    }
    return static_cast<Vertex>(cap);
}

static Vertex edge_count_degeneracy_lower(Vertex n, Edge m) {
    if (n <= 1 || m == 0) return 0;
    Vertex low = 0;
    Vertex high = n - 1;
    while (low < high) {
        const Vertex middle = low + (high - low) / 2;
        std::uint64_t factor_a = middle;
        std::uint64_t factor_b =
            2ULL * n - static_cast<std::uint64_t>(middle) - 1;
        if ((factor_a & 1ULL) == 0) factor_a /= 2;
        else factor_b /= 2;
        if (static_cast<WideInteger>(factor_a) * factor_b >= m) high = middle;
        else low = middle + 1;
    }
    return low;
}

struct Stage {
    Vertex threshold = 0, upper_after = 0;
    std::size_t candidates = 0, survivors = 0;
    Edge raw_entries = 0, build_entries = 0, compact_entries = 0;
    Edge raw_batches = 0, raw_batch_overscan = 0, raw_wide_batches = 0;
    Edge raw_histogram_updates = 0, raw_histogram_contributions = 0;
    Edge raw_histogram_zeroed_bins = 0;
    Edge raw_sync_published_rows = 0;
    Edge h_updates = 0, h_entries = 0, certificate_entries = 0;
    Edge skipped_vertex_blocks = 0;
    Edge local_h_updates = 0, local_h_entries = 0, async_h_descents = 0;
    Edge top_cap_descents = 0, active_row_checks = 0, remapped_entries = 0;
    Edge lazy_removed_entries = 0, lazy_tail_probes = 0, lazy_row_writes = 0;
    Edge lazy_remaining_entries = 0, local_label_reads = 0;
    Edge support_front_swaps = 0, support_front_probes = 0, support_front_writes = 0;
    Vertex prebuild_upper = 0;
    Edge prebuild_label_caps = 0, prebuild_top_rows = 0, prebuild_entries = 0;
    bool prebuild_success = false;
    Vertex effective_threshold = 0;
    unsigned rounds = 0;
    double domain_time = 0, raw_time = 0, build_time = 0, solve_time = 0;
    double scatter_time = 0;
};
struct PlanGroupStats {
    Edge batches=0,vertices=0,groups=0;
    void merge(const PlanGroupStats& x) {
        batches+=x.batches;vertices+=x.vertices;groups+=x.groups;
    }
};
#pragma omp declare reduction(plan_group_merge:PlanGroupStats:omp_out.merge(omp_in)) initializer(omp_priv=PlanGroupStats{})
[[maybe_unused]] static const char* plan_group_compiled_path() {
#ifdef BGHC_PLAN_GROUP_AVX512
    return "AVX512F_BW_CONTIGUOUS_DEGREE_GROUP16";
#else
    return "SCALAR_EXACT_DEGREE_GROUP16";
#endif
}
struct Result {
    Vertex k = 0, lower = 0, best_lower = 0, initial_upper = 0;
    std::vector<Vertex> core;
    std::vector<Stage> stages;
    double plan_time = 0;
    PlanGroupStats plan_groups;
    struct Seed {
        bool attempted=false, positive_query=false;
        Vertex before=0, after=0;
        Edge vertices=0, build_entries=0, compact_entries=0, reference_raw_entries=0;
        Edge h_entries=0, certificate_entries=0, h_updates=0, rounds=0;
        double build_time=0, solve_time=0, total_time=0;
    } seed;
};

// Exact local h-index kernel


static Vertex support_lazy_h(
    Vertex previous,Vertex floor,AtomicLabelView labels,
    std::vector<Vertex>& row,Edge begin,Edge& end,std::vector<Vertex>& histogram,
    Edge& label_reads,Edge& removed,Edge& tail_probes,Edge& writes,Edge& front_swaps) {
    if(previous<=floor) return previous;
    const Edge length=end-begin;
    if(length<=floor) return floor;
    Edge reject_budget=length-floor-1;
    // Count only unresolved ranks.
    bool histogram_ready=false;
    Vertex high=0;
    Edge cursor=begin,front=begin;
    while(cursor<end) {
        ++label_reads;
        const Vertex u=row[cursor],value=labels[u];
        if(value<=floor) {
            --end;++removed;
            if(cursor<end) {
                row[cursor]=row[end];
                ++tail_probes;++writes;
            }
            if(reject_budget--==0) return floor;
            continue;
        }
        if(value>=previous) {
            if(front!=cursor) {
                const Vertex displaced=row[front];
                row[front]=u;row[cursor]=displaced;
                ++front_swaps;
            }
            ++front;++cursor;
            if(++high==previous) return previous;
        } else {
            if(value>high) {
                if(!histogram_ready) {
                    histogram.assign(static_cast<std::size_t>(previous-floor-1),0);
                    histogram_ready=true;
                }
                ++histogram[value-floor-1];
            }
            ++cursor;
        }
    }
    if(!histogram_ready) return std::max(floor,high);
    const Vertex known=std::max(floor,high);
    for(Vertex q=previous-1;q>known;--q) {
        high+=histogram[q-floor-1];
        if(high>=q) return q;
    }
    return known;
}

// Compressed bounds and degree grouping


static std::uint8_t envelope_code(Vertex value) {
    if(value<=15) return static_cast<std::uint8_t>(value);
    unsigned shift=28-__builtin_clz(value);
    const std::uint64_t unit=std::uint64_t{1}<<shift;
    unsigned mantissa=static_cast<unsigned>((std::uint64_t(value)+unit-1)/unit);
    if(mantissa==16) {++shift;mantissa=8;}
    return static_cast<std::uint8_t>(shift*8+mantissa);
}
static Vertex envelope_value(unsigned code) {
    if(code<=15) return code;
    if(code>=240) return std::numeric_limits<Vertex>::max();
    return Vertex(8+code%8)<<(code/8-1);
}
// Group degrees from CSR offsets.
template<class Histogram>
static Vertex plan_degree_group16(const Edge* offsets,unsigned length,Vertex cap,
    std::uint8_t* codes,Histogram& histogram,PlanGroupStats& stats) {
    if(length==0) return 0;
    ++stats.batches;stats.vertices+=length;
    alignas(64) Vertex keys[16];
    Vertex maximum=0;
#ifdef BGHC_PLAN_GROUP_AVX512
    if(length==16) {
        const __m512i d0=_mm512_sub_epi64(
            _mm512_loadu_si512(static_cast<const void*>(offsets+1)),
            _mm512_loadu_si512(static_cast<const void*>(offsets)));
        const __m512i d1=_mm512_sub_epi64(
            _mm512_loadu_si512(static_cast<const void*>(offsets+9)),
            _mm512_loadu_si512(static_cast<const void*>(offsets+8)));
        const __m256i lo=_mm512_cvtepi64_epi32(d0),hi=_mm512_cvtepi64_epi32(d1);
        const __m512i degrees=_mm512_inserti64x4(_mm512_castsi256_si512(lo),hi,1);
        maximum=_mm512_reduce_max_epu32(degrees);
        const __m512i key_vector=_mm512_min_epu32(degrees,_mm512_set1_epi32(static_cast<int>(cap)));
        _mm512_store_si512(static_cast<void*>(keys),key_vector);
        __m512i code_vector=_mm512_setzero_si512();
        unsigned remaining=0xffffU;
        while(remaining) {
            const unsigned lane=static_cast<unsigned>(__builtin_ctz(remaining));
            const Vertex key=keys[lane];
            const unsigned group=static_cast<unsigned>(_mm512_cmpeq_epi32_mask(
                key_vector,_mm512_set1_epi32(static_cast<int>(key))))&remaining;
            histogram[key]+=static_cast<Edge>(__builtin_popcount(group));
            const auto code=envelope_code(key);
            code_vector=_mm512_mask_set1_epi32(code_vector,static_cast<__mmask16>(group),code);
            ++stats.groups;remaining&=~group;
        }
        _mm_storeu_si128(reinterpret_cast<__m128i*>(codes),_mm512_cvtepi32_epi8(code_vector));
        return maximum;
    }
#endif
    // Handle the remaining vertices.
    for(unsigned lane=0;lane<length;++lane) {
        const Vertex d=static_cast<Vertex>(offsets[lane+1]-offsets[lane]);
        maximum=std::max(maximum,d);keys[lane]=std::min(cap,d);
    }
    unsigned remaining=(1U<<length)-1;
    while(remaining) {
        const unsigned lane=static_cast<unsigned>(__builtin_ctz(remaining));
        const Vertex key=keys[lane];
        unsigned group=0;
        for(unsigned other=0;other<length;++other)
            if(keys[other]==key) group|=1U<<other;
        group&=remaining;
        histogram[key]+=static_cast<Edge>(__builtin_popcount(group));
        const auto code=envelope_code(key);
        for(unsigned pending=group;pending;pending&=pending-1)
            codes[static_cast<unsigned>(__builtin_ctz(pending))]=code;
        ++stats.groups;remaining&=~group;
    }
    return maximum;
}

// Candidate-domain indexing

struct DomainRank {
    const std::vector<std::uint64_t>& mask;
    std::unique_ptr<Vertex[]> prefix;
    explicit DomainRank(const std::vector<std::uint64_t>& bits)
        :mask(bits),prefix(new Vertex[bits.size()]) {}
    void build(int workers) {
        std::vector<Vertex> sums(workers);
#pragma omp parallel num_threads(workers)
        {
            const int tid=omp_get_thread_num(),nt=omp_get_num_threads();
            const std::size_t first=mask.size()*tid/nt,last=mask.size()*(tid+1)/nt;
            Vertex total=0;
            for(std::size_t w=first;w<last;++w) {
                prefix[w]=total;total+=__builtin_popcountll(mask[w]);
            }
            sums[tid]=total;
#pragma omp barrier
#pragma omp single
            {
                Vertex offset=0;
                for(int t=0;t<nt;++t) {
                    const Vertex count=sums[t];sums[t]=offset;offset+=count;
                }
            }
            for(std::size_t w=first;w<last;++w) prefix[w]+=sums[tid];
        }
    }
    Vertex operator[](Vertex v) const {
        const unsigned bit=v%64;
        return prefix[v/64]+__builtin_popcountll(mask[v/64]&((1ULL<<bit)-1));
    }
};

struct SparseBoundView {
    const DomainRank& rank;
    Vertex* values;
    Vertex& operator[](Vertex v) const {
        const Vertex i=rank[v];
        return values[i];
    }
};

// Failed-domain seed

static Result::Seed certify_failed_domain_seed(
    const Graph& graph,const std::vector<Vertex>& ids,
    const std::vector<std::uint64_t>& membership,const SparseBoundView& global_labels,
    const DomainRank& global_to_local,Vertex lower,Edge reference_raw_entries,int workers) {
    Result::Seed stats;
    stats.attempted=true;stats.before=stats.after=lower;
    stats.vertices=ids.size();stats.reference_raw_entries=reference_raw_entries;
    if(ids.empty()) return stats;
    Timer build;
    std::vector<std::vector<Vertex>> pools(workers),histograms(workers);
    std::vector<Edge> begin(ids.size()),end(ids.size());
    std::vector<int> owner(ids.size());
    std::vector<Vertex> labels(ids.size()),pending(ids.size());
    Edge build_entries=0;
#pragma omp parallel num_threads(workers) reduction(+:build_entries)
    {
        const int tid=omp_get_thread_num();
        auto pool=std::move(pools[tid]);
#pragma omp for schedule(dynamic,64)
        for(std::size_t i=0;i<ids.size();++i) {
            const Vertex v=ids[i];
            owner[i]=tid;begin[i]=pool.size();
            for(Edge e=graph.xadj[v];e<graph.xadj[static_cast<std::size_t>(v)+1];++e) {
                ++build_entries;
                const Vertex u=graph.adj[e];
                if((membership[u/64]>>(u%64))&1ULL) pool.push_back(global_to_local[u]);
            }
            end[i]=pool.size();
            // Refine only values above the current lower bound.
            labels[i]=std::max(lower,std::min(global_labels[v],static_cast<Vertex>(end[i]-begin[i])));
        }
        pools[tid]=std::move(pool);
    }
    stats.build_entries=build_entries;
    for(const auto& pool:pools) stats.compact_entries+=pool.size();
    stats.build_time=build.seconds();
    Timer solve;
    const AtomicLabelView current{labels.data()};
    std::vector<TopTwoStats> thread_tops(workers);
    while(stats.compact_entries) {
        ++stats.rounds;
        const Vertex floor=stats.after;
        Edge scans=0,updates=0,removed=0,tail_probes=0,writes=0,front_swaps=0;
        int top_workers=0;
#pragma omp parallel num_threads(workers) reduction(+:scans,updates,removed,tail_probes,writes,front_swaps)
        {
            const int tid=omp_get_thread_num();
            if(tid==0) top_workers=omp_get_num_threads();
            TopTwoStats top;
            auto& hist=histograms[tid];
#pragma omp for schedule(dynamic,64)
            for(std::size_t i=0;i<ids.size();++i) {
                const Vertex previous=current[static_cast<Vertex>(i)];
                if(previous<=floor) {top.add(previous);continue;}
                ++updates;
                auto& pool=pools[owner[i]];


                const Vertex next=support_lazy_h(previous,floor,current,
                    pool,begin[i],end[i],hist,scans,removed,tail_probes,writes,front_swaps);
                __atomic_store_n(labels.data()+i,next,__ATOMIC_RELAXED);
                top.add(next);
            }
            thread_tops[tid]=top;
        }
        stats.h_entries+=scans+tail_probes+front_swaps;stats.h_updates+=updates;
        TopTwoStats merged;
        for(int tid=0;tid<top_workers;++tid) merged.merge(thread_tops[tid]);
        const Vertex maximum=merged.maximum;
        Vertex second=std::max(stats.after,merged.second),witness=maximum;
        const Edge top_count=merged.count;
        if(maximum<=stats.after) break;
        Edge cert_scans=0,failures=0;
#pragma omp parallel for num_threads(workers) schedule(dynamic,64) reduction(+:cert_scans,failures) reduction(min:witness)
        for(std::size_t i=0;i<ids.size();++i) {
            pending[i]=current[static_cast<Vertex>(i)];
            if(pending[i]!=maximum) continue;
            if(top_count<=maximum) {
                witness=0;pending[i]=std::max(second,static_cast<Vertex>(top_count-1));++failures;
                continue;
            }
            Vertex support=0;
            const auto& pool=pools[owner[i]];
            for(Edge e=begin[i];e<end[i];++e) {
                ++cert_scans;
                if(current[pool[e]]==maximum && ++support==maximum) break;
            }
            witness=std::min(witness,support);
            if(support<maximum) {pending[i]=std::max(second,support);++failures;}
        }
        stats.certificate_entries+=cert_scans;
        stats.after=std::max(stats.after,witness);
        if(failures==0) break;
#pragma omp parallel for num_threads(workers) schedule(static)
        for(std::size_t i=0;i<ids.size();++i)
            __atomic_store_n(labels.data()+i,pending[i],__ATOMIC_RELAXED);
    }
    stats.solve_time=solve.seconds();
    return stats;
}

// Batched h-index over compressed bounds
template<class Codes>
static Vertex batched_envelope_h(Vertex previous,Vertex floor,Codes codes,
    const Vertex* begin,const Vertex* end,Edge& scans,Edge& batches,Edge& overscan,Edge& histogram_updates,Edge& histogram_contributions,Edge& histogram_zeroed_bins,Edge& wide_batches) {
    if(previous<=floor) return previous;
    Edge possible=static_cast<Edge>(end-begin);
    if(possible<=floor) return floor;
    Vertex frequencies[241];
    Vertex high=0;
    const unsigned top=envelope_code(previous),bottom=envelope_code(floor+1);
    std::fill(frequencies+bottom,frequencies+top,Vertex{0});
    histogram_zeroed_bins+=top-bottom;
#if defined(__SSE2__) && !defined(BGHC_SCALAR)
    const __m128i flip=_mm_set1_epi8(static_cast<char>(0x80));
    const __m128i top_compare=_mm_set1_epi8(static_cast<char>((top-1)^0x80u));
    const __m128i bottom_compare=_mm_set1_epi8(static_cast<char>((bottom-1)^0x80u));
#endif
    auto p=begin;
    while(end-p>=64) {
        alignas(16) std::uint8_t gathered[64];
#if defined(__GNUC__)
#pragma GCC unroll 64
#endif
        for(unsigned lane=0;lane<64;++lane) gathered[lane]=codes[p[lane]];
        scans+=64;batches+=4;++wide_batches;
        std::uint64_t high_mask=0,middle_mask=0;
#if defined(BGHC_RAW64_AVX512BW)

        const __m512i wide_values=_mm512_loadu_si512(static_cast<const void*>(gathered));
        high_mask=static_cast<std::uint64_t>(_mm512_cmpge_epu8_mask(
            wide_values,_mm512_set1_epi8(static_cast<char>(top))));
        middle_mask=static_cast<std::uint64_t>(_mm512_cmpge_epu8_mask(
            wide_values,_mm512_set1_epi8(static_cast<char>(bottom))))&~high_mask;
#elif defined(__SSE2__) && !defined(BGHC_SCALAR)
        __m128i byte_values[4];
        for(unsigned block=0;block<4;++block) {
            byte_values[block]=_mm_load_si128(reinterpret_cast<const __m128i*>(gathered+16*block));
            const __m128i values=_mm_xor_si128(byte_values[block],flip);
            high_mask|=static_cast<std::uint64_t>(static_cast<unsigned>(
                _mm_movemask_epi8(_mm_cmpgt_epi8(values,top_compare))))<<(16*block);
            middle_mask|=static_cast<std::uint64_t>(static_cast<unsigned>(
                _mm_movemask_epi8(_mm_cmpgt_epi8(values,bottom_compare))))<<(16*block);
        }
        middle_mask&=~high_mask;
#else
        for(unsigned lane=0;lane<64;++lane) {
            if(gathered[lane]>=top) high_mask|=std::uint64_t{1}<<lane;
            else if(gathered[lane]>=bottom) middle_mask|=std::uint64_t{1}<<lane;
        }
#endif
        // Stop when the remaining entries cannot exceed the floor.
        possible-=64-static_cast<unsigned>(__builtin_popcountll(high_mask|middle_mask));
        if(possible<=floor) return floor;
        const Vertex qualifying=static_cast<Vertex>(__builtin_popcountll(high_mask));
        if(qualifying>=previous-high) {
            unsigned needed=previous-high;
            std::uint64_t mask=high_mask;
            while(--needed) mask&=mask-1;
            const unsigned last=static_cast<unsigned>(__builtin_ctzll(mask));
            overscan+=63-last;
            return previous;
        }
        high+=qualifying;
        histogram_contributions+=static_cast<unsigned>(__builtin_popcountll(middle_mask));
        while(middle_mask) {
            const unsigned lane=static_cast<unsigned>(__builtin_ctzll(middle_mask));
            const unsigned code=gathered[lane];
            std::uint64_t group_mask=0;
#if defined(BGHC_RAW64_AVX512BW)
            group_mask=static_cast<std::uint64_t>(_mm512_cmpeq_epi8_mask(
                wide_values,_mm512_set1_epi8(static_cast<char>(code))))&middle_mask;
#elif defined(__SSE2__) && !defined(BGHC_SCALAR)
            const __m128i equal_code=_mm_set1_epi8(static_cast<char>(code));
            for(unsigned block=0;block<4;++block)
                group_mask|=static_cast<std::uint64_t>(static_cast<unsigned>(
                    _mm_movemask_epi8(_mm_cmpeq_epi8(byte_values[block],equal_code))))<<(16*block);
            group_mask&=middle_mask;
#else
            for(unsigned j=0;j<64;++j)
                if(gathered[j]==code) group_mask|=std::uint64_t{1}<<j;
            group_mask&=middle_mask;
#endif
            frequencies[code]+=static_cast<Vertex>(__builtin_popcountll(group_mask));
            ++histogram_updates;
            middle_mask&=~group_mask;
        }
        p+=64;
    }
    while(end-p>=16) {
        alignas(16) std::uint8_t gathered[16];

#if defined(__GNUC__)
#pragma GCC unroll 16
#endif
        for(unsigned lane=0;lane<16;++lane) gathered[lane]=codes[p[lane]];
        scans+=16;++batches;
        unsigned high_mask=0,middle_mask=0;
#if defined(__SSE2__) && !defined(BGHC_SCALAR)
        const __m128i byte_values=_mm_load_si128(reinterpret_cast<const __m128i*>(gathered));
        const __m128i values=_mm_xor_si128(byte_values,flip);
        high_mask=static_cast<unsigned>(_mm_movemask_epi8(_mm_cmpgt_epi8(values,top_compare)));
        middle_mask=static_cast<unsigned>(_mm_movemask_epi8(_mm_cmpgt_epi8(values,bottom_compare)))&~high_mask;
#else
        for(unsigned lane=0;lane<16;++lane) {
            if(gathered[lane]>=top) high_mask|=1U<<lane;
            else if(gathered[lane]>=bottom) middle_mask|=1U<<lane;
        }
#endif
        // Stop when the remaining entries cannot exceed the floor.
        possible-=16-static_cast<unsigned>(__builtin_popcount(high_mask|middle_mask));
        if(possible<=floor) return floor;
        const Vertex qualifying=static_cast<Vertex>(__builtin_popcount(high_mask));
        if(qualifying>=previous-high) {
            unsigned needed=previous-high,mask=high_mask;
            while(--needed) mask&=mask-1;
            const unsigned last=static_cast<unsigned>(__builtin_ctz(mask));
            overscan+=15-last;
            return previous;
        }
        high+=qualifying;
        histogram_contributions+=static_cast<unsigned>(__builtin_popcount(middle_mask));
        while(middle_mask) {
            const unsigned lane=static_cast<unsigned>(__builtin_ctz(middle_mask));
            const unsigned code=gathered[lane];
            unsigned group_mask=0;
#if defined(__SSE2__) && !defined(BGHC_SCALAR)
            group_mask=static_cast<unsigned>(_mm_movemask_epi8(_mm_cmpeq_epi8(
                byte_values,_mm_set1_epi8(static_cast<char>(code)))))&middle_mask;
#else
            for(unsigned j=0;j<16;++j)
                if(gathered[j]==code) group_mask|=1U<<j;
            group_mask&=middle_mask;
#endif
            frequencies[code]+=static_cast<Vertex>(__builtin_popcount(group_mask));
            ++histogram_updates;
            middle_mask&=~group_mask;
        }
        p+=16;
    }
    for(;p!=end;++p) {
        ++scans;
        const unsigned code=codes[*p];
        if(code>=top) {
            if(++high==previous) return previous;
        } else if(code>=bottom) {++frequencies[code];++histogram_updates;++histogram_contributions;}
        else if(--possible<=floor) return floor;
    }
    Vertex answer=std::max(floor,high),total=high;
    for(unsigned code=top-1;code>0;--code) {
        const Vertex value=envelope_value(code);
        if(value<=answer) break;
        total+=frequencies[code];
        answer=std::max(answer,std::min(value,total));
    }
    return answer;
}


static std::uint64_t envelope_domain_word(const std::uint8_t* codes,
                                         std::size_t size,unsigned cut) {
    std::uint64_t mask=0;
#if defined(__SSE2__) && !defined(BGHC_SCALAR)
    if(size==64) {
        const auto bias=_mm_set1_epi8(static_cast<char>(0x80));
        const auto bound=_mm_set1_epi8(static_cast<char>((cut^128)-1));
        if(cut==0) return ~std::uint64_t{0};
        for(unsigned i=0;i<64;i+=16) {
            auto v=_mm_loadu_si128(reinterpret_cast<const __m128i*>(codes+i));
            v=_mm_xor_si128(v,bias);
            mask|=static_cast<std::uint64_t>(static_cast<unsigned>(
                _mm_movemask_epi8(_mm_cmpgt_epi8(v,bound))))<<i;
        }
        return mask;
    }
#endif
    for(std::size_t i=0;i<size;++i) if(codes[i]>=cut) mask|=1ULL<<i;
    return mask;
}


struct CompactRows {
    std::unique_ptr<Vertex[]> labels;
    std::vector<std::vector<Vertex>> pools;
    std::vector<Edge> begin, end;
    std::vector<int> owner;

    CompactRows(std::size_t size, int workers)
        : labels(new Vertex[size]), pools(workers), begin(size), end(size), owner(size) {}
};

static CompactRows build_compact_rows(
    const Graph& graph, const std::vector<Vertex>& ids, const SparseBoundView& labels,
    const std::vector<std::uint64_t>& compact_membership, const DomainRank& global_to_local,
    Vertex floor, int workers, Stage& stage) {
    CompactRows rows(ids.size(), workers);
    Vertex* local_labels=rows.labels.get();

    auto& pools=rows.pools;
    auto& begin=rows.begin;
    auto& end=rows.end;
    auto& owner=rows.owner;
    Edge build_scans=0;
#pragma omp parallel num_threads(workers) reduction(+:build_scans)
    {
        const int tid=omp_get_thread_num();
        auto pool=std::move(pools[tid]);
#pragma omp for schedule(dynamic,64)
        for(std::size_t i=0;i<ids.size();++i) {
            const Vertex v=ids[i];
            owner[i]=tid;
            begin[i]=pool.size();
            for(Edge e=graph.xadj[v];e<graph.xadj[static_cast<std::size_t>(v)+1];++e) {
                ++build_scans;
                Vertex u=graph.adj[e];
                if((compact_membership[u/64]>>(u%64))&1ULL)
                    pool.push_back(global_to_local[u]);
            }
            end[i]=pool.size();
            __atomic_store_n(local_labels+i,std::max(floor,std::min(
                labels[v],static_cast<Vertex>(end[i]-begin[i]))),__ATOMIC_RELAXED);
        }
        pools[tid]=std::move(pool);
    }
    for(const auto& p:pools) stage.compact_entries+=p.size();
    stage.remapped_entries=stage.compact_entries;
    stage.build_entries=build_scans;
    return rows;
}


static std::pair<bool, Vertex> refine_compact_rows(
    CompactRows& rows, Vertex threshold, Vertex& lower, int workers,
    std::vector<std::vector<Vertex>>& histograms, std::vector<Vertex>& pending,
    Stage& stage) {
    Vertex* local_labels=rows.labels.get();
    const AtomicLabelView current{local_labels};
    auto& pools=rows.pools;
    auto& begin=rows.begin;
    auto& end=rows.end;
    auto& owner=rows.owner;
    Timer local;
    std::vector<TopTwoStats> thread_tops(workers);
    bool success=false;
    Vertex maximum=0;
    for(;;) {
        ++stage.rounds;
        const Vertex active_threshold=std::max(threshold,lower);
        const Vertex active_floor=active_threshold-1;
        stage.effective_threshold=active_threshold;
        Edge scans=0,updates=0,descents=0,removed=0,tail_probes=0,writes=0,front_swaps=0;
        int top_workers=0;
#pragma omp parallel num_threads(workers) reduction(+:scans,updates,descents,removed,tail_probes,writes,front_swaps)
        {
            const int tid=omp_get_thread_num();
            if(tid==0) top_workers=omp_get_num_threads();
            TopTwoStats top;
            auto& hist=histograms[tid];
#pragma omp for schedule(dynamic,64)
            for(std::size_t i=0;i<rows.owner.size();++i) {
                const Vertex previous=current[static_cast<Vertex>(i)];
                if(previous<active_threshold) {top.add(previous);continue;}
                ++updates;
                auto& p=pools[owner[i]];
                const Vertex next=support_lazy_h(previous,active_floor,current,
                    p,begin[i],end[i],hist,scans,removed,tail_probes,writes,front_swaps);
                if(next<previous) {
                    __atomic_store_n(local_labels+i,next,__ATOMIC_RELAXED);
                    ++descents;
                }
                top.add(next);
            }
            thread_tops[tid]=top;
        }
        stage.h_entries+=scans+tail_probes+front_swaps;
        stage.h_updates+=updates;
        stage.local_h_updates+=updates;
        stage.local_h_entries+=scans+tail_probes+front_swaps;
        stage.local_label_reads+=scans;
        stage.lazy_removed_entries+=removed;
        stage.lazy_tail_probes+=tail_probes;
        stage.lazy_row_writes+=writes;
        stage.support_front_swaps+=front_swaps;
        stage.support_front_probes+=front_swaps;
        stage.support_front_writes+=2*front_swaps;
        stage.async_h_descents+=descents;
        stage.active_row_checks+=rows.owner.size();
        TopTwoStats merged;
        for(int tid=0;tid<top_workers;++tid) merged.merge(thread_tops[tid]);
        maximum=merged.maximum;
        Vertex second=merged.second;
        const std::size_t top_count=static_cast<std::size_t>(merged.count);
        if(maximum<active_threshold) break;
        second=std::max(second,active_floor);
        Edge cert_scans=0;
        int failures=0;
        Vertex witness_floor=maximum;
#pragma omp parallel for num_threads(workers) schedule(dynamic,64) reduction(+:cert_scans,failures) reduction(min:witness_floor)
        for(std::size_t i=0;i<rows.owner.size();++i) {
            pending[i]=current[static_cast<Vertex>(i)];
            if(pending[i]!=maximum) continue;
            if(top_count<=maximum) {
                witness_floor=0;
                pending[i]=std::max(second,static_cast<Vertex>(top_count-1));
                ++failures;
                continue;
            }
            Vertex support=0;
            const auto& p=pools[owner[i]];
            for(Edge e=begin[i];e<end[i];++e) {
                ++cert_scans;
                if(current[p[e]]==maximum && ++support>=maximum) break;
            }
            witness_floor=std::min(witness_floor,support);
            if(support<maximum) {
                pending[i]=std::max(second,support);
                ++failures;
            }
        }
        stage.certificate_entries+=cert_scans;
        lower=std::max(lower,witness_floor);
        if(failures==0) {success=true;break;}
        Edge cap_descents=0;
#pragma omp parallel for num_threads(workers) schedule(static) reduction(+:cap_descents)
        for(std::size_t i=0;i<rows.owner.size();++i) {
            cap_descents+=pending[i]<current[static_cast<Vertex>(i)];
            __atomic_store_n(local_labels+i,pending[i],__ATOMIC_RELAXED);
        }
        stage.top_cap_descents+=cap_descents;
    }
    stage.lazy_remaining_entries=stage.compact_entries-stage.lazy_removed_entries;
    stage.solve_time=local.seconds();
    return {success,maximum};
}

// Global threshold search
static Result solve_bghc(const Graph& graph, int workers) {
    Result result;
    Timer plan;
    if (graph.n == 0) return result;
    const Vertex cap = clique_edge_upper(graph.n, graph.m);
    const std::size_t block_count=(static_cast<std::size_t>(graph.n)+63)/64;
    std::unique_ptr<Vertex[]> block_max(new Vertex[block_count]);

    std::unique_ptr<std::uint8_t[]> envelope(new std::uint8_t[graph.n]);
    const std::size_t width = static_cast<std::size_t>(cap)+1;
    std::vector<std::vector<Edge>> counts(workers, std::vector<Edge>(width));
    PlanGroupStats plan_groups;
#pragma omp parallel num_threads(workers) reduction(plan_group_merge:plan_groups)
    {
        const int tid = omp_get_thread_num();
#pragma omp for schedule(static)
        for (std::size_t word=0; word<block_count; ++word) {
            Vertex maximum=0;
            const auto end=std::min<std::size_t>(graph.n,(word+1)*64);
            for(std::size_t id=word*64;id<end;id+=16) {
                const unsigned length=static_cast<unsigned>(std::min<std::size_t>(16,end-id));
                maximum=std::max(maximum,plan_degree_group16(graph.xadj.data()+id,length,cap,
                    envelope.get()+id,counts[tid],plan_groups));
            }
            block_max[word]=maximum;
        }
    }
    for (int t=1;t<workers;++t)
        for(std::size_t q=0;q<width;++q) counts[0][q]+=counts[t][q];
    Vertex lower=edge_count_degeneracy_lower(graph.n,graph.m), upper=0;
    Edge tail=0, volume=0, below=0;
    for(Vertex q=0;q<cap;++q) below+=counts[0][q]*q;
    const Edge overflow=2*graph.m-below;
    for(Vertex q=cap;q>0;--q) {
        tail+=counts[0][q];
        volume+= q==cap ? overflow : counts[0][q]*q;
        if(upper==0 && tail>=static_cast<Edge>(q)+1) upper=q;
        if(volume>graph.m) {
            const Vertex cut=edge_count_degeneracy_lower(
                static_cast<Vertex>(tail),volume-graph.m);
            lower=std::max(lower,cut);
        }
    }
    if(lower>upper) throw std::logic_error("invalid certified interval");
    result.lower=lower;
    result.best_lower=lower;
    result.initial_upper=upper;
    result.plan_groups=plan_groups;
    result.plan_time=plan.seconds();
    if(upper==0) {
        result.core.resize(graph.n);
        std::iota(result.core.begin(),result.core.end(),Vertex{0});
        return result;
    }
    std::vector<std::vector<Vertex>> bins(workers), bound_bins(workers), histograms(workers);
    // Keep raw and compact membership separate.
    std::vector<std::uint64_t> membership(block_count,0);
    DomainRank raw_rank(membership);
    std::vector<std::uint64_t> compact_membership(block_count,0);
    DomainRank global_to_local(compact_membership);

    Vertex threshold=lower+(upper-lower+1)/2;
    for(;;) {
        Stage stage;
        stage.threshold=threshold;
        Timer domain;
        Edge skipped_blocks=0,raw_volume=0;
        const unsigned domain_code=envelope_code(threshold);
        for(auto& b:bins) b.clear();
        for(auto& b:bound_bins) b.clear();
#pragma omp parallel num_threads(workers) reduction(+:skipped_blocks,raw_volume)
        {
            const int tid=omp_get_thread_num();
            auto local_ids=std::move(bins[tid]);
            auto local_bounds=std::move(bound_bins[tid]);
#pragma omp for schedule(static)
            for(std::size_t word=0;word<membership.size();++word) {
                if(block_max[word]<threshold) {
                    membership[word]=0;
                    ++skipped_blocks;
                    continue;
                }
                std::uint64_t bits=0;
                const std::size_t end=std::min<std::size_t>(graph.n,(word+1)*64);
                std::uint64_t eligible=envelope_domain_word(envelope.get()+word*64,end-word*64,domain_code);
                while(eligible) {
                    const std::size_t id=word*64+__builtin_ctzll(eligible);
                    eligible&=eligible-1;
                    const Vertex v=static_cast<Vertex>(id);
                    const Vertex d=graph_degree(graph,v);
                    if(d<threshold) continue;
                    // Combine valid coreness upper bounds.
                    const Vertex previous=std::min({d,upper,envelope_value(envelope[v])});
                    bits|=1ULL<<(v%64);
                    local_ids.push_back(v);
                    local_bounds.push_back(previous);
                    raw_volume+=d;
                }
                membership[word]=bits;
            }
            bins[tid]=std::move(local_ids);
            bound_bins[tid]=std::move(local_bounds);
        }
        std::vector<Vertex> ids;
        for(const auto& b:bins) ids.insert(ids.end(),b.begin(),b.end());
        stage.candidates=ids.size();
        raw_rank.build(workers);
        std::unique_ptr<Vertex[]> raw_storage(new Vertex[ids.size()]);
        std::vector<std::size_t> bound_offsets(workers+1,0);
        for(int t=0;t<workers;++t) {
            bound_offsets[t+1]=bound_offsets[t]+bound_bins[t].size();

        }
#pragma omp parallel for num_threads(workers) schedule(static)
        for(int t=0;t<workers;++t)
            std::copy(bound_bins[t].begin(),bound_bins[t].end(),raw_storage.get()+bound_offsets[t]);
        const SparseBoundView labels{raw_rank,raw_storage.get()};

        stage.raw_entries=raw_volume;
        stage.skipped_vertex_blocks=skipped_blocks;
        stage.domain_time=domain.seconds();
        const Vertex floor=threshold-1;
        if(ids.size()<=threshold) {
            upper=floor;
            for(Vertex v:ids) {
                labels[v]=std::min(labels[v],upper);
                envelope[v]=envelope_code(labels[v]);
            }
            stage.upper_after=upper;
            result.stages.push_back(stage);
            if(upper<lower) throw std::logic_error("cardinality contradicts lower");
            threshold=lower+(upper-lower+1)/2;
            continue;
        }
        Timer raw;
        std::vector<Vertex> pending(ids.size());
        Edge raw_scans=0,raw_batches=0,raw_overscan=0,raw_histogram_updates=0,raw_wide_batches=0;
        Edge raw_histogram_contributions=0,raw_histogram_zeroed_bins=0;
#pragma omp parallel num_threads(workers) reduction(+:raw_scans,raw_batches,raw_overscan,raw_histogram_updates,raw_histogram_contributions,raw_histogram_zeroed_bins,raw_wide_batches)
        {
#pragma omp for schedule(dynamic,64)
            for(std::size_t i=0;i<ids.size();++i) {
                const Vertex v=ids[i];
                pending[i]=batched_envelope_h(
                    std::min(raw_storage[i],upper),lower ? lower-1 : 0,
                    static_cast<const std::uint8_t*>(envelope.get()),
                    graph.adj.data()+graph.xadj[v],
                    graph.adj.data()+graph.xadj[static_cast<std::size_t>(v)+1],
                    raw_scans,raw_batches,raw_overscan,raw_histogram_updates,raw_histogram_contributions,raw_histogram_zeroed_bins,raw_wide_batches);
            }
        }
        stage.raw_sync_published_rows=ids.size();
        stage.raw_wide_batches=raw_wide_batches;stage.raw_batches=raw_batches;
        stage.raw_batch_overscan=raw_overscan;stage.raw_histogram_updates=raw_histogram_updates;
        stage.raw_histogram_contributions=raw_histogram_contributions;
        stage.raw_histogram_zeroed_bins=raw_histogram_zeroed_bins;
        stage.h_updates+=ids.size();
        stage.h_entries+=raw_scans;
        Vertex observed_max=0;
        std::size_t raw_survivors=0;
#pragma omp parallel for num_threads(workers) schedule(static) reduction(max:observed_max) reduction(+:raw_survivors)
        for(std::size_t i=0;i<ids.size();++i) {
            raw_storage[i]=pending[i];
            envelope[ids[i]]=envelope_code(pending[i]);
            observed_max=std::max(observed_max,pending[i]);
            raw_survivors+=pending[i]>=threshold;
        }
        std::vector<Vertex> seed_ids;
        if(!result.seed.attempted && raw_survivors<=threshold) seed_ids=ids;
        ids.erase(std::remove_if(ids.begin(),ids.end(),[&](Vertex v){
            return labels[v]<threshold;
        }),ids.end());
        stage.survivors=ids.size();
        stage.raw_time=raw.seconds();
        if(ids.size()<=threshold) {
            upper=floor;
            for(Vertex v:ids) {
                labels[v]=std::min(labels[v],upper);
                envelope[v]=envelope_code(labels[v]);
            }
            bool positive_seed_query=false;
            if(!result.seed.attempted) {
                Timer seed_timer;
                result.seed=certify_failed_domain_seed(graph,seed_ids,membership,labels,
                    raw_rank,lower,raw_scans,workers);
                lower=result.seed.after;result.best_lower=lower;
                positive_seed_query=result.seed.after>result.seed.before;
                result.seed.positive_query=positive_seed_query;
                result.seed.total_time=seed_timer.seconds();
            }
            stage.upper_after=upper;
            result.stages.push_back(stage);
            if(upper<lower) throw std::logic_error("h-index contradicts lower");
            const Vertex midpoint=lower+(upper-lower+1)/2;
            threshold=positive_seed_query ? lower : std::max(lower,std::min(midpoint,observed_max));
            continue;
        }
        const Vertex histogram_upper=std::min(upper,static_cast<Vertex>(ids.size()-1));
        int histogram_workers=0;
#pragma omp parallel num_threads(workers)
        {
            const int tid=omp_get_thread_num();
#pragma omp single nowait
            histogram_workers=omp_get_num_threads();
            std::fill(counts[tid].begin(),counts[tid].begin()+histogram_upper+1,Edge{0});
#pragma omp for schedule(static)
            for(std::size_t i=0;i<ids.size();++i)
                ++counts[tid][std::min(labels[ids[i]],histogram_upper)];
        }
        Edge prebuild_tail=0;
        Vertex cardinality_upper=floor;
        for(Vertex q=histogram_upper;q>=threshold;--q) {
            for(int t=0;t<histogram_workers;++t) prebuild_tail+=counts[t][q];
            if(prebuild_tail>=static_cast<Edge>(q)+1) {cardinality_upper=q;break;}
        }
        if(cardinality_upper<threshold) throw std::logic_error("prebuild cardinality lost survivors");
        upper=cardinality_upper;
        Edge label_caps=0,top_rows=0;
#pragma omp parallel for num_threads(workers) schedule(static) reduction(+:label_caps,top_rows)
        for(std::size_t i=0;i<ids.size();++i) {
            const Vertex v=ids[i];
            label_caps+=labels[v]>upper;
            labels[v]=std::min(labels[v],upper);
            top_rows+=labels[v]==upper;
        }
        stage.prebuild_upper=upper;

        stage.prebuild_label_caps=label_caps;
        stage.prebuild_top_rows=top_rows;

        Edge prebuild_scans=0,prebuild_failures=0;
        Vertex prebuild_witness=upper;
#pragma omp parallel for num_threads(workers) schedule(dynamic,64) reduction(+:prebuild_scans,prebuild_failures) reduction(min:prebuild_witness)
        for(std::size_t i=0;i<ids.size();++i) {
            const Vertex v=ids[i];
            if(labels[v]!=upper) continue;
            Vertex support=0;
            for(Edge e=graph.xadj[v];e<graph.xadj[static_cast<std::size_t>(v)+1];++e) {
                ++prebuild_scans;
                const Vertex u=graph.adj[e];
                if(((membership[u/64]>>(u%64))&1ULL) && labels[u]==upper && ++support==upper) break;
            }
            prebuild_witness=std::min(prebuild_witness,support);
            prebuild_failures+=support<upper;
        }
        lower=std::max(lower,prebuild_witness);
        result.best_lower=lower;
        stage.prebuild_entries=prebuild_scans;
        stage.certificate_entries+=prebuild_scans;
        stage.prebuild_success=prebuild_failures==0;

        if(stage.prebuild_success) {
            result.k=upper;
            for(Vertex v:ids) if(labels[v]==upper) result.core.push_back(v);
            std::sort(result.core.begin(),result.core.end());
            stage.upper_after=upper;
            result.stages.push_back(stage);
            return result;
        }
        Timer build;
#pragma omp parallel for num_threads(workers) schedule(static)
        for(std::size_t w=0;w<membership.size();++w) {
            std::uint64_t bits=membership[w],kept=0;
            while(bits) {
                const unsigned bit=__builtin_ctzll(bits);bits&=bits-1;
                if(labels[w*64+bit]>=threshold) kept|=1ULL<<bit;
            }
            compact_membership[w]=kept;
        }
        global_to_local.build(workers);

        auto rows=build_compact_rows(graph,ids,labels,compact_membership,
                                     global_to_local,floor,workers,stage);
        stage.build_time=build.seconds();
        const AtomicLabelView current{rows.labels.get()};
        const auto [success,maximum]=refine_compact_rows(
            rows,threshold,lower,workers,histograms,pending,stage);
        result.best_lower=lower;
        if(success) {
            result.k=maximum;
            for(std::size_t i=0;i<ids.size();++i)
                if(current[static_cast<Vertex>(i)]==maximum) result.core.push_back(ids[i]);
            std::sort(result.core.begin(),result.core.end());
            stage.upper_after=maximum;
            result.stages.push_back(stage);
            return result;
        }
        Timer scatter;
#pragma omp parallel for num_threads(workers) schedule(static)
        for(std::size_t i=0;i<ids.size();++i) {
            envelope[ids[i]]=envelope_code(current[static_cast<Vertex>(i)]);
        }
        stage.scatter_time=scatter.seconds();
        upper=floor;
        stage.upper_after=upper;
        result.stages.push_back(stage);
        if(upper<lower) throw std::logic_error("local solver contradicts lower");
        threshold=lower+(upper-lower+1)/2;
    }
}
// Command-line interface 
#ifndef BGHC_NO_MAIN
struct Options {
    const char* graph_path = nullptr;
    const char* core_path = nullptr;
    int workers = 1;
};

static void print_usage(const char* program) {
    std::cout << "Usage: " << program << " GRAPH [THREADS=1] [--save-core FILE.txt]\n"
              << "Prints IO time, Algorithm time, and K*.\n"
              << "--save-core FILE.txt  Save C* as sorted vertex IDs, one per line.\n"
              << "                      Disabled by default; IDs are normalized to zero-based.\n";
}

static Options parse_options(int argc, char** argv) {
    Options options;
    options.graph_path = argv[1];
    bool have_threads = false;
    for (int i = 2; i < argc; ++i) {
        const std::string_view argument(argv[i]);
        if (argument == "--save-core") {
            if (options.core_path || i + 1 == argc) {
                throw std::runtime_error("--save-core requires one output path and may appear only once");
            }
            options.core_path = argv[++i];
            if (std::string_view(options.core_path).empty() ||
                std::string_view(options.core_path).substr(0, 2) == "--") {
                throw std::runtime_error("--save-core requires an output path");
            }
        } else {
            int value = 0;
            const auto parsed = std::from_chars(argument.data(), argument.data() + argument.size(), value);
            if (have_threads || parsed.ec != std::errc{} ||
                parsed.ptr != argument.data() + argument.size() || value < 1) {
                throw std::runtime_error("expected a positive thread count or --save-core FILE.txt");
            }
            options.workers = value;
            have_threads = true;
        }
    }
    if (options.core_path) {
        std::error_code error;
        if (std::filesystem::equivalent(options.graph_path, options.core_path, error)) {
            throw std::runtime_error("core output must differ from the input graph");
        }
    }
    return options;
}

static void save_core(const char* path, const std::vector<Vertex>& core) {
    std::ofstream output(path);
    if (!output) throw std::runtime_error("cannot open core output file");
    for (Vertex vertex : core) output << vertex << '\n';
    output.close();
    if (!output) throw std::runtime_error("failed to write core output file");
}

int main(int argc, char** argv) {
    try {
        if (argc < 2 || std::string_view(argv[1]) == "--help") {
            print_usage(argv[0]);
            return argc < 2 ? 2 : 0;
        }
        const Options options = parse_options(argc, argv);
        omp_set_dynamic(0);
        omp_set_num_threads(options.workers);

        const Timer io;
        const Graph graph = read_graph(options.graph_path);
        const double io_time = io.seconds();

        const Timer algorithm;
        const Result result = solve_bghc(graph, options.workers);
        const double algorithm_time = algorithm.seconds();

        // Export after timing.
        if (options.core_path) save_core(options.core_path, result.core);
        std::cout << std::fixed << std::setprecision(9)
                  << "IO time: " << io_time << '\n'
                  << "Algorithm time: " << algorithm_time << '\n'
                  << "K*: " << result.k << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
#endif
