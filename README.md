# Bound-Guided Parallel Densest Subgraph Discovery
This repository contains C++ implementations and small example graphs for the paper:
> Bound-Guided Parallel Densest Subgraph Discovery

## Introduction
We propose BGHC and BGWC, two bound-guided parallel algorithms with 2-approximation guarantees for undirected and directed graphs.

BGHC restricts h-index refinement to candidate subgraphs while preserving valid upper bounds on core numbers in the original graph,
enabling safe filtering and bound reuse across queries. 
Internal-degree checks among vertices attaining the maximum h value further tighten these bounds or certify the complete highest-order core without global convergence. 

BGWC combines degree-guided edge filtering with density-guided incremental core search. The current solution density guides query selection and early termination,
while nested queries reuse peeling states and deletion-derived bounds prune subsequent parameter branches.

## Environment
The codes are implemented and tested under the following development environment:
- Hardware: Two Intel Xeon Gold 5218R processors (2.10 GHz base frequency), 40 physical cores, and 255 GB of memory.
- Operating system: Ubuntu 20.04.5 LTS (GNU/Linux 5.8.0-63-generic x86_64)

## Datasets

The paper evaluates six undirected and six directed real-world graphs. The counts below describe the **processed inputs used in the paper**, which may differ from the source websites' raw statistics. Undirected edges are counted once; directed arcs retain their original directions.

### Undirected graphs

| Graph | Vertices | Edges | Description | Original source |
| --- | ---: | ---: | --- | --- |
| PT | 623,766 | 15,695,166 | Petster pet network | [KONECT: petster-carnivore](http://konect.cc/networks/petster-carnivore/) |
| Hollywood | 1,139,905 | 56,375,711 | Actor collaboration | [LAW: hollywood-2009](https://law.di.unimi.it/webdata/hollywood-2009/) |
| AMiner | 92,830,928 | 323,836,571 | Coauthorship | [Benson: coauth-AMiner](https://www.cs.cornell.edu/~arb/data/coauth-AMiner/) |
| MAG | 173,195,936 | 544,577,123 | Coauthorship | [Benson: coauth-MAG](https://www.cs.cornell.edu/~arb/data/coauth-MAG/) |
| FS | 65,608,366 | 1,806,067,135 | Friendster social network | [SNAP: Friendster](https://snap.stanford.edu/data/com-Friendster.html) |
| STRING | 57,162,259 | 10,026,197,021 | Functional protein associations | [STRING v11.5 edge file](https://stringdb-downloads.org/download/protein.links.v11.5.txt.gz) |

AMiner and MAG use their unweighted topology. STRING uses the full `protein.links.v11.5` file, ignores confidence scores, removes self-loops, merges duplicate undirected edges, and retains all proteins appearing in the file. The protein identifiers include their taxonomic prefixes before conversion to integer IDs.

### Directed graphs

| Graph | Vertices | Arcs | Description | Original source |
| --- | ---: | ---: | --- | --- |
| AM | 403,394 | 3,387,388 | Product co-purchasing | [SNAP: Amazon0601](https://snap.stanford.edu/data/amazon0601.html) |
| BA | 2,140,198 | 17,632,190 | Baidu Zhishi hyperlinks | [KONECT: Baidu Zhishi](http://konect.cc/networks/zhishi-baidu-internallink/) |
| EW | 4,203,323 | 101,311,613 | English Wikipedia hyperlinks | [LAW: enwiki-2013](https://law.di.unimi.it/webdata/enwiki-2013/) |
| WE | 13,593,032 | 437,167,958 | English Wikipedia hyperlinks | [KONECT: Wikipedia-Link-EN](http://konect.cc/networks/wikipedia_link_en/) |
| SK | 50,636,059 | 1,930,292,948 | Web hyperlinks | [LAW: sk-2005](https://law.di.unimi.it/webdata/sk-2005/) |
| CC | 250,753,325 | 10,858,559,096 | Common Crawl host-level web graph | [Common Crawl: October–December 2025](https://commoncrawl.org/blog/host--and-domain-level-web-graphs-october-november-december-2025) |

## Quick start

Requirements: 64-bit Linux (including WSL) and GCC with C++17 and OpenMP support. The code uses GCC extensions, including `unsigned __int128`; no additional third-party library is required.

### Compile both programs directly from the repository root:

```bash
g++ -O3 -std=c++17 -fopenmp BGHC/BGHC.cpp -o BGHC/BGHC
g++ -O3 -std=c++17 -fopenmp BGWC/BGWC.cpp -o BGWC/BGWC
```
`PackedEdgeIds.hpp` provides compact edge-ID storage for BGWC. Keep it in the same directory as `BGWC.cpp`.

### Undirected graphs: BGHC

```bash
./BGHC/BGHC BGHC/examples/triangle.txt 32 --save-core core.txt
```

The example contains a triangle and one isolated vertex. The expected output is:

```text
IO time: <seconds>
Algorithm time: <seconds>
K*: 2
```

`core.txt` contains the complete highest-order core: vertex IDs `0`, `1`, and `2`, one per line.

General usage:

```text
./BGHC/BGHC GRAPH [THREADS=1] [--save-core FILE.txt]
```

The thread count defaults to one. `--save-core` exports sorted, zero-based vertex IDs after input normalization. It is optional and does not change the algorithm result.

### Directed graphs: BGWC

```bash
./BGWC/BGWC BGWC/examples/rectangular.txt 32 --verify-output --save result
```

The example is a complete directed bipartite graph with three source vertices, four target vertices, and twelve arcs. The expected output is:

```text
algorithm_time: <seconds>
io_time: <seconds>
w_star: 12
density: 3.464102
|S|: 3
|T|: 4
```

General usage:

```text
./BGWC/BGWC GRAPH THREADS [--verify-output] [--save PREFIX]
```

- `THREADS` is required and specifies the requested OpenMP thread count.
- `w_star` is the maximum induced number $w^*$.
- `|S|` and `|T|` report the numbers of selected source and target vertices.
- `--verify-output` checks that the returned memberships reproduce the reported source count, target count, and induced-edge count in the input graph.
- `--save PREFIX` writes `PREFIX.S.txt`, `PREFIX.T.txt`, and `PREFIX.edges.txt`, containing the selected source IDs, target IDs, and induced arcs. These files use the numeric vertex IDs supplied to the reader. The output directory must already exist.





## Contact

- Jingqiao Li: [lijingqiao@hnu.edu.cn](mailto:lijingqiao@hnu.edu.cn)
