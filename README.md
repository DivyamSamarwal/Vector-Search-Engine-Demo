# Distributed Vector Database (Rust)

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-blue)
![Rust](https://img.shields.io/badge/rust-1.70%2B-orange)

A high-performance, distributed vector database built from scratch in Rust. This project serves as an educational implementation of a modern vector search engine, demonstrating core concepts like durability, approximate nearest neighbor search, and distributed systems.

## 🚀 Features

- **🛡️ Durability (WAL)**: Write-Ahead Log ensures data safety. Every operation is checksummed (CRC32) and persisted to disk before being applied.
- **⚡ Fast Search (HNSW)**: Hierarchical Navigable Small World graph index for sub-linear time Approximate Nearest Neighbor (ANN) search.
- **🌐 Distributed Architecture**:
    - **Sharding**: Consistent Hashing distributes vectors across multiple nodes.
    - **Replication**: Configurable replication factor for high availability.
    - **Scatter-Gather**: Queries are broadcast to shards and results are aggregated.
- **🔌 gRPC API**: High-performance Protocol Buffers interface for all interactions.

## 🏗️ Architecture

The system consists of three main layers:

1.  **Storage Layer (WAL)**:
    - Handles sequential writes to disk.
    - Provides crash recovery by replaying the log on startup.
2.  **Index Layer (HNSW)**:
    - In-memory graph structure.
    - Supports `Euclidean`, `Cosine`, and `DotProduct` distance metrics.
    - Implements neighbor pruning to maintain graph quality (`M`, `ef_construction`).
3.  **Network Layer (gRPC)**:
    - **Router**: The entry point. Routes `Put` requests to the correct shard(s) and scatters `Search` requests.
    - **Server**: The storage node. Manages the WAL and HNSW index.

### Project Structure

```
src/
├── bin/
│   ├── server.rs    # The Storage Node (gRPC Server)
│   ├── router.rs    # The Gateway/Router (Sharding & Replication logic)
│   └── client.rs    # CLI Tool for testing
├── index/
│   ├── hnsw.rs      # Core HNSW Graph implementation
│   └── distance.rs  # Distance metrics (Euclidean, etc.)
├── wal.rs           # Write-Ahead Log implementation
└── lib.rs           # Shared library code
```

## 🛠️ Getting Started

### Prerequisites

- Rust (latest stable)
- Cargo
- Protobuf Compiler (`protoc`) - *Handled automatically by build script*

### Building

```bash
cargo build --release
```

### Running Tests

```bash
cargo test
```

## 🌍 Running a Distributed Cluster

Follow these steps to spin up a local cluster with **3 Shards** and **1 Router**.

### 1. Start Backend Servers
Open 3 separate terminals. Each server listens on a different port and maintains its own WAL file.

```bash
# Node 1
cargo run --bin server -- --port 50051

# Node 2
cargo run --bin server -- --port 50052

# Node 3
cargo run --bin server -- --port 50053
```

### 2. Start the Router
Open a 4th terminal. The router is configured to discover the 3 nodes above.

```bash
# Router (Listens on 50050)
cargo run --bin router
```

### 3. Run the Client
Open a 5th terminal to interact with the cluster.

#### Insert Vectors
The router will hash the ID and forward the data to the appropriate node(s).

```bash
# Insert ID 100
cargo run --bin client -- put --id 100 --vector 0.1,0.2,0.3

# Insert ID 101
cargo run --bin client -- put --id 101 --vector 0.9,0.8,0.7
```

#### Search Vectors
The router will query all shards and merge the top-k results.

```bash
# Find nearest neighbor to [0.1, 0.2, 0.3]
cargo run --bin client -- search --vector 0.1,0.2,0.3 --k 1
```

## 📚 API Reference

The service is defined in `proto/vector_db.proto`.

### `Put(PutRequest) returns (PutResponse)`
Inserts a vector into the database.
- **id**: Unique identifier (uint32).
- **vector**: List of floats.

### `Search(SearchRequest) returns (SearchResponse)`
Finds the `k` nearest neighbors.
- **vector**: Query vector.
- **k**: Number of results to return.

## 🗺️ Roadmap

- [x] **Write-Ahead Log (WAL)**: Append-only log with CRC32.
- [x] **HNSW Index**: Graph-based ANN search.
- [x] **gRPC Network**: Server and Client implementation.
- [x] **Sharding & Replication**: Consistent Hash Ring and Router.
- [ ] **Persistence**: Snapshotting the HNSW graph to disk.
- [ ] **Dynamic Membership**: Gossip protocol for node discovery.

## 📄 License

This project is licensed under the MIT License.
