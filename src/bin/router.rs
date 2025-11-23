use tonic::{transport::Channel, transport::Server, Request, Response, Status};
use std::collections::{BTreeMap, HashSet};
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;
use std::sync::{Arc, Mutex};
use tokio::sync::RwLock;

pub mod vector_db {
    tonic::include_proto!("vector_db");
}

use vector_db::vector_db_server::{VectorDb, VectorDbServer};
use vector_db::vector_db_client::VectorDbClient;
use vector_db::{PutRequest, PutResponse, SearchRequest, SearchResponse, SearchResult};

#[derive(Clone)]
pub struct ConsistentHashRing {
    nodes: BTreeMap<u64, String>, // Hash -> Node Address
}

impl ConsistentHashRing {
    pub fn new() -> Self {
        ConsistentHashRing {
            nodes: BTreeMap::new(),
        }
    }

    pub fn add_node(&mut self, node_addr: &str) {
        let mut hasher = DefaultHasher::new();
        node_addr.hash(&mut hasher);
        let hash = hasher.finish();
        self.nodes.insert(hash, node_addr.to_string());
        println!("Added node {} at hash {}", node_addr, hash);
    }

    // Get the primary node and its successors for replication
    pub fn get_preference_list(&self, key: u32, n: usize) -> Vec<String> {
        if self.nodes.is_empty() {
            return vec![];
        }

        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        let hash = hasher.finish();

        let mut nodes = Vec::new();
        let mut unique_nodes = HashSet::new();

        // Find the first node with a hash >= key hash
        let mut iter = self.nodes.range(hash..).chain(self.nodes.iter());

        while nodes.len() < n && unique_nodes.len() < self.nodes.len() {
            if let Some((_, addr)) = iter.next() {
                if unique_nodes.insert(addr.clone()) {
                    nodes.push(addr.clone());
                }
            } else {
                // Restart iterator from beginning if we ran out (should be handled by chain, but just in case of logic gaps)
                // actually chain(self.nodes.iter()) handles the wrap around once. 
                // If we need to wrap around multiple times (e.g. n > total nodes), we stop at total nodes.
                break;
            }
        }
        
        nodes
    }

    pub fn get_all_nodes(&self) -> Vec<String> {
        self.nodes.values().cloned().collect()
    }
}

pub struct Router {
    ring: Arc<RwLock<ConsistentHashRing>>,
    replication_factor: usize,
    write_quorum: usize,
}

impl Router {
    pub fn new(ring: ConsistentHashRing, replication_factor: usize, write_quorum: usize) -> Self {
        Router {
            ring: Arc::new(RwLock::new(ring)),
            replication_factor,
            write_quorum,
        }
    }
}

#[tonic::async_trait]
impl VectorDb for Router {
    async fn put(
        &self,
        request: Request<PutRequest>,
    ) -> Result<Response<PutResponse>, Status> {
        let req = request.into_inner();
        let id = req.id;
        
        let targets = {
            let ring = self.ring.read().await;
            ring.get_preference_list(id, self.replication_factor)
        };

        if targets.is_empty() {
            return Err(Status::unavailable("No nodes available"));
        }

        let mut successes = 0;
        let mut errors = Vec::new();

        // Simple scatter-gather for Put
        // In a real system, we might want to do this in parallel
        for target in targets {
            // Connect to node
            // TODO: Connection pooling
            // For now, we connect on every request (inefficient but simple)
            let mut client = match VectorDbClient::connect(target.clone()).await {
                Ok(c) => c,
                Err(e) => {
                    println!("Failed to connect to {}: {}", target, e);
                    errors.push(e.to_string());
                    continue;
                }
            };

            let req_clone = tonic::Request::new(PutRequest {
                id,
                vector: req.vector.clone(),
            });

            match client.put(req_clone).await {
                Ok(_) => successes += 1,
                Err(e) => {
                    println!("Failed to write to {}: {}", target, e);
                    errors.push(e.to_string());
                }
            }
        }

        if successes >= self.write_quorum {
            Ok(Response::new(PutResponse { success: true }))
        } else {
            Err(Status::internal(format!(
                "Write quorum not met. Successes: {}, Errors: {:?}",
                successes, errors
            )))
        }
    }

    async fn search(
        &self,
        request: Request<SearchRequest>,
    ) -> Result<Response<SearchResponse>, Status> {
        let req = request.into_inner();
        
        // Broadcast to all unique nodes
        // Optimization: Only query one replica set that covers the whole ring?
        // For simplicity: Query ALL nodes in the ring.
        let targets = {
            let ring = self.ring.read().await;
            let mut unique = HashSet::new();
            for node in ring.get_all_nodes() {
                unique.insert(node);
            }
            unique.into_iter().collect::<Vec<_>>()
        };

        if targets.is_empty() {
            return Err(Status::unavailable("No nodes available"));
        }

        let mut all_results = Vec::new();

        for target in targets {
             // TODO: Parallelize this!
            let mut client = match VectorDbClient::connect(target.clone()).await {
                Ok(c) => c,
                Err(e) => {
                    println!("Failed to connect to {}: {}", target, e);
                    continue;
                }
            };

            let req_clone = tonic::Request::new(SearchRequest {
                vector: req.vector.clone(),
                k: req.k,
            });

            match client.search(req_clone).await {
                Ok(resp) => {
                    all_results.extend(resp.into_inner().results);
                }
                Err(e) => {
                    println!("Failed to search on {}: {}", target, e);
                }
            }
        }

        // Merge and Sort
        // Sort by distance ascending
        all_results.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(std::cmp::Ordering::Equal));
        
        // Deduplicate by ID?
        // If we have replication, we might get the same ID from multiple nodes.
        // We should take the one with the same ID (distance should be same).
        let mut unique_results = Vec::new();
        let mut seen_ids = HashSet::new();
        for res in all_results {
            if seen_ids.insert(res.id) {
                unique_results.push(res);
            }
            if unique_results.len() >= req.k as usize {
                break;
            }
        }

        Ok(Response::new(SearchResponse {
            results: unique_results,
        }))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let addr = "[::1]:50050".parse()?;
    
    let mut ring = ConsistentHashRing::new();
    // Hardcoded backend nodes for MVP
    // Ensure these are running!
    ring.add_node("http://[::1]:50051");
    ring.add_node("http://[::1]:50052");
    ring.add_node("http://[::1]:50053");

    // Replication Factor = 2, Write Quorum = 1 (for speed/availability in this demo)
    let router = Router::new(ring, 2, 1);

    println!("Router listening on {}", addr);

    Server::builder()
        .add_service(VectorDbServer::new(router))
        .serve(addr)
        .await?;

    Ok(())
}
