use tonic::{transport::Server, Request, Response, Status};
use my_vector_db::index::hnsw::Hnsw;
use my_vector_db::wal::{Wal, WalEntry, OpType};
use ndarray::Array1;
use std::sync::{Arc, Mutex};
use tokio::sync::RwLock;

pub mod vector_db {
    tonic::include_proto!("vector_db");
}

use vector_db::vector_db_server::{VectorDb, VectorDbServer};
use vector_db::{PutRequest, PutResponse, SearchRequest, SearchResponse, SearchResult, SnapshotRequest, SnapshotResponse};

pub struct MyVectorDb {
    index: Arc<RwLock<Hnsw>>,
    wal: Arc<Mutex<Wal>>,
    snapshot_path: String,
}

impl MyVectorDb {
    pub fn new(index: Arc<RwLock<Hnsw>>, wal: Arc<Mutex<Wal>>, snapshot_path: String) -> Self {
        MyVectorDb { index, wal, snapshot_path }
    }
}

#[tonic::async_trait]
impl VectorDb for MyVectorDb {
    async fn put(
        &self,
        request: Request<PutRequest>,
    ) -> Result<Response<PutResponse>, Status> {
        let req = request.into_inner();
        let id = req.id;
        let vector_data = req.vector;

        // Validate vector
        if vector_data.is_empty() {
            return Err(Status::invalid_argument("Vector cannot be empty"));
        }

        let vector = Array1::from(vector_data.clone());

        // 1. Write to WAL
        let entry = WalEntry {
            op: OpType::Insert,
            vector_id: id,
            vector: vector_data,
        };

        {
            let wal = self.wal.lock().unwrap();
            if let Err(e) = wal.append(&entry) {
                return Err(Status::internal(format!("Failed to write to WAL: {}", e)));
            }
        }

        // 2. Update Index
        {
            let mut index = self.index.write().await;
            index.insert(id, vector);
        }

        Ok(Response::new(PutResponse { success: true }))
    }

    async fn search(
        &self,
        request: Request<SearchRequest>,
    ) -> Result<Response<SearchResponse>, Status> {
        let req = request.into_inner();
        let vector_data = req.vector;
        let k = req.k as usize;

        if vector_data.is_empty() {
            return Err(Status::invalid_argument("Vector cannot be empty"));
        }

        let vector = Array1::from(vector_data);

        let results = {
            let index = self.index.read().await;
            index.search(&vector.view(), k)
        };

        let search_results = results
            .into_iter()
            .map(|(id, dist)| SearchResult {
                id,
                distance: dist,
            })
            .collect();

        Ok(Response::new(SearchResponse {
            results: search_results,
        }))
    }

    async fn snapshot(
        &self,
        _request: Request<SnapshotRequest>,
    ) -> Result<Response<SnapshotResponse>, Status> {
        let index = self.index.read().await;
        if let Err(e) = index.save_snapshot(&self.snapshot_path) {
            return Err(Status::internal(format!("Failed to save snapshot: {}", e)));
        }
        println!("Snapshot saved to {}", self.snapshot_path);
        Ok(Response::new(SnapshotResponse { success: true }))
    }
}

use clap::Parser;

#[derive(Parser)]
struct Args {
    #[arg(long, default_value_t = 50051)]
    port: u16,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let addr = format!("[::1]:{}", args.port).parse()?;
    
    // Initialize components
    let snapshot_path = format!("vectors_{}.snap", args.port);
    let wal_path = format!("vectors_{}.wal", args.port);

    let hnsw = if std::path::Path::new(&snapshot_path).exists() {
        println!("Loading snapshot from {}", snapshot_path);
        Hnsw::load_snapshot(&snapshot_path)?
    } else {
        println!("Creating new HNSW index");
        // M=16, ef_construction=100
        Hnsw::new(16, 100)
    };
    
    let hnsw = Arc::new(RwLock::new(hnsw));
    let wal = Arc::new(Mutex::new(Wal::new(&wal_path)?));

    // TODO: Replay WAL here. 
    // Ideally we should only replay entries AFTER the snapshot.
    // For now, we are just loading the snapshot. 
    // If we crash between snapshot and now, we might lose data unless we replay WAL.
    // Implementing full WAL replay with snapshot offset is complex for this step.
    // We will assume manual snapshotting for now.

    let service = MyVectorDb::new(hnsw, wal, snapshot_path);

    println!("Vector DB Server listening on {}", addr);

    Server::builder()
        .add_service(VectorDbServer::new(service))
        .serve(addr)
        .await?;

    Ok(())
}
