use tonic::transport::Channel;
use clap::{Parser, Subcommand};

pub mod vector_db {
    tonic::include_proto!("vector_db");
}

use vector_db::vector_db_client::VectorDbClient;
use vector_db::{PutRequest, SearchRequest};

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[arg(long, default_value = "http://[::1]:50050")]
    url: String,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Put {
        #[arg(long)]
        id: u32,
        #[arg(long, value_delimiter = ',')]
        vector: Vec<f32>,
    },
    Search {
        #[arg(long, value_delimiter = ',')]
        vector: Vec<f32>,
        #[arg(long, default_value_t = 5)]
        k: u32,
    },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    let mut client = VectorDbClient::connect(cli.url).await?;

    match &cli.command {
        Commands::Put { id, vector } => {
            let request = tonic::Request::new(PutRequest {
                id: *id,
                vector: vector.clone(),
            });

            let response = client.put(request).await?;
            println!("Put response: {:?}", response.into_inner());
        }
        Commands::Search { vector, k } => {
            let request = tonic::Request::new(SearchRequest {
                vector: vector.clone(),
                k: *k,
            });

            let response = client.search(request).await?;
            println!("Search response: {:?}", response.into_inner());
        }
    }

    Ok(())
}
