use std::fs::{File, OpenOptions};
use std::io::{self, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;
use serde::{Deserialize, Serialize};
use crc32fast::Hasher;
use std::sync::{Arc, Mutex};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum OpType {
    Insert,
    Delete,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct WalEntry {
    pub op: OpType,
    pub vector_id: u32,
    pub vector: Vec<f32>,
}

pub struct Wal {
    file: Arc<Mutex<BufWriter<File>>>,
    path: String,
}

impl Wal {
    pub fn new(path: &str) -> io::Result<Self> {
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        
        Ok(Wal {
            file: Arc::new(Mutex::new(BufWriter::new(file))),
            path: path.to_string(),
        })
    }

    pub fn append(&self, entry: &WalEntry) -> io::Result<()> {
        let mut file = self.file.lock().unwrap();
        
        // Serialize entry
        let data = bincode::serialize(entry).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
        
        // Calculate CRC32
        let mut hasher = Hasher::new();
        hasher.update(&data);
        let checksum = hasher.finalize();

        // Write format: [CRC32 (4 bytes)] [Length (8 bytes)] [Data]
        file.write_all(&checksum.to_le_bytes())?;
        file.write_all(&(data.len() as u64).to_le_bytes())?;
        file.write_all(&data)?;
        
        // Ensure it hits the disk
        file.flush()?;
        file.get_ref().sync_all()?;
        
        Ok(())
    }

    pub fn read_all(&self) -> io::Result<Vec<WalEntry>> {
        let file = File::open(&self.path)?;
        let mut reader = BufReader::new(file);
        let mut entries = Vec::new();

        loop {
            // Read CRC32
            let mut crc_buf = [0u8; 4];
            if let Err(e) = reader.read_exact(&mut crc_buf) {
                if e.kind() == io::ErrorKind::UnexpectedEof {
                    break;
                }
                return Err(e);
            }
            let expected_crc = u32::from_le_bytes(crc_buf);

            // Read Length
            let mut len_buf = [0u8; 8];
            reader.read_exact(&mut len_buf)?;
            let len = u64::from_le_bytes(len_buf);

            // Read Data
            let mut data = vec![0u8; len as usize];
            reader.read_exact(&mut data)?;

            // Verify CRC
            let mut hasher = Hasher::new();
            hasher.update(&data);
            if hasher.finalize() != expected_crc {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "CRC mismatch - corrupted WAL entry"));
            }

            // Deserialize
            let entry: WalEntry = bincode::deserialize(&data)
                .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
            entries.push(entry);
        }

        Ok(entries)
    }
}
