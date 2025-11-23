#[cfg(test)]
mod tests {
    use crate::index::hnsw::Hnsw;
    use ndarray::Array1;
    use rand::Rng;

    #[test]
    fn test_hnsw_basic_insert_search() {
        let mut hnsw = Hnsw::new(16, 100);
        
        // Insert a vector
        let v1 = Array1::from(vec![1.0, 2.0, 3.0]);
        hnsw.insert(1, v1.clone());
        
        // Search for it
        let results = hnsw.search(&v1.view(), 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 1);
        // Distance should be 0 (or very close due to float precision)
        assert!(results[0].1 < 1e-6);
    }

    #[test]
    fn test_hnsw_multiple_inserts() {
        let mut hnsw = Hnsw::new(16, 100);
        let mut rng = rand::thread_rng();
        
        let mut vectors = Vec::new();
        for i in 0..100 {
            let v: Array1<f32> = Array1::from((0..10).map(|_| rng.gen()).collect::<Vec<f32>>());
            hnsw.insert(i, v.clone());
            vectors.push(v);
        }
        
        // Search for the 50th vector
        let query = &vectors[50];
        let results = hnsw.search(&query.view(), 1);
        assert_eq!(results[0].0, 50);
        assert!(results[0].1 < 1e-6);
    }

    #[test]
    fn test_hnsw_recall() {
        // This test verifies that we can find the nearest neighbor in a small dataset
        let mut hnsw = Hnsw::new(16, 100);
        let mut rng = rand::thread_rng();
        
        let mut vectors = Vec::new();
        for i in 0..200 {
            let v: Array1<f32> = Array1::from((0..10).map(|_| rng.gen()).collect::<Vec<f32>>());
            hnsw.insert(i, v.clone());
            vectors.push(v);
        }
        
        // Generate a random query
        let query: Array1<f32> = Array1::from((0..10).map(|_| rng.gen()).collect::<Vec<f32>>());
        
        // Brute force search
        let mut best_id = 0;
        let mut best_dist = f32::MAX;
        
        for (i, v) in vectors.iter().enumerate() {
            let dist = crate::index::distance::euclidean_distance(&query.view(), &v.view());
            if dist < best_dist {
                best_dist = dist;
                best_id = i as u32;
            }
        }
        
        // HNSW search
        let results = hnsw.search(&query.view(), 10); // Search for top 10 to increase chance
        
        // Check if the best_id is in the results
        let found = results.iter().any(|(id, _)| *id == best_id);
        
        // Note: HNSW is approximate, so it might not ALWAYS find the exact nearest, 
        // but with these parameters and small size, it should be very likely.
        // If this flakes, we might need to relax it or increase ef_construction.
        assert!(found, "HNSW failed to find the nearest neighbor. Best: {}, Found: {:?}", best_id, results);
    }
}
