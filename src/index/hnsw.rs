

use std::collections::BinaryHeap;

#[derive(Debug, Clone, Copy)]
struct Candidate {
    id: u32,
    distance: f32,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Eq for Candidate {}

// Max-heap behavior: larger distance is "greater"
impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.distance.partial_cmp(&other.distance)
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

#[derive(Serialize, Deserialize)]
pub struct Hnsw {
    pub nodes: HashMap<u32, Arc<RwLock<Node>>>,
    pub entry_point: Option<u32>,
    pub max_layers: usize,
    pub ef_construction: usize,
    pub m: usize,
    pub m_max0: usize,
    pub level_mult: f64,
}

impl Hnsw {
    pub fn new(m: usize, ef_construction: usize) -> Self {
        let m_max0 = m * 2;
        let level_mult = 1.0 / (m as f64).ln();
        Hnsw {
            nodes: HashMap::new(),
            entry_point: None,
            max_layers: 0,
            ef_construction,
            m,
            m_max0,
            level_mult,
        }
    }

    pub fn save_snapshot(&self, path: &str) -> io::Result<()> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        bincode::serialize_into(writer, self).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
        Ok(())
    }

    pub fn load_snapshot(path: &str) -> io::Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let hnsw: Hnsw = bincode::deserialize_from(reader).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
        Ok(hnsw)
    }

    fn random_level(&self) -> usize {
        let mut rng = rand::thread_rng();
        let r: f64 = rng.gen();
        (-r.ln() * self.level_mult) as usize
    }

    fn dist(&self, v1: &ArrayView1<f32>, v2: &ArrayView1<f32>) -> f32 {
        euclidean_distance(v1, v2)
    }

    pub fn insert(&mut self, id: u32, vector: Array1<f32>) {
        let level = self.random_level();
        let new_node = Arc::new(RwLock::new(Node {
            id,
            vector: vector.clone(),
            layers: vec![vec![]; level + 1],
        }));
        
        self.nodes.insert(id, new_node.clone());

        let entry_point = match self.entry_point {
            Some(ep) => ep,
            None => {
                self.entry_point = Some(id);
                self.max_layers = level;
                return;
            }
        };

        let mut curr_ep = entry_point;
        let mut curr_dist = self.dist(&vector.view(), &self.nodes[&curr_ep].read().unwrap().vector.view());

        // Phase 1: Zoom down to the insertion level
        for l in (level + 1..=self.max_layers).rev() {
            let mut changed = true;
            while changed {
                changed = false;
                let curr_node = self.nodes.get(&curr_ep).unwrap();
                let guard = curr_node.read().unwrap();
                
                if let Some(neighbors) = guard.layers.get(l) {
                    for &neighbor_id in neighbors {
                        let neighbor_node = self.nodes.get(&neighbor_id).unwrap();
                        let d = self.dist(&vector.view(), &neighbor_node.read().unwrap().vector.view());
                        if d < curr_dist {
                            curr_dist = d;
                            curr_ep = neighbor_id;
                            changed = true;
                        }
                    }
                }
            }
        }

        // Phase 2: Insert at each level from `level` down to 0
        for l in (0..=std::cmp::min(level, self.max_layers)).rev() {
            // Find ef_construction nearest neighbors at this layer
            let mut candidates = self.search_layer(&vector.view(), curr_ep, self.ef_construction, l);
            
            // Select M neighbors
            let neighbors = self.select_neighbors(&mut candidates, self.m);

            // Add bidirectional connections
            {
                let mut new_guard = new_node.write().unwrap();
                new_guard.layers[l] = neighbors.clone();
            }

            for &neighbor_id in &neighbors {
                let neighbor_node = self.nodes.get(&neighbor_id).unwrap();
                let mut neighbor_guard = neighbor_node.write().unwrap();
                neighbor_guard.layers[l].push(id);
                
                // Prune connections if too many
                let m_max = if l == 0 { self.m_max0 } else { self.m };
                if neighbor_guard.layers[l].len() > m_max {
                    // Re-evaluate neighbors for this node to prune the worst one
                    // Optimization: Just remove the furthest one? 
                    // Better: Run select_neighbors on the existing list + new one.
                    // For MVP, let's just keep the closest ones.
                    let mut neighbor_candidates = BinaryHeap::new();
                    let neighbor_vec = neighbor_guard.vector.clone(); // Clone to avoid borrow issues
                    
                    for &n_id in &neighbor_guard.layers[l] {
                        let n_node = self.nodes.get(&n_id).unwrap();
                        let d = self.dist(&neighbor_vec.view(), &n_node.read().unwrap().vector.view());
                        neighbor_candidates.push(Candidate { id: n_id, distance: d });
                    }
                    
                    // Keep only m_max
                    while neighbor_candidates.len() > m_max {
                        neighbor_candidates.pop(); // Pops the largest distance
                    }
                    
                    neighbor_guard.layers[l] = neighbor_candidates.into_vec().iter().map(|c| c.id).collect();
                }
            }
            
            // Update entry point for next layer (closest from candidates)
            if candidates.peek().is_some() {
                 // candidates is MaxHeap, so peek gives furthest? No wait.
                 // search_layer returns 'nearest_neighbors' which is a MaxHeap of the *closest* found so far.
                 // So peek gives the *worst* of the best.
                 // We want the *best* of the best to be the entry point for the next layer.
                 // We need to iterate to find the min.
                 if let Some(best_cand) = candidates.iter().min_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap()) {
                     curr_ep = best_cand.id;
                     curr_dist = best_cand.distance;
                 }
            }
        }
        
        if level > self.max_layers {
            self.max_layers = level;
            self.entry_point = Some(id);
        }
    }

    pub fn search(&self, query: &ArrayView1<f32>, k: usize) -> Vec<(u32, f32)> {
        if self.entry_point.is_none() {
            return vec![];
        }

        let mut curr_ep = self.entry_point.unwrap();
        let mut curr_dist = self.dist(query, &self.nodes[&curr_ep].read().unwrap().vector.view());

        // 1. Zoom down to layer 1 (greedy search)
        for l in (1..=self.max_layers).rev() {
             let mut changed = true;
            while changed {
                changed = false;
                let curr_node = self.nodes.get(&curr_ep).unwrap();
                let guard = curr_node.read().unwrap();
                
                if let Some(neighbors) = guard.layers.get(l) {
                    for &neighbor_id in neighbors {
                        let neighbor_node = self.nodes.get(&neighbor_id).unwrap();
                        let d = self.dist(query, &neighbor_node.read().unwrap().vector.view());
                        if d < curr_dist {
                            curr_dist = d;
                            curr_ep = neighbor_id;
                            changed = true;
                        }
                    }
                }
            }
        }

        // 2. Search layer 0 (Beam search / search_layer)
        let mut candidates = self.search_layer(query, curr_ep, self.ef_construction.max(k), 0);
        
        // Return top K
        let mut results = Vec::new();
        while let Some(c) = candidates.pop() {
            results.push((c.id, c.distance));
        }
        // candidates popped from MaxHeap -> largest distance first.
        // We want smallest distance first.
        results.reverse();
        results.truncate(k);
        results
    }

    fn search_layer(&self, query: &ArrayView1<f32>, entry_point: u32, ef: usize, layer: usize) -> BinaryHeap<Candidate> {
        let mut visited = HashSet::new();
        let mut candidates = BinaryHeap::new(); // Min-heap of candidates to explore (closest first)
        let mut nearest_neighbors = BinaryHeap::new(); // Max-heap of found neighbors (furthest first)

        let entry_dist = self.dist(query, &self.nodes[&entry_point].read().unwrap().vector.view());
        let entry_cand = Candidate { id: entry_point, distance: entry_dist };

        visited.insert(entry_point);
        candidates.push(Reverse(entry_cand));
        nearest_neighbors.push(entry_cand);

        while let Some(Reverse(curr)) = candidates.pop() {
            if let Some(furthest_found) = nearest_neighbors.peek() {
                 if curr.distance > furthest_found.distance && nearest_neighbors.len() >= ef {
                    break;
                }
            }

            let curr_node = self.nodes.get(&curr.id).unwrap();
            let guard = curr_node.read().unwrap();
            
            if let Some(neighbors) = guard.layers.get(layer) {
                for &neighbor_id in neighbors {
                    if visited.contains(&neighbor_id) {
                        continue;
                    }
                    visited.insert(neighbor_id);

                    let neighbor_node = self.nodes.get(&neighbor_id).unwrap();
                    let dist = self.dist(query, &neighbor_node.read().unwrap().vector.view());
                    let neighbor_cand = Candidate { id: neighbor_id, distance: dist };

                    if nearest_neighbors.len() < ef || dist < nearest_neighbors.peek().unwrap().distance {
                        candidates.push(Reverse(neighbor_cand));
                        nearest_neighbors.push(neighbor_cand);
                        
                        if nearest_neighbors.len() > ef {
                            nearest_neighbors.pop();
                        }
                    }
                }
            }
        }
        nearest_neighbors
    }

    fn select_neighbors(&self, candidates: &mut BinaryHeap<Candidate>, m: usize) -> Vec<u32> {
        // candidates is a MaxHeap of the 'ef' nearest neighbors found.
        // We want the 'm' closest ones.
        // Since it's a MaxHeap, popping gives the furthest.
        // We can pop until we have m left? No, we might have fewer than m.
        
        let mut result = Vec::new();
        // Drain the heap
        let sorted_candidates = candidates.clone().into_sorted_vec(); 
        // into_sorted_vec returns ascending order (smallest distance first)
        
        for c in sorted_candidates.iter().take(m) {
            result.push(c.id);
        }
        result
    }
}
