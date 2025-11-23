use ndarray::{ArrayView1, Array1};

pub fn euclidean_distance(a: &ArrayView1<f32>, b: &ArrayView1<f32>) -> f32 {
    let diff = a - b;
    diff.dot(&diff).sqrt()
}

pub fn cosine_similarity(a: &ArrayView1<f32>, b: &ArrayView1<f32>) -> f32 {
    let dot = a.dot(b);
    let norm_a = a.dot(a).sqrt();
    let norm_b = b.dot(b).sqrt();
    dot / (norm_a * norm_b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn test_euclidean() {
        let a = arr1(&[1.0, 2.0, 3.0]);
        let b = arr1(&[4.0, 5.0, 6.0]);
        // diff = [-3, -3, -3]
        // dot = 9 + 9 + 9 = 27
        // sqrt(27) approx 5.196
        let dist = euclidean_distance(&a.view(), &b.view());
        assert!((dist - 5.196152).abs() < 1e-5);
    }
}
