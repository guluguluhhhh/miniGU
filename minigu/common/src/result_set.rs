use std::collections::HashSet;
use std::sync::Arc;

use crate::data_chunk::DataChunk;

/// Position of a DataChunk within a ResultSet
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DataChunkPos(pub usize);

/// Position of data within a ResultSet (DataChunk + column position)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DataPos {
    pub data_chunk_pos: DataChunkPos,
    pub column_pos: usize,
}

impl DataPos {
    pub fn new(data_chunk_pos: DataChunkPos, column_pos: usize) -> Self {
        Self {
            data_chunk_pos,
            column_pos,
        }
    }
}

/// A collection of DataChunks representing factorized query results
#[derive(Debug, Clone)]
pub struct ResultSet {
    /// Multiplicity factor for tuple counts
    pub multiplicity: u64,
    /// Vector of DataChunks containing the actual data
    data_chunks: Vec<Arc<DataChunk>>,
}

impl ResultSet {
    #[inline]
    pub fn new() -> Self {
        Self {
            multiplicity: 1,
            data_chunks: Vec::new(),
        }
    }

    #[inline]
    pub fn is_chunk_flat(&self, pos: DataChunkPos) -> bool {
        if let Some(chunk) = self.data_chunks.get(pos.0) {
            !chunk.is_unflat()
        } else {
            false
        }
    }

    #[inline]
    pub fn get_unflat_chunks(&self) -> Vec<DataChunkPos> {
        self.data_chunks
            .iter()
            .enumerate()
            .filter_map(|(i, chunk)| {
                if chunk.is_unflat() {
                    Some(DataChunkPos(i))
                } else {
                    None
                }
            })
            .collect()
    }

    #[inline]
    pub fn get_flat_chunks(&self) -> Vec<DataChunkPos> {
        self.data_chunks
            .iter()
            .enumerate()
            .filter_map(|(i, chunk)| {
                if !chunk.is_unflat() {
                    Some(DataChunkPos(i))
                } else {
                    None
                }
            })
            .collect()
    }

    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            multiplicity: 1,
            data_chunks: Vec::with_capacity(capacity),
        }
    }

    #[inline]
    pub fn push(&mut self, data_chunk: DataChunk) {
        self.data_chunks.push(Arc::new(data_chunk));
    }

    #[inline]
    pub fn get_data_chunk(&self, pos: DataChunkPos) -> Option<&Arc<DataChunk>> {
        self.data_chunks.get(pos.0)
    }

    /// Get a specific column from a specific chunk
    /// Panics if the chunk or column doesn't exist (for simplicity)
    #[inline]
    pub fn get_column(&self, data_pos: &DataPos) -> &Arc<dyn arrow::array::Array> {
        let chunk = self
            .get_data_chunk(data_pos.data_chunk_pos)
            .expect("Chunk must exist");
        &chunk.columns()[data_pos.column_pos]
    }

    /// Get the total number of tuples in the specified data chunks without considering base multiplicity
    /// For unflat chunks, this computes the Cartesian product size
    /// For flat chunks, pass
    #[inline]
    pub fn get_num_tuples_without_multiplicity(
        &self,
        data_chunks_pos_in_scope: &HashSet<u32>,
    ) -> u64 {
        assert!(
            !data_chunks_pos_in_scope.is_empty(),
            "data_chunks_pos_in_scope must not be empty"
        );

        let mut num_tuples = 1u64;

        for &data_chunk_pos in data_chunks_pos_in_scope {
            if let Some(chunk) = self.data_chunks.get(data_chunk_pos as usize) {
                if chunk.is_unflat() {
                    // Only unflat chunks participate in Cartesian product
                    num_tuples *= chunk.cardinality() as u64;
                }
                // Flat chunks are ignored - they don't contribute to tuple count
            }
        }

        num_tuples
    }

    /// Get the total number of tuples in the specified data chunks considering multiplicity
    #[inline]
    pub fn get_num_tuples(&self, data_chunks_pos_in_scope: &HashSet<u32>) -> u64 {
        self.get_num_tuples_without_multiplicity(data_chunks_pos_in_scope) * self.multiplicity
    }

    #[inline]
    pub fn num_data_chunks(&self) -> usize {
        self.data_chunks.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data_chunks.is_empty()
    }

    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &Arc<DataChunk>> {
        self.data_chunks.iter()
    }
}

impl Default for ResultSet {
    fn default() -> Self {
        Self::new()
    }
}

impl FromIterator<DataChunk> for ResultSet {
    fn from_iter<T: IntoIterator<Item = DataChunk>>(iter: T) -> Self {
        let data_chunks = iter.into_iter().map(Arc::new).collect();
        Self {
            multiplicity: 1,
            data_chunks,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;
    use crate::data_chunk;

    #[test]
    fn test_result_set_basic() {
        let mut result_set = ResultSet::new();
        assert!(result_set.is_empty());
        assert_eq!(result_set.multiplicity, 1);

        let chunk = data_chunk!((Int32, [1, 2, 3]));
        result_set.push(chunk);
        assert_eq!(result_set.num_data_chunks(), 1);
    }

    #[test]
    fn test_get_num_tuples() {
        let mut result_set = ResultSet::new();
        result_set.push(data_chunk!((Int32, [1, 2, 3]))); // 3 tuples
        result_set.push(data_chunk!((Int32, [4, 5]))); // 2 tuples
        result_set.multiplicity = 2;

        let scope = HashSet::from([0, 1]);
        // Without multiplicity: 3 * 2 = 6 (Cartesian product)
        assert_eq!(result_set.get_num_tuples_without_multiplicity(&scope), 6);
        // With multiplicity: 6 * 2 = 12
        assert_eq!(result_set.get_num_tuples(&scope), 12);
    }

    #[test]
    fn test_get_num_tuples_without_multiplicity() {
        let mut result_set = ResultSet::new();
        result_set.push(data_chunk!((Int32, [1, 2, 3]))); // 3 tuples
        result_set.push(data_chunk!((Int32, [4, 5]))); // 2 tuples
        result_set.push(data_chunk!((Int32, [7]))); // 1 tuple

        // Test single data chunk
        let scope_single = HashSet::from([0]);
        assert_eq!(
            result_set.get_num_tuples_without_multiplicity(&scope_single),
            3
        );

        // Test multiple data chunks (Cartesian product)
        let scope_multiple = HashSet::from([0, 1, 2]);
        assert_eq!(
            result_set.get_num_tuples_without_multiplicity(&scope_multiple),
            6
        ); // 3 * 2 * 1
    }

    #[test]
    fn test_from_iter() {
        let chunks = vec![data_chunk!((Int32, [1, 2])), data_chunk!((Int32, [3, 4]))];

        let result_set: ResultSet = chunks.into_iter().collect();
        assert_eq!(result_set.num_data_chunks(), 2);
    }
}
