//! Streaming execution support for large graphs and datasets.

use tensorlogic_ir::EinsumGraph;

use crate::batch::BatchResult;

/// Streaming execution mode for handling large datasets
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamingMode {
    /// Process all at once (no streaming)
    None,
    /// Stream inputs in fixed-size chunks
    FixedChunk(usize),
    /// Stream with dynamic chunk sizing based on memory
    DynamicChunk { target_memory_mb: usize },
    /// Stream with adaptive chunking based on performance
    Adaptive { initial_chunk: usize },
}

/// Configuration for streaming execution
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    pub mode: StreamingMode,
    pub prefetch_chunks: usize,
    pub overlap_compute_io: bool,
    pub checkpoint_interval: Option<usize>,
}

impl StreamingConfig {
    pub fn new(mode: StreamingMode) -> Self {
        StreamingConfig {
            mode,
            prefetch_chunks: 1,
            overlap_compute_io: true,
            checkpoint_interval: None,
        }
    }

    pub fn with_prefetch(mut self, num_chunks: usize) -> Self {
        self.prefetch_chunks = num_chunks;
        self
    }

    pub fn with_checkpointing(mut self, interval: usize) -> Self {
        self.checkpoint_interval = Some(interval);
        self
    }

    pub fn disable_overlap(mut self) -> Self {
        self.overlap_compute_io = false;
        self
    }
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self::new(StreamingMode::None)
    }
}

/// Stream chunk metadata
#[derive(Debug, Clone)]
pub struct ChunkMetadata {
    pub chunk_id: usize,
    pub start_idx: usize,
    pub end_idx: usize,
    pub size: usize,
    pub is_last: bool,
}

impl ChunkMetadata {
    pub fn new(chunk_id: usize, start_idx: usize, end_idx: usize, total_size: usize) -> Self {
        let size = end_idx - start_idx;
        let is_last = end_idx >= total_size;
        ChunkMetadata {
            chunk_id,
            start_idx,
            end_idx,
            size,
            is_last,
        }
    }
}

/// Streaming execution result with chunk information
#[derive(Debug, Clone)]
pub struct StreamResult<T> {
    pub outputs: Vec<T>,
    pub metadata: ChunkMetadata,
    pub processing_time_ms: f64,
}

impl<T> StreamResult<T> {
    pub fn new(outputs: Vec<T>, metadata: ChunkMetadata, processing_time_ms: f64) -> Self {
        StreamResult {
            outputs,
            metadata,
            processing_time_ms,
        }
    }

    pub fn throughput_items_per_sec(&self) -> f64 {
        if self.processing_time_ms > 0.0 {
            (self.metadata.size as f64) / (self.processing_time_ms / 1000.0)
        } else {
            0.0
        }
    }
}

/// Trait for executors that support streaming execution
pub trait TlStreamingExecutor {
    type Tensor;
    type Error;

    /// Execute graph on a stream of input chunks
    fn execute_stream(
        &mut self,
        graph: &EinsumGraph,
        input_stream: Vec<Vec<Vec<Self::Tensor>>>,
        config: &StreamingConfig,
    ) -> Result<Vec<StreamResult<Self::Tensor>>, Self::Error>;

    /// Execute graph on a single chunk with metadata
    fn execute_chunk(
        &mut self,
        graph: &EinsumGraph,
        chunk_inputs: Vec<Self::Tensor>,
        metadata: &ChunkMetadata,
    ) -> Result<StreamResult<Self::Tensor>, Self::Error>;

    /// Get recommended chunk size based on available memory
    fn recommend_chunk_size(&self, graph: &EinsumGraph, available_memory_mb: usize) -> usize {
        let _ = (graph, available_memory_mb);
        32 // Default recommendation
    }

    /// Estimate memory usage per chunk
    fn estimate_chunk_memory(&self, graph: &EinsumGraph, chunk_size: usize) -> usize {
        let _ = (graph, chunk_size);
        chunk_size * 1024 * 1024 // Default: 1MB per item
    }
}

/// Chunk iterator for breaking large batches into streams
pub struct ChunkIterator {
    total_size: usize,
    chunk_size: usize,
    current_chunk: usize,
}

impl ChunkIterator {
    pub fn new(total_size: usize, chunk_size: usize) -> Self {
        ChunkIterator {
            total_size,
            chunk_size,
            current_chunk: 0,
        }
    }

    pub fn from_config(total_size: usize, config: &StreamingConfig) -> Self {
        let chunk_size = match config.mode {
            StreamingMode::None => total_size,
            StreamingMode::FixedChunk(size) => size,
            StreamingMode::DynamicChunk { target_memory_mb } => {
                // Estimate: ~1MB per item, adjust based on target memory
                (target_memory_mb).max(1)
            }
            StreamingMode::Adaptive { initial_chunk } => initial_chunk,
        };

        ChunkIterator::new(total_size, chunk_size)
    }

    pub fn num_chunks(&self) -> usize {
        self.total_size.div_ceil(self.chunk_size)
    }

    pub fn current_chunk(&self) -> usize {
        self.current_chunk
    }
}

impl Iterator for ChunkIterator {
    type Item = ChunkMetadata;

    fn next(&mut self) -> Option<Self::Item> {
        let start_idx = self.current_chunk * self.chunk_size;
        if start_idx >= self.total_size {
            return None;
        }

        let end_idx = (start_idx + self.chunk_size).min(self.total_size);
        let metadata = ChunkMetadata::new(self.current_chunk, start_idx, end_idx, self.total_size);

        self.current_chunk += 1;
        Some(metadata)
    }
}

/// Stream processor for handling streaming execution
pub struct StreamProcessor {
    config: StreamingConfig,
}

impl StreamProcessor {
    pub fn new(config: StreamingConfig) -> Self {
        StreamProcessor { config }
    }

    /// Split batch result into chunks based on configuration
    pub fn split_batch<T: Clone>(&self, batch: &BatchResult<T>) -> Vec<(ChunkMetadata, Vec<T>)> {
        let total_size = batch.len();
        let iter = ChunkIterator::from_config(total_size, &self.config);

        iter.map(|metadata| {
            let chunk_data: Vec<T> = batch.outputs[metadata.start_idx..metadata.end_idx].to_vec();
            (metadata, chunk_data)
        })
        .collect()
    }

    /// Merge stream results back into a single batch
    pub fn merge_results<T>(results: Vec<StreamResult<T>>) -> BatchResult<T> {
        let total_size: usize = results.iter().map(|r| r.outputs.len()).sum();
        let mut outputs = Vec::with_capacity(total_size);

        for result in results {
            outputs.extend(result.outputs);
        }

        BatchResult::new(outputs)
    }

    /// Calculate adaptive chunk size based on performance metrics
    pub fn adaptive_chunk_size(&self, results: &[StreamResult<impl Clone>]) -> usize {
        if results.is_empty() {
            return 32; // Default
        }

        // Calculate average throughput
        let avg_throughput: f64 = results
            .iter()
            .map(|r| r.throughput_items_per_sec())
            .sum::<f64>()
            / results.len() as f64;

        // Adjust chunk size based on throughput
        // Goal: maintain ~100ms per chunk for good responsiveness
        let target_time_ms = 100.0;
        let items_per_chunk = (avg_throughput * target_time_ms / 1000.0) as usize;

        items_per_chunk.clamp(1, 1000) // Clamp between 1 and 1000
    }

    pub fn config(&self) -> &StreamingConfig {
        &self.config
    }
}

impl Default for StreamProcessor {
    fn default() -> Self {
        Self::new(StreamingConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_streaming_config() {
        let config = StreamingConfig::new(StreamingMode::FixedChunk(64))
            .with_prefetch(2)
            .with_checkpointing(100);

        assert_eq!(config.mode, StreamingMode::FixedChunk(64));
        assert_eq!(config.prefetch_chunks, 2);
        assert_eq!(config.checkpoint_interval, Some(100));
    }

    #[test]
    fn test_chunk_metadata() {
        let metadata = ChunkMetadata::new(0, 0, 32, 100);
        assert_eq!(metadata.chunk_id, 0);
        assert_eq!(metadata.size, 32);
        assert!(!metadata.is_last);

        let last_metadata = ChunkMetadata::new(3, 96, 100, 100);
        assert!(last_metadata.is_last);
    }

    #[test]
    fn test_stream_result() {
        let metadata = ChunkMetadata::new(0, 0, 32, 100);
        let result: StreamResult<i32> = StreamResult::new(vec![1, 2, 3], metadata, 100.0);

        assert_eq!(result.outputs.len(), 3);
        let throughput = result.throughput_items_per_sec();
        assert!(throughput > 0.0);
    }

    #[test]
    fn test_chunk_iterator() {
        let iter = ChunkIterator::new(100, 32);
        assert_eq!(iter.num_chunks(), 4); // 32, 32, 32, 4

        let chunks: Vec<_> = iter.collect();
        assert_eq!(chunks.len(), 4);
        assert_eq!(chunks[0].size, 32);
        assert_eq!(chunks[3].size, 4);
        assert!(chunks[3].is_last);
    }

    #[test]
    fn test_chunk_iterator_from_config() {
        let config = StreamingConfig::new(StreamingMode::FixedChunk(25));
        let iter = ChunkIterator::from_config(100, &config);

        assert_eq!(iter.chunk_size, 25);
        assert_eq!(iter.num_chunks(), 4);
    }

    #[test]
    fn test_stream_processor_split() {
        let batch = BatchResult::new(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        let config = StreamingConfig::new(StreamingMode::FixedChunk(3));
        let processor = StreamProcessor::new(config);

        let chunks = processor.split_batch(&batch);
        assert_eq!(chunks.len(), 4); // 3, 3, 3, 1

        assert_eq!(chunks[0].1, vec![1, 2, 3]);
        assert_eq!(chunks[1].1, vec![4, 5, 6]);
        assert_eq!(chunks[2].1, vec![7, 8, 9]);
        assert_eq!(chunks[3].1, vec![10]);
    }

    #[test]
    fn test_stream_processor_merge() {
        let metadata1 = ChunkMetadata::new(0, 0, 3, 10);
        let metadata2 = ChunkMetadata::new(1, 3, 6, 10);
        let metadata3 = ChunkMetadata::new(2, 6, 10, 10);

        let results = vec![
            StreamResult::new(vec![1, 2, 3], metadata1, 10.0),
            StreamResult::new(vec![4, 5, 6], metadata2, 10.0),
            StreamResult::new(vec![7, 8, 9, 10], metadata3, 10.0),
        ];

        let batch = StreamProcessor::merge_results(results);
        assert_eq!(batch.len(), 10);
        assert_eq!(batch.outputs, vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    }

    #[test]
    fn test_adaptive_chunk_size() {
        let processor = StreamProcessor::default();

        let metadata = ChunkMetadata::new(0, 0, 100, 1000);
        let results = vec![
            StreamResult::new(vec![(); 100], metadata.clone(), 50.0), // 2000 items/sec
            StreamResult::new(vec![(); 100], metadata.clone(), 100.0), // 1000 items/sec
            StreamResult::new(vec![(); 100], metadata, 75.0),         // 1333 items/sec
        ];

        let chunk_size = processor.adaptive_chunk_size(&results);
        assert!(chunk_size > 0);
        assert!(chunk_size <= 1000); // Within clamp range
    }

    #[test]
    fn test_streaming_modes() {
        assert_eq!(StreamingMode::None, StreamingConfig::default().mode);

        let fixed = StreamingMode::FixedChunk(64);
        assert_eq!(fixed, StreamingMode::FixedChunk(64));

        let dynamic = StreamingMode::DynamicChunk {
            target_memory_mb: 512,
        };
        match dynamic {
            StreamingMode::DynamicChunk { target_memory_mb } => {
                assert_eq!(target_memory_mb, 512);
            }
            _ => panic!("Wrong mode"),
        }
    }
}
