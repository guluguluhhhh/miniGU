pub mod display;
pub mod row;

use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, AsArray, BooleanArray, ListArray, RecordBatch, new_empty_array,
};
use arrow::buffer::OffsetBuffer;
use arrow::compute;
use arrow::datatypes::DataType;
use itertools::Itertools;
use row::{RowIndexIter, Rows};

use crate::data_type::DataSchema;

#[derive(Debug, Clone, PartialEq)]
pub struct DataChunk {
    columns: Vec<ArrayRef>,
    filter: Option<BooleanArray>,
    cur_idx: Option<usize>, // only used in factorized result_set. None represents unflat
}

impl DataChunk {
    #[inline]
    pub fn new(columns: Vec<ArrayRef>) -> Self {
        assert!(!columns.is_empty(), "columns must not be empty");
        assert!(
            columns.iter().map(|c| c.len()).all_equal(),
            "all columns must have the same length"
        );
        Self {
            columns,
            filter: None,
            cur_idx: Some(0),
        }
    }

    #[inline]
    pub fn new_empty(schema: &DataSchema) -> Self {
        let columns = schema
            .fields()
            .iter()
            .map(|f| {
                let ty = f.ty().to_arrow_data_type();
                new_empty_array(&ty)
            })
            .collect();
        Self::new(columns)
    }

    #[inline]
    pub fn with_filter(self, filter: BooleanArray) -> Self {
        assert_eq!(
            self.len(),
            filter.len(),
            "filter must have the same length as the data chunk"
        );
        Self {
            filter: Some(filter),
            ..self
        }
    }

    #[inline]
    pub fn unfiltered(self) -> Self {
        Self {
            filter: None,
            ..self
        }
    }

    #[inline]
    pub fn cardinality(&self) -> usize {
        if let Some(filter) = &self.filter {
            filter.true_count()
        } else {
            self.len()
        }
    }

    /// Returns `true` if the data chunk is compact.
    #[inline]
    pub fn is_compact(&self) -> bool {
        self.filter.is_some()
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.columns[0].len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    pub fn columns(&self) -> &[ArrayRef] {
        &self.columns
    }

    pub fn factorized_compact(&mut self, filter_list: &ListArray, unflat_column_indices: &[usize]) {
        // Ensure the filter list has the same number of lists as the chunk has rows.
        assert_eq!(
            self.len(),
            filter_list.len(),
            "Filter list must have the same length as the chunk"
        );

        // Flatten the 2D boolean filter list into a 1D boolean array.
        let flat_filter: &BooleanArray = filter_list.values().as_boolean();

        for &col_idx in unflat_column_indices {
            let column = &self.columns[col_idx];
            let list_array: &ListArray = column.as_list();
            let DataType::List(field) = column.data_type() else {
                unreachable!()
            };

            // Flatten the unflat data column into a 1D data array.
            let flat_values = list_array.values();

            // Ensure the flattened data and filter have the same length.
            assert_eq!(
                flat_values.len(),
                flat_filter.len(),
                "Flattened data and filter must have the same length"
            );

            // Apply the 1D filter to the 1D data array.
            let new_flat_values = compute::kernels::filter::filter(flat_values, flat_filter)
                .expect("Vectorized filter kernel failed");

            // Reconstruct the ListArray structure using the original offsets.
            let mut new_offsets = Vec::with_capacity(self.len() + 1);
            new_offsets.push(0);
            let mut current_offset = 0;

            for i in 0..filter_list.len() {
                // Get the boolean sub-list for the current row.
                let sub_filter_array_ref = filter_list.value(i);
                let sub_filter: &BooleanArray = sub_filter_array_ref.as_boolean();

                // The length of the new sub-list is the number of `true` values.
                let true_count = sub_filter.true_count();
                current_offset += true_count;
                new_offsets.push(current_offset as i32);
            }

            let new_list_array = ListArray::new(
                field.clone(),
                OffsetBuffer::new(new_offsets.into()),
                new_flat_values,
                None,
            );

            self.columns[col_idx] = Arc::new(new_list_array);
        }
    }

    /// Compacts the data chunk by eagerly applying the filter to each column.
    ///
    /// # Panics
    ///
    /// Panics if the filter is not applied successfully.
    #[inline]
    pub fn compact(&mut self) {
        if let Some(filter) = self.filter.take() {
            self.columns = self
                .columns
                .iter()
                .map(|column| {
                    compute::kernels::filter::filter(column, &filter)
                        .expect("filter should be applied successfully")
                })
                .collect();
        }
    }

    /// Returns a zero-copy slice of this chunk with the indicated offset and length.
    ///
    /// # Panics
    ///
    /// Panics if the offset and length are out of bounds.
    pub fn slice(&self, offset: usize, length: usize) -> Self {
        let columns = self
            .columns
            .iter()
            .map(|c| c.slice(offset, length))
            .collect();
        let filter = self.filter.as_ref().map(|f| f.slice(offset, length));
        Self {
            columns,
            filter,
            cur_idx: Some(0),
        }
    }

    #[inline]
    pub fn filter(&self) -> Option<&BooleanArray> {
        self.filter.as_ref()
    }

    #[inline]
    pub fn set_cur_idx(&mut self, idx: Option<usize>) {
        self.cur_idx = idx;
    }

    #[inline]
    pub fn cur_idx(&self) -> Option<usize> {
        self.cur_idx
    }

    /// Sets the chunk as unflat (cur_idx = None)
    #[inline]
    pub fn set_unflat(&mut self) {
        self.cur_idx = None;
    }

    /// Checks if the chunk is unflat
    #[inline]
    pub fn is_unflat(&self) -> bool {
        self.cur_idx.is_none()
    }

    #[inline]
    pub fn rows(&self) -> Rows<'_> {
        let iter = if let Some(filter) = self.filter() {
            RowIndexIter::Filtered(filter.iter().enumerate())
        } else {
            RowIndexIter::Unfiltered(0..self.len())
        };
        Rows { chunk: self, iter }
    }

    /// Extends the data chunk horizontally, i.e., appends columns to the right.
    ///
    /// # Panics
    ///
    /// Panics if the lengths of the new columns are not equal to the length of the data
    /// chunk.
    #[inline]
    pub fn append_columns<I>(&mut self, columns: I)
    where
        I: IntoIterator<Item = ArrayRef>,
    {
        self.columns.extend(columns);
        assert!(
            self.columns.iter().all(|c| c.len() == self.len()),
            "all columns must have the same length"
        );
    }

    /// Concatenates multiple data chunks vertically into a single data chunk.
    ///
    /// # Notes
    /// `compact` is called automatically for each input chunk before concatenation, that is, the
    /// resulting data chunk is guaranteed to be unfiltered.
    pub fn concat<I>(chunks: I) -> Self
    where
        I: IntoIterator<Item = DataChunk>,
    {
        let mut chunks = chunks.into_iter().collect_vec();
        assert!(!chunks.is_empty(), "chunks must not be empty");
        assert!(
            chunks.iter().map(|chunk| chunk.columns.len()).all_equal(),
            "all chunks must have the same number of columns"
        );
        chunks.iter_mut().for_each(|chunk| chunk.compact());
        let num_columns = chunks[0].columns.len();
        let columns = (0..num_columns)
            .map(|i| {
                compute::kernels::concat::concat(
                    &chunks
                        .iter()
                        .map(|chunk| chunk.columns[i].as_ref())
                        .collect_vec(),
                )
                .expect("concatenation should be successful")
            })
            .collect();
        Self {
            columns,
            filter: None,
            cur_idx: Some(0),
        }
    }

    /// Takes rows from the data chunk by given indices and creates a new data chunk from these
    /// rows.
    ///
    /// # Panics
    ///
    /// Panics if any of the indices is out of bounds.
    pub fn take(&self, indices: &dyn Array) -> Self {
        let columns = compute::take_arrays(&self.columns, indices, None)
            .expect("`take_arrays` should be successful");
        let filter = self
            .filter
            .as_ref()
            .map(|f| compute::take(f, indices, None).expect("`take` should be successful"))
            .map(|f| f.as_boolean().clone());
        Self {
            columns,
            filter,
            cur_idx: Some(0),
        }
    }

    /// Converts the data chunk to an arrow [`RecordBatch`].
    ///
    /// # Panics
    ///
    /// Panics if the schema does not match the data chunk.
    #[inline]
    pub fn to_arrow_record_batch(&self, schema: &DataSchema) -> RecordBatch {
        let schema = schema.to_arrow_schema();
        let mut chunk = self.clone();
        chunk.compact();
        RecordBatch::try_new(Arc::new(schema), chunk.columns)
            .expect("`schema` should match the data chunk")
    }
}

impl FromIterator<DataChunk> for DataChunk {
    #[inline]
    fn from_iter<T: IntoIterator<Item = DataChunk>>(iter: T) -> Self {
        DataChunk::concat(iter)
    }
}

#[macro_export]
macro_rules! data_chunk {
    ($(($type:ident, [$($values:expr),*])),*) => {
        {
            let columns: Vec<arrow::array::ArrayRef> = vec![$(
                arrow::array::create_array!($type, [$($values),*]),
            )*];
            $crate::data_chunk::DataChunk::new(columns)
        }
    };
    ({ $($filter_values:expr),* }, $(($type:ident, [$($values:expr),*])),*) => {
        {
            let columns: Vec<arrow::array::ArrayRef> = vec![$(
                arrow::array::create_array!($type, [$($values),*]),
            )*];
            let filter = arrow::buffer::BooleanBuffer::from_iter([$($filter_values),*]);
            $crate::data_chunk::DataChunk::new(columns).with_filter(filter.into())
        }
    };
}

#[cfg(test)]
mod tests {
    use arrow::array::create_array;
    use row::OwnedRow;

    use super::*;
    use crate::data_type::{DataField, LogicalType};

    #[test]
    fn test_rows_1() {
        let chunk = data_chunk!((Int32, [1, 2, 3]), (Utf8, ["abc", "def", "ghi"]));
        let rows: Vec<_> = chunk.rows().map(|r| r.into_owned()).collect();
        let expected = vec![
            OwnedRow::new(vec![1i32.into(), "abc".into()]),
            OwnedRow::new(vec![2i32.into(), "def".into()]),
            OwnedRow::new(vec![3i32.into(), "ghi".into()]),
        ];
        assert_eq!(rows, expected);
    }

    #[test]
    fn test_rows_2() {
        let chunk = data_chunk!(
            { true, false, true },
            (Int32, [1, 2, 3]),
            (Utf8, ["abc", "def", "ghi"])
        );
        let rows: Vec<_> = chunk.rows().map(|r| r.into_owned()).collect();
        let expected = vec![
            OwnedRow::new(vec![1i32.into(), "abc".into()]),
            OwnedRow::new(vec![3i32.into(), "ghi".into()]),
        ];
        assert_eq!(rows, expected);
    }

    #[test]
    fn test_slice() {
        let chunk = data_chunk!(
            { true, false, true },
            (Int32, [1, 2, 3]),
            (Utf8, ["abc", "def", "ghi"])
        );
        let sliced = chunk.slice(1, 1);
        let expected = data_chunk!({ false }, (Int32, [2]), (Utf8, ["def"]));
        assert_eq!(sliced, expected);
    }

    #[test]
    #[should_panic]
    fn test_slice_out_of_bounds_1() {
        let chunk = data_chunk!(
            { true, false, true },
            (Int32, [1, 2, 3]),
            (Utf8, ["abc", "def", "ghi"])
        );
        let _sliced = chunk.slice(2, 2);
    }

    #[test]
    #[should_panic]
    fn test_slice_out_of_bounds_2() {
        let chunk = data_chunk!(
            { true, false, true },
            (Int32, [1, 2, 3]),
            (Utf8, ["abc", "def", "ghi"])
        );
        let _sliced = chunk.slice(3, 1);
    }

    #[test]
    fn test_compact() {
        let mut chunk = data_chunk!(
            { false, true, false },
            (Int32, [1, 2, 3]),
            (Utf8, ["abc", "def", "ghi"])
        );
        chunk.compact();
        let expected = data_chunk!((Int32, [2]), (Utf8, ["def"]));
        assert_eq!(chunk, expected);
    }

    #[test]
    fn test_concat() {
        let chunk1 =
            data_chunk!( {false, true, true}, (Int32, [1, 2, 3]), (Utf8, ["aaa", "bbb", "ccc"]));
        let chunk2 = data_chunk!((Int32, [4, 5, 6]), (Utf8, ["ddd", "eee", "fff"]));
        let expected = data_chunk!(
            (Int32, [2, 3, 4, 5, 6]),
            (Utf8, ["bbb", "ccc", "ddd", "eee", "fff"])
        );
        assert_eq!(DataChunk::concat([chunk1, chunk2]), expected);
    }

    #[test]
    fn test_take() {
        let chunk =
            data_chunk!( {false, true, true}, (Int32, [1, 2, 3]), (Utf8, ["abc", "def", "ghi"]));
        let indices = create_array!(Int32, [0, 2]);
        let taken = chunk.take(indices.as_ref());
        let expected = data_chunk!( {false, true}, (Int32, [1, 3]), (Utf8, ["abc", "ghi"]));
        assert_eq!(taken, expected);
    }

    #[test]
    fn test_to_arrow_record_batch() {
        let chunk = data_chunk!((Int32, [1, 2, 3]), (Utf8, ["abc", "def", "ghi"]));
        let schema = DataSchema::new(vec![
            DataField::new("a".to_string(), LogicalType::Int32, false),
            DataField::new("b".to_string(), LogicalType::String, false),
        ]);
        let record_batch = chunk.to_arrow_record_batch(&schema);
        assert_eq!(record_batch.num_rows(), 3);
        assert_eq!(record_batch.num_columns(), 2);
    }
}
