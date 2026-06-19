use super::SparseSetColumn;

#[test]
fn new_sorts_and_deduplicates_rows() {
    let column = SparseSetColumn::new(vec![vec![7, 3, 7, 1], vec![], vec![2, 2]]);

    assert_eq!(column.row(0), Some(&[1, 3, 7][..]));
    assert_eq!(column.row(1), Some(&[][..]));
    assert_eq!(column.row(2), Some(&[2][..]));
}

#[test]
fn contains_any_handles_empty_duplicate_and_missing_rows() {
    let column = SparseSetColumn::new(vec![vec![1, 3, 7], vec![]]);

    assert!(column.contains_any(0, [9, 7, 7]));
    assert!(!column.contains_any(0, [9, 11]));
    assert!(!column.contains_any(0, Vec::<u64>::new()));
    assert!(!column.contains_any(1, [1]));
    assert!(!column.contains_any(2, [1]));
}

#[test]
fn contains_checks_single_id_membership() {
    let column = SparseSetColumn::new(vec![vec![4, 5]]);

    assert!(column.contains(0, 4));
    assert!(!column.contains(0, 6));
    assert!(!column.contains(1, 4));
}

#[test]
fn membership_handles_direct_unsorted_rows() {
    let column = SparseSetColumn {
        values: vec![vec![9, 1, 7]],
    };

    assert!(column.contains(0, 7));
    assert!(column.contains_any(0, [3, 9]));
    assert!(!column.contains_any(0, [3, 5]));
}
