use super::{FeatureKind, FeatureSchema};

#[test]
fn unnamed_numeric_schema_is_deterministic() {
    let schema = FeatureSchema::unnamed_numeric(3);

    assert_eq!(
        schema.names,
        vec![
            "feature_0".to_string(),
            "feature_1".to_string(),
            "feature_2".to_string()
        ]
    );
    assert_eq!(schema.kinds, vec![FeatureKind::Numeric; 3]);
    assert_eq!(schema.len(), 3);
}

#[test]
fn schema_validation_rejects_mismatched_fields() {
    let schema = FeatureSchema {
        names: vec!["x".to_string()],
        kinds: vec![FeatureKind::Numeric, FeatureKind::SparseSet],
    };

    assert!(schema.validate().is_err());
}
