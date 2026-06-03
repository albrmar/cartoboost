use crate::data::FeatureKind;
use crate::tree::Model;

#[test]
fn deserializes_v1_json_without_optional_metadata_fields() {
    let json = r#"{
        "artifact_version": 1,
        "init_prediction": 0.0,
        "learning_rate": 0.1,
        "feature_count": 2,
        "target_name": null,
        "trees": []
    }"#;

    let model: Model = serde_json::from_str(json).unwrap();

    assert!(model.metadata.is_none());
    assert!(model.feature_schema.is_none());
    assert!(model.training_config.is_none());

    let schema = model.feature_schema_or_default();
    assert_eq!(
        schema.names,
        vec!["feature_0".to_string(), "feature_1".to_string()]
    );
    assert_eq!(schema.kinds, vec![FeatureKind::Numeric; 2]);
}
