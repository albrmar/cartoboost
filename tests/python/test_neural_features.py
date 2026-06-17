import numpy as np

from cartoboost.neural import NeuralEmbeddingFeatures


def test_train_export_load_roundtrip(tmp_path):
    ids = np.array([10, 10, 20, 30, 30], dtype=np.uint64)
    residual = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)

    transformer = NeuralEmbeddingFeatures(dim=4, random_state=7).fit(ids, residual)
    artifact_path = tmp_path / "embeddings.json"
    transformer.export(artifact_path)

    loaded = NeuralEmbeddingFeatures.from_artifact(artifact_path)

    assert loaded.dim == transformer.dim
    assert np.allclose(loaded.transform([10, 20, 30, 99]), transformer.transform([10, 20, 30, 99]))


def test_transformed_features_can_concat_to_dense_input(tmp_path):
    ids = np.array([1, 2, 3], dtype=np.uint64)
    residual = np.array([10.0, 12.0, 14.0], dtype=np.float64)

    transformer = NeuralEmbeddingFeatures(dim=3, fallback="zero_vector").fit(ids, residual)
    neural_features = transformer.transform(ids)

    x = np.array(
        [
            [0.0, 0.5],
            [1.0, 1.5],
            [2.0, 2.5],
        ],
        dtype=np.float64,
    )
    x_aug = np.hstack([x, neural_features])

    assert x_aug.shape == (3, 5)
    assert np.allclose(x_aug[:, 0:2], x)
