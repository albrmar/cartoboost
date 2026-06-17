import numpy as np

from cartoboost import CartoBoostRegressor
from cartoboost.neural import NeuralEmbeddingFeatures


def test_regressor_can_fit_on_augmented_neural_features():
    row_count = 120
    ids = np.tile(np.array([10, 11, 12, 13, 14], dtype=np.uint64), row_count // 5)
    residual = np.array(
        [1.0 + 0.5 * (idx % 5) for idx in range(row_count)],
        dtype=np.float64,
    )

    transformer = NeuralEmbeddingFeatures(dim=4, random_state=12).fit(ids, residual)
    x_base = np.column_stack(
        [
            np.linspace(0.0, 1.0, row_count),
            np.sin(np.linspace(0.0, 4.0, row_count)),
        ]
    )
    neural = transformer.transform(ids)
    x_augmented = np.hstack([x_base, neural])

    y = (
        1.5 * x_base[:, 0]
        + 0.7 * x_base[:, 1]
        + 2.0 * neural[:, 0]
        + 0.2 * neural[:, 1]
    )

    model = CartoBoostRegressor(
        n_estimators=12,
        learning_rate=0.2,
        max_depth=2,
        min_samples_leaf=1,
    ).fit(x_augmented, y)

    predictions = model.predict(x_augmented)
    mae = float(np.mean(np.abs(predictions - y)))

    assert mae < 0.8
