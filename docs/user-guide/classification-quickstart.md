# Classification Quickstart

Use `CartoBoostClassifier` when each row has a discrete taxi-domain label:
airport trip versus non-airport trip, high-delay bucket, route-risk class, or a
pickup-demand surge class. The Python estimator keeps sklearn-style labels and
delegates binary or multiclass logloss training to the Rust backend.

## Fit A Binary Classifier

```python
from cartoboost import CartoBoostClassifier

clf = CartoBoostClassifier(
    n_estimators=200,
    learning_rate=0.04,
    max_depth=4,
    min_samples_leaf=20,
    splitters=["axis", "diagonal_2d", "gaussian_2d", "periodic:24"],
    class_weight="balanced",
)
clf.fit(X_train, airport_trip_flag)

prob_airport = clf.predict_proba(X_validation)[:, list(clf.classes_).index(1)]
labels = clf.predict(X_validation)
margin = clf.decision_function(X_validation)
```

For two classes, `objective="auto"` selects binary logloss. For three or more
labels, it selects multiclass logloss. Pass `objective="binary_logloss"` or
`objective="multiclass_logloss"` when the benchmark protocol needs an explicit
objective name in artifacts.

## Evaluate

Use the same validation rows and feature columns for CartoBoost and baseline
classifiers. Report logloss plus threshold-free metrics that match the
question:

```python
from cartoboost import brier_score, ece_calibration_error, logloss, pr_auc, roc_auc

metrics = {
    "logloss": logloss(y_validation, clf.predict_proba(X_validation), labels=clf.classes_),
    "roc_auc": roc_auc(y_validation, prob_airport, positive_label=1),
    "pr_auc": pr_auc(y_validation, prob_airport, positive_label=1),
    "brier": brier_score(y_validation, prob_airport, positive_label=1),
    "ece": ece_calibration_error(y_validation, prob_airport, positive_label=1),
}
```

For rare positive labels, PR-AUC is usually more revealing than accuracy. Keep
dummy, logistic, LightGBM, or XGBoost baselines on the same split before
interpreting any CartoBoost gain.

## Save And Load

```python
clf.save("airport-trip-classifier.json")
restored = CartoBoostClassifier.load("airport-trip-classifier.json")
restored.predict_proba(X_validation)
```

Classifier artifacts include native trees plus Python class-label metadata.
Categorical feature mappings are saved in the same wrapper artifact when used.
