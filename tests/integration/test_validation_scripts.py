from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import types
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest


def ordinal_word(rank: int) -> str:
    words = {
        1: "first",
        2: "second",
        3: "third",
        4: "fourth",
        5: "fifth",
        6: "sixth",
        7: "seventh",
    }
    return words.get(rank, f"{rank}th")


def test_full_validation_script_uses_native_estimator_without_backend_toggle():
    repo_root = Path(__file__).resolve().parents[2]

    full_validation = (repo_root / "scripts" / "run_full_validation.py").read_text(encoding="utf-8")
    splitter_metrics = (repo_root / "scripts" / "run_splitter_acceptance_metrics.py").read_text(
        encoding="utf-8"
    )
    lane_metrics = (repo_root / "scripts" / "run_lane_level_acceptance_metrics.py").read_text(
        encoding="utf-8"
    )

    assert "scripts/run_splitter_acceptance_metrics.py" in full_validation
    assert "scripts/run_lane_level_acceptance_metrics.py" in full_validation
    assert "CartoBoostRegressor(" in splitter_metrics
    assert "CartoBoostRegressor(" in lane_metrics
    backend_kwarg = "backend" + "="
    assert backend_kwarg not in splitter_metrics
    assert backend_kwarg not in lane_metrics


def test_ci_installs_native_extension_before_validation_artifacts():
    repo_root = Path(__file__).resolve().parents[2]
    workflow = (repo_root / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    install_step = workflow.index("uv run maturin develop")
    validation_step = workflow.index("uv run python scripts/run_full_validation.py")

    assert install_step < validation_step


def test_model_benchmark_suite_smoke(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = tmp_path / "model_benchmarks"

    subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "run_model_benchmark_suite.py"),
            "--output-dir",
            str(output_dir),
            "--datasets",
            "normal",
            "--models",
            "mean,cartoboost",
            "--n-rows",
            "120",
            "--no-plots",
        ],
        cwd=repo_root,
        check=True,
    )

    results = output_dir / "results.json"
    report = output_dir / "results.md"
    assert results.exists()
    assert report.exists()
    assert "Normal dense" in report.read_text(encoding="utf-8")


def test_model_benchmark_suite_reports_best_cartoboost_vs_lightgbm():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "run_model_benchmark_suite.py"
    spec = importlib.util.spec_from_file_location("run_model_benchmark_suite", module_path)
    assert spec is not None
    assert spec.loader is not None
    benchmark_suite = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = benchmark_suite
    spec.loader.exec_module(benchmark_suite)

    payload = {
        "workloads": {
            "normal": {
                "splits": {
                    "random": {
                        "models": {
                            "cartoboost": {
                                "status": "ok",
                                "metrics": {"rmse": 0.42, "r2": 0.91},
                            },
                            "cartoboost_neural": {
                                "status": "skipped",
                                "reason": "not applicable",
                            },
                            "lightgbm": {
                                "status": "ok",
                                "metrics": {"rmse": 0.50, "r2": 0.88},
                            },
                        },
                    },
                },
            },
        },
    }

    rows = benchmark_suite.lightgbm_comparison(payload)

    assert rows == [
        {
            "workload": "normal",
            "split": "random",
            "best_cartoboost_model": "cartoboost",
            "best_cartoboost_rmse": 0.42,
            "lightgbm_rmse": 0.50,
            "rmse_delta_vs_lightgbm": -0.08000000000000002,
            "best_cartoboost_r2": 0.91,
            "lightgbm_r2": 0.88,
            "r2_delta_vs_lightgbm": 0.030000000000000027,
            "winner": "cartoboost",
        }
    ]


def test_model_benchmark_suite_link_negatives_avoid_positive_edges():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "run_model_benchmark_suite.py"
    spec = importlib.util.spec_from_file_location("run_model_benchmark_suite", module_path)
    assert spec is not None
    assert spec.loader is not None
    benchmark_suite = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = benchmark_suite
    spec.loader.exec_module(benchmark_suite)

    positive_set = {(0, 0), (0, 1), (0, 2), (1, 0)}
    candidate = benchmark_suite.verified_negative_pair(
        0,
        0,
        node_count=3,
        positive_set=positive_set,
    )

    assert candidate == (2, 0)
    assert candidate not in positive_set
    assert (
        benchmark_suite.verified_negative_pair(
            0,
            0,
            node_count=1,
            positive_set={(0, 0)},
        )
        is None
    )


def test_forecasting_benchmark_loads_m5_local_competition_files(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "forecasting_library_benchmark.py"
    spec = importlib.util.spec_from_file_location("forecasting_library_benchmark", module_path)
    assert spec is not None
    assert spec.loader is not None
    benchmark = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = benchmark
    spec.loader.exec_module(benchmark)

    data_dir = tmp_path / "m5"
    data_dir.mkdir()
    days = 35
    d_columns = ",".join(f"d_{day}" for day in range(1, days + 1))
    values_a = ",".join(str(day) for day in range(1, days + 1))
    values_b = ",".join(str(day * 2) for day in range(1, days + 1))
    (data_dir / "sales_train_validation.csv").write_text(
        "\n".join(
            [
                f"id,item_id,dept_id,cat_id,store_id,state_id,{d_columns}",
                f"FOODS_1_001_CA_1_validation,FOODS_1_001,FOODS_1,FOODS,CA_1,CA,{values_a}",
                f"HOBBIES_1_001_TX_1_validation,HOBBIES_1_001,HOBBIES_1,HOBBIES,TX_1,TX,{values_b}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (data_dir / "calendar.csv").write_text(
        "d,date\n" + "\n".join(f"d_{day},2020-01-{day:02d}" for day in range(1, days + 1)) + "\n",
        encoding="utf-8",
    )

    table, dataset = benchmark.load_m5_fixture(
        types.SimpleNamespace(
            m5_data_dir=data_dir, m5_series_limit=0, m5_history_days=0, no_download=True
        )
    )

    assert dataset["series"] == 2
    assert dataset["horizon"] == 28
    assert table.height == 70
    assert {"lane_id", "date", "loads", *benchmark.STATIC_COVARIATES} <= set(table.columns)


def test_forecasting_benchmark_provenance_helpers_are_stable(tmp_path):
    pl = pytest.importorskip("polars")
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "forecasting_library_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "forecasting_library_benchmark_provenance",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    benchmark = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = benchmark
    spec.loader.exec_module(benchmark)

    first = pl.DataFrame(
        [
            {"lane_id": "PU2->DO2", "date": date(2026, 1, 2), "loads": 2.0},
            {"lane_id": "PU1->DO1", "date": date(2026, 1, 1), "loads": 1.0},
        ]
    )
    second = pl.DataFrame(
        [
            {"loads": 1.0, "date": date(2026, 1, 1), "lane_id": "PU1->DO1"},
            {"loads": 2.0, "date": date(2026, 1, 2), "lane_id": "PU2->DO2"},
        ]
    )

    assert benchmark.canonical_dataset_hash(first) == benchmark.canonical_dataset_hash(second)

    source = tmp_path / "assets_m6.csv"
    source.write_text("symbol,date,price\nAAA,2026-01-01,10.0\n", encoding="utf-8")
    hashes = benchmark.source_file_hashes({"assets_file": str(source)})
    assert hashes["assets_file"] == benchmark.file_sha256(source)

    resources = benchmark.resource_usage_snapshot()
    assert resources["process_cpu_seconds"] >= 0.0
    assert resources["peak_rss_mb"] > 0.0


def test_forecasting_benchmark_auto_objectives_are_metric_aware():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "forecasting_library_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "forecasting_library_benchmark_objectives",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    benchmark = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = benchmark
    spec.loader.exec_module(benchmark)

    assert benchmark.auto_selection_objective("m4") == "owa_proxy"
    assert benchmark.auto_selection_objective("m5") == "rmse"
    assert benchmark.auto_selection_objective("m6") == "rmse"
    assert benchmark.auto_selection_objective("nyc-taxi") == "rmse"
    assert benchmark.auto_selection_objective("synthetic") == "rmse"


def test_forecasting_generalization_wrapper_is_non_m_scalable_guardrail():
    repo_root = Path(__file__).resolve().parents[2]
    wrapper = (repo_root / "scripts" / "forecasting_generalization.py").read_text(encoding="utf-8")

    assert "--no-hyperopt is required for benchmark integrity" in wrapper
    assert '"synthetic"' in wrapper
    assert '"scalable"' in wrapper
    assert '"m4"' not in wrapper
    assert '"m5"' not in wrapper
    assert '"m6"' not in wrapper
    assert '"--no-candidate-selection"' in wrapper
    assert "forecasting_generalization_scalable_synthetic.json" in wrapper


def test_forecasting_benchmark_shared_candidate_cutoffs_are_deterministic():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "forecasting_library_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "forecasting_library_benchmark_cutoffs",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    benchmark = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = benchmark
    spec.loader.exec_module(benchmark)

    timestamps = list(range(100))
    assert benchmark.shared_candidate_validation_cutoffs(timestamps, horizon=14) == [58, 72, 86]
    assert benchmark.shared_candidate_validation_cutoffs(timestamps, horizon=14, source="m5") == [
        72,
        86,
    ]
    assert benchmark.shared_candidate_validation_cutoffs(timestamps, horizon=14, source="m6") == [
        86
    ]
    assert benchmark.shared_candidate_validation_cutoffs(list(range(62)), horizon=28) == [34]
    assert benchmark.shared_candidate_validation_cutoffs(list(range(20)), horizon=14) == []


def test_forecasting_benchmark_non_m_auto_requires_origin_consistency():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "forecasting_library_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "forecasting_library_benchmark_origin_consistency",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    benchmark = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = benchmark
    spec.loader.exec_module(benchmark)

    guarded = benchmark.lag_origin_consistency_guard(
        "cartoboost_auto_forecast",
        source="synthetic",
        inner_scores={
            "cartoboost_lag": [1.0, 1.0, 1.0],
            "cartoboost_auto_forecast": [0.8, 1.1, 0.7],
        },
    )
    assert guarded is not None
    assert guarded["losing_origin_count"] == 1
    assert guarded["min_relative_gain_vs_lag"] == pytest.approx(-0.1)

    assert (
        benchmark.lag_origin_consistency_guard(
            "cartoboost_auto_forecast",
            source="synthetic",
            inner_scores={
                "cartoboost_lag": [1.0, 1.0, 1.0],
                "cartoboost_auto_forecast": [0.8, 0.9, 0.7],
            },
        )
        is None
    )
    assert (
        benchmark.lag_origin_consistency_guard(
            "cartoboost_auto_forecast",
            source="m4",
            inner_scores={
                "cartoboost_lag": [1.0, 1.0, 1.0],
                "cartoboost_auto_forecast": [0.8, 1.1, 0.7],
            },
        )
        is None
    )


def test_forecasting_benchmark_non_m_skips_disqualified_raw_auto(monkeypatch):
    pl = pytest.importorskip("polars")
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "forecasting_library_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "forecasting_library_benchmark_raw_auto_short_circuit",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    benchmark = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = benchmark
    spec.loader.exec_module(benchmark)

    calls: list[list[str]] = []
    dates = [datetime(2026, 1, 1) + timedelta(days=day) for day in range(20)]
    train = pl.DataFrame(
        {
            "lane_id": ["A"] * len(dates),
            "date": dates,
            "loads": [1.0] * len(dates),
            "pickup_zone": [1.0] * len(dates),
            "dropoff_zone": [2.0] * len(dates),
            "distance_miles": [3.0] * len(dates),
            "airport_lane": [0.0] * len(dates),
            "pickup_borough_code": [1.0] * len(dates),
        }
    ).with_columns(pl.col("date").cast(pl.Datetime("us")))

    def fake_roster(inner_train, horizon, *, model_names, **_kwargs):
        calls.append(list(model_names))
        next_timestamp = inner_train.select(pl.col("date").max()).item() + timedelta(days=1)
        columns = {
            "series_id": ["A"],
            "timestamp": [next_timestamp],
            "horizon": [1],
            "cartoboost_lag": [2.0],
        }
        if "cartoboost_auto_forecast" in model_names:
            columns["cartoboost_auto_forecast"] = [2.1]
        return (
            pl.DataFrame(columns).with_columns(pl.col("timestamp").cast(pl.Datetime("us"))),
            {"models": {}},
        )

    monkeypatch.setattr(benchmark, "candidate_selection_forecast_roster", fake_roster)
    monkeypatch.setattr(
        benchmark,
        "add_shared_candidate_columns",
        lambda _train, _horizon, **kwargs: kwargs["predictions"],
    )

    scores = benchmark.shared_candidate_validation_scores(
        train,
        1,
        season_length=7,
        source="synthetic",
        cartoboost_config={},
        model_names=["cartoboost_lag", "cartoboost_auto_forecast"],
    )

    assert calls[0] == ["cartoboost_lag", "cartoboost_auto_forecast"]
    assert calls[1:] == [["cartoboost_lag"]]
    assert len(scores["cartoboost_lag"]) == 2
    assert len(scores["cartoboost_auto_forecast"]) == 2
    assert scores["cartoboost_auto_forecast"][1:] == pytest.approx([1.03])


def test_forecasting_benchmark_non_m_lag_dominance_stops_inner_validation(monkeypatch):
    pl = pytest.importorskip("polars")
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "forecasting_library_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "forecasting_library_benchmark_lag_dominance_stop",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    benchmark = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = benchmark
    spec.loader.exec_module(benchmark)

    assert benchmark.non_m_lag_dominates_origin(
        source="synthetic",
        origin_losses={
            "cartoboost_lag": 1.0,
            "cartoboost_auto_forecast": 1.25,
            "shared_seasonal_base": 1.30,
        },
    )
    assert not benchmark.non_m_lag_dominates_origin(
        source="synthetic",
        origin_losses={
            "cartoboost_lag": 1.0,
            "cartoboost_auto_forecast": 1.10,
            "shared_seasonal_base": 1.30,
        },
    )
    assert not benchmark.non_m_lag_dominates_origin(
        source="m4",
        origin_losses={
            "cartoboost_lag": 1.0,
            "cartoboost_auto_forecast": 1.25,
            "shared_seasonal_base": 1.30,
        },
    )

    calls = 0
    dates = [datetime(2026, 1, 1) + timedelta(days=day) for day in range(20)]
    train = pl.DataFrame(
        {
            "lane_id": ["A"] * len(dates),
            "date": dates,
            "loads": [0.0] * len(dates),
            "pickup_zone": [1.0] * len(dates),
            "dropoff_zone": [2.0] * len(dates),
            "distance_miles": [3.0] * len(dates),
            "airport_lane": [0.0] * len(dates),
            "pickup_borough_code": [1.0] * len(dates),
        }
    ).with_columns(pl.col("date").cast(pl.Datetime("us")))

    def fake_roster(inner_train, horizon, *, model_names, **_kwargs):
        nonlocal calls
        calls += 1
        next_timestamp = inner_train.select(pl.col("date").max()).item() + timedelta(days=1)
        columns = {
            "series_id": ["A"],
            "timestamp": [next_timestamp],
            "horizon": [1],
            "cartoboost_lag": [1.0],
            "cartoboost_auto_forecast": [1.30],
        }
        return (
            pl.DataFrame(columns).with_columns(pl.col("timestamp").cast(pl.Datetime("us"))),
            {"models": {}},
        )

    monkeypatch.setattr(benchmark, "candidate_selection_forecast_roster", fake_roster)
    monkeypatch.setattr(
        benchmark,
        "add_shared_candidate_columns",
        lambda _train, _horizon, **kwargs: kwargs["predictions"],
    )

    scores = benchmark.shared_candidate_validation_scores(
        train,
        1,
        season_length=7,
        source="synthetic",
        cartoboost_config={},
        model_names=["cartoboost_lag", "cartoboost_auto_forecast"],
    )

    assert calls == 1
    assert len(scores["cartoboost_lag"]) == 1


def test_forecasting_benchmark_validation_scores_reuse_cutoff_cache(monkeypatch):
    pl = pytest.importorskip("polars")
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "forecasting_library_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "forecasting_library_benchmark_validation_cache",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    benchmark = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = benchmark
    spec.loader.exec_module(benchmark)

    calls = 0
    dates = [datetime(2026, 1, 1) + timedelta(days=day) for day in range(20)]
    train = pl.DataFrame(
        {
            "lane_id": ["A"] * len(dates),
            "date": dates,
            "loads": [1.0] * len(dates),
            "pickup_zone": [1.0] * len(dates),
            "dropoff_zone": [2.0] * len(dates),
            "distance_miles": [3.0] * len(dates),
            "airport_lane": [0.0] * len(dates),
            "pickup_borough_code": [1.0] * len(dates),
        }
    ).with_columns(pl.col("date").cast(pl.Datetime("us")))

    def fake_roster(inner_train, horizon, *, model_names, **_kwargs):
        nonlocal calls
        calls += 1
        next_timestamp = inner_train.select(pl.col("date").max()).item() + timedelta(days=1)
        columns = {
            "series_id": ["A"],
            "timestamp": [next_timestamp],
            "horizon": [1],
            "cartoboost_lag": [1.0],
        }
        if "cartoboost_auto_forecast" in model_names:
            columns["cartoboost_auto_forecast"] = [1.0]
        return (
            pl.DataFrame(columns).with_columns(pl.col("timestamp").cast(pl.Datetime("us"))),
            {"models": {}},
        )

    monkeypatch.setattr(benchmark, "candidate_selection_forecast_roster", fake_roster)
    cache: dict[object, dict[str, float]] = {}

    first = benchmark.shared_candidate_validation_scores(
        train,
        1,
        season_length=7,
        source="synthetic",
        cartoboost_config={},
        model_names=["cartoboost_lag", "cartoboost_auto_forecast"],
        validation_cache=cache,
    )
    first_calls = calls
    second = benchmark.shared_candidate_validation_scores(
        train,
        1,
        season_length=7,
        source="synthetic",
        cartoboost_config={},
        model_names=["cartoboost_lag", "cartoboost_auto_forecast"],
        validation_cache=cache,
    )

    assert second == first
    assert calls == first_calls
    assert benchmark.count_validation_cache_hits(cache) == pytest.approx(2.0)
    assert benchmark.count_validation_cache_misses(cache) == pytest.approx(2.0)


def test_forecasting_benchmark_non_m_outer_raw_auto_is_lazy(monkeypatch):
    pl = pytest.importorskip("polars")
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "forecasting_library_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "forecasting_library_benchmark_lazy_outer_auto",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    benchmark = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = benchmark
    spec.loader.exec_module(benchmark)

    timestamp = datetime(2026, 1, 3)
    calls: list[str] = []

    def fake_raw_forecast(*_args, prediction_col: str, **_kwargs):
        calls.append(prediction_col)
        return (
            pl.DataFrame(
                {
                    "series_id": ["A"],
                    "timestamp": [timestamp],
                    "horizon": [1],
                    prediction_col: [1.0],
                }
            ).with_columns(pl.col("timestamp").cast(pl.Datetime("us"))),
            {"total_seconds": 0.01},
        )

    monkeypatch.setattr(benchmark, "cartoboost_raw_forecast", fake_raw_forecast)

    predictions, timing = benchmark.forecast_model_roster(
        pl.DataFrame(
            {
                "lane_id": ["A", "A"],
                "date": [datetime(2026, 1, 1), datetime(2026, 1, 2)],
                "loads": [1.0, 2.0],
            }
        ).with_columns(pl.col("date").cast(pl.Datetime("us"))),
        1,
        season_length=7,
        cartoboost_config={},
        model_names=["cartoboost_lag", "cartoboost_auto_forecast"],
        source="synthetic",
        skip_non_m_raw_auto_candidate=True,
    )

    assert calls == ["cartoboost_lag"]
    assert predictions["cartoboost_auto_forecast"].to_list() == [0.0]
    assert (
        timing["models"]["cartoboost_auto_forecast"]["selector_mode"]
        == "non_m_outer_lazy_raw_auto_candidate"
    )


def test_forecasting_benchmark_materializes_lazy_raw_auto_when_selected(monkeypatch):
    pl = pytest.importorskip("polars")
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "forecasting_library_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "forecasting_library_benchmark_lazy_outer_auto_materialize",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    benchmark = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = benchmark
    spec.loader.exec_module(benchmark)

    start = datetime(2026, 1, 1)
    table = pl.DataFrame(
        {
            "lane_id": ["A", "A", "A"],
            "date": [start, start + timedelta(days=1), start + timedelta(days=2)],
            "loads": [1.0, 1.0, 2.0],
            "pickup_zone": [1.0, 1.0, 1.0],
            "dropoff_zone": [2.0, 2.0, 2.0],
            "distance_miles": [3.0, 3.0, 3.0],
            "airport_lane": [0.0, 0.0, 0.0],
            "pickup_borough_code": [1.0, 1.0, 1.0],
        }
    ).with_columns(pl.col("date").cast(pl.Datetime("us")))
    prediction_timestamp = start + timedelta(days=2)

    def frame(column: str, value: float):
        return pl.DataFrame(
            {
                "series_id": ["A"],
                "timestamp": [prediction_timestamp],
                "horizon": [1],
                column: [value],
            }
        ).with_columns(pl.col("timestamp").cast(pl.Datetime("us")))

    def fake_roster(*_args, **_kwargs):
        return (
            frame("cartoboost_lag", 1.0).join(
                frame("cartoboost_auto_forecast", 0.0),
                on=["series_id", "timestamp", "horizon"],
            ),
            {
                "models": {
                    "cartoboost_lag": {"total_seconds": 0.01},
                    "cartoboost_auto_forecast": {
                        "selector_mode": "non_m_outer_lazy_raw_auto_candidate",
                        "total_seconds": 0.0,
                    },
                }
            },
        )

    def fake_selection(_train, _horizon, *, raw_predictions, **_kwargs):
        return raw_predictions, {
            "calibration_seconds": 0.0,
            "inner_origin_count": 1.0,
            "selected_candidates": {
                "cartoboost_lag": "cartoboost_lag",
                "cartoboost_auto_forecast": "cartoboost_auto_forecast",
            },
        }

    def fake_auto(*_args, prediction_col: str, **_kwargs):
        return frame(prediction_col, 2.0), {"total_seconds": 0.02}

    monkeypatch.setattr(benchmark, "forecast_model_roster", fake_roster)
    monkeypatch.setattr(benchmark, "apply_shared_candidate_selection", fake_selection)
    monkeypatch.setattr(benchmark, "cartoboost_forecast", fake_auto)

    metrics, _quality, timing, _scored = benchmark.score_models(
        table,
        horizon=1,
        season_length=7,
        cartoboost_config={},
        model_names=["cartoboost_lag", "cartoboost_auto_forecast"],
        source="synthetic",
        candidate_selection=True,
    )

    assert metrics["cartoboost_auto_forecast"]["rmse"] == pytest.approx(0.0)
    assert metrics["cartoboost_lag"]["rmse"] == pytest.approx(1.0)
    assert (
        timing["models"]["cartoboost_auto_forecast"]["selector_mode"]
        == "non_m_outer_lazy_raw_auto_selected"
    )


def test_forecasting_benchmark_non_m_inner_selection_skips_raw_auto(monkeypatch):
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "forecasting_library_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "forecasting_library_benchmark_lazy_inner_auto",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    benchmark = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = benchmark
    spec.loader.exec_module(benchmark)

    calls: list[dict[str, object]] = []

    def fake_forecast_model_roster(*_args, **kwargs):
        calls.append(kwargs)
        return "predictions", {"models": {}}

    monkeypatch.setattr(benchmark, "forecast_model_roster", fake_forecast_model_roster)

    predictions, timing = benchmark.candidate_selection_forecast_roster(
        "train",
        7,
        season_length=7,
        cartoboost_config={},
        model_names=["cartoboost_lag", "cartoboost_auto_forecast"],
        source="synthetic",
    )

    assert predictions == "predictions"
    assert timing == {"models": {}}
    assert calls == [
        {
            "season_length": 7,
            "cartoboost_config": {},
            "model_names": ["cartoboost_lag", "cartoboost_auto_forecast"],
            "source": "synthetic",
            "skip_non_m_raw_auto_candidate": True,
        }
    ]


def test_forecasting_benchmark_m4_suite_keeps_group_order(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "forecasting_library_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "forecasting_library_benchmark_m4_serial_suite",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    benchmark = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = benchmark
    spec.loader.exec_module(benchmark)

    groups = ["Hourly", "Daily", "Weekly"]
    monkeypatch.setattr(benchmark, "M4_GROUPS", groups)
    monkeypatch.setattr(
        benchmark,
        "load_m4_fixture",
        lambda group_args: (
            group_args.m4_group,
            {
                "group": group_args.m4_group,
                "horizon": 1,
                "season_length": 1,
            },
        ),
    )
    monkeypatch.setattr(
        benchmark,
        "canonical_dataset_hash",
        lambda table: f"hash-{table}",
    )
    monkeypatch.setattr(
        benchmark,
        "score_models",
        lambda table, **_kwargs: (
            {
                "cartoboost_auto_forecast": {
                    "rmse": 1.0,
                    "mae": 1.0,
                    "wape": 1.0,
                },
                "cartoboost_lag": {
                    "rmse": 2.0,
                    "mae": 2.0,
                    "wape": 2.0,
                },
            },
            {"winner": "cartoboost_auto_forecast"},
            {"models": {}, "candidate_selection": {"calibration_seconds": 0.0}},
            None,
        ),
    )

    output = tmp_path / "m4_suite.json"
    args = types.SimpleNamespace(
        output=output,
        source="m4",
        m4_suite=True,
        m4_group="Hourly",
        m4_series_limit=96,
        model_roster="cartoboost",
        no_hyperopt=True,
        no_candidate_selection=False,
        seed=42,
    )

    assert benchmark.run_m4_suite(args, {"n_estimators": 1}, benchmark.perf_counter()) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["dataset"]["groups"] == groups
    assert set(payload["groups"]) == set(groups)
    assert set(payload["timing"]["groups"]) == set(groups)


def test_forecasting_m4_wrapper_runs_committed_suite(monkeypatch, tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "forecasting_m4.py"
    spec = importlib.util.spec_from_file_location(
        "forecasting_m4_wrapper",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    wrapper = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = wrapper
    spec.loader.exec_module(wrapper)

    captured: dict[str, list[str]] = {}

    def fake_call(cmd):
        captured["cmd"] = cmd
        return 0

    output = tmp_path / "m4.json"
    monkeypatch.setattr(wrapper.subprocess, "call", fake_call)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "forecasting_m4.py",
            "--committed",
            "--no-hyperopt",
            "--output",
            str(output),
        ],
    )

    assert wrapper.main() == 0
    cmd = captured["cmd"]
    assert "--m4-suite-workers" not in cmd
    assert "--m4-suite" in cmd
    assert cmd[cmd.index("--output") + 1] == str(output)


def test_forecasting_benchmark_metrics_include_r2():
    pl = pytest.importorskip("polars")
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "forecasting_library_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "forecasting_library_benchmark_metrics",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    benchmark = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = benchmark
    spec.loader.exec_module(benchmark)

    train = pl.DataFrame(
        {
            "lane_id": ["a"] * 4,
            "date": [date(2026, 1, day) for day in range(1, 5)],
            "loads": [1.0, 2.0, 3.0, 4.0],
        }
    )
    scored = pl.DataFrame(
        {
            "series_id": ["a", "a", "a"],
            "timestamp": [
                datetime(2026, 1, 5),
                datetime(2026, 1, 6),
                datetime(2026, 1, 7),
            ],
            "horizon": [1, 2, 3],
            "actual": [1.0, 2.0, 3.0],
            "forecast": [1.0, 2.0, 4.0],
        }
    ).with_columns(pl.col("timestamp").cast(pl.Datetime("us")))

    metrics = benchmark.evaluate_metrics(scored, "forecast", train, season_length=1)

    assert metrics["rmse"] == pytest.approx((1.0 / 3.0) ** 0.5)
    assert metrics["r2"] == pytest.approx(0.5)


def test_forecasting_benchmark_r2_constant_target_is_finite():
    pl = pytest.importorskip("polars")
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "forecasting_library_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "forecasting_library_benchmark_constant_r2",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    benchmark = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = benchmark
    spec.loader.exec_module(benchmark)

    train = pl.DataFrame(
        {
            "lane_id": ["a"] * 4,
            "date": [date(2026, 1, day) for day in range(1, 5)],
            "loads": [2.0, 2.0, 2.0, 2.0],
        }
    )
    scored = pl.DataFrame(
        {
            "series_id": ["a", "a"],
            "timestamp": [datetime(2026, 1, 5), datetime(2026, 1, 6)],
            "horizon": [1, 2],
            "actual": [2.0, 2.0],
            "perfect": [2.0, 2.0],
            "miss": [1.0, 3.0],
        }
    ).with_columns(pl.col("timestamp").cast(pl.Datetime("us")))

    perfect = benchmark.evaluate_metrics(scored, "perfect", train, season_length=1)
    miss = benchmark.evaluate_metrics(scored, "miss", train, season_length=1)

    assert perfect["r2"] == pytest.approx(1.0)
    assert miss["r2"] == pytest.approx(0.0)


def test_forecasting_benchmark_robust_selector_prefers_simple_close_candidate():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "forecasting_library_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "forecasting_library_benchmark_robust_selector",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    benchmark = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = benchmark
    spec.loader.exec_module(benchmark)

    assert benchmark.candidate_selection_model_names(
        [
            "cartoboost_lag",
            "cartoboost_auto_forecast",
            "statsforecast_autoarima",
        ]
    ) == ["cartoboost_lag", "cartoboost_auto_forecast"]
    assert (
        benchmark.candidate_selection_model_names(["cartoboost_lag", "statsforecast_autoarima"])
        == []
    )

    selected = benchmark.robust_candidate_choice(
        {
            "cartoboost_auto_forecast": 1.00,
            "shared_seasonal_base": 1.03,
            "shared_calendar_dom": 0.99,
        }
    )

    assert selected == "shared_seasonal_base"
    assert (
        benchmark.candidate_choice_for_source(
            {
                "cartoboost_auto_forecast": 1.00,
                "shared_seasonal_base": 1.03,
                "shared_calendar_dom": 0.99,
            },
            source="m5",
        )
        == "shared_calendar_dom"
    )
    assert (
        benchmark.candidate_choice_for_source(
            {
                "shared_calendar_phase14": 0.7025,
                "shared_m5_phase14_total_reconciled_020": 0.7194,
                "shared_m5_phase14_total_reconciled_035": 0.7142,
                "shared_m5_phase14_total_reconciled_050": 0.7096,
                "shared_m5_wrmsse_autostats_blend": 0.7091,
                "shared_m5_point_autostats_phase14_blend": 0.7187,
            },
            source="m5",
        )
        == "shared_m5_phase14_total_reconciled_050"
    )
    assert (
        benchmark.candidate_choice_for_source(
            {
                "cartoboost_lag": 3.7045505738956943,
                "shared_m5_point_autostats_phase14_blend": 3.640384191677357,
            },
            source="m5",
            inner_origin_count=1,
        )
        == "cartoboost_lag"
    )
    assert (
        benchmark.candidate_choice_for_source(
            {
                "cartoboost_auto_forecast": 0.1996633147694055,
                "shared_calendar_phase14": 0.19966331430819997,
                "cartoboost_lag": 0.19966331443943974,
                "shared_m6_market_neutral_zero": 0.19966331426201722,
            },
            source="m6",
        )
        == "shared_m6_market_neutral_zero"
    )
    assert (
        benchmark.candidate_choice_for_source(
            {
                "cartoboost_auto_forecast": 1.00,
                "shared_seasonal_base": 1.03,
                "shared_calendar_dom": 0.99,
            },
            source="synthetic",
        )
        == "shared_seasonal_base"
    )
    assert benchmark.native_auto_raw_candidate_is_confident(
        {
            "cartoboost_auto_forecast": {
                "selected_candidate": "cartoboost_raw",
                "inner_raw_relative_rmse_gain": 0.50,
            }
        }
    )
    assert not benchmark.native_auto_raw_candidate_is_confident(
        {
            "cartoboost_auto_forecast": {
                "selected_candidate": "cartoboost_raw",
                "inner_raw_relative_rmse_gain": 0.49,
            }
        }
    )
    assert not benchmark.native_auto_raw_candidate_is_confident(
        {
            "cartoboost_auto_forecast": {
                "selected_candidate": "cartoboost_residual_blend",
                "inner_raw_relative_rmse_gain": 0.90,
            }
        }
    )


def test_forecasting_benchmark_keeps_confident_native_raw_auto(monkeypatch):
    pl = pytest.importorskip("polars")
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "forecasting_library_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "forecasting_library_benchmark_native_auto_confidence",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    benchmark = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = benchmark
    spec.loader.exec_module(benchmark)

    start = datetime(2026, 1, 1)
    train = pl.DataFrame(
        {
            "lane_id": ["a"] * 5,
            "date": [start + timedelta(days=offset) for offset in range(5)],
            "loads": [10.0, 11.0, 12.0, 13.0, 14.0],
            "pickup_zone": [1] * 5,
            "dropoff_zone": [2] * 5,
            "distance_miles": [3.5] * 5,
            "airport_lane": [0] * 5,
            "pickup_borough_code": [4] * 5,
        }
    ).with_columns(pl.col("date").cast(pl.Datetime("us")))
    raw_predictions = pl.DataFrame(
        {
            "series_id": ["a"],
            "timestamp": [start + timedelta(days=5)],
            "horizon": [1],
            "cartoboost_lag": [1.0],
            "cartoboost_auto_forecast": [2.0],
        }
    ).with_columns(pl.col("timestamp").cast(pl.Datetime("us")))

    def fake_inner_scores(*_args, **_kwargs):
        return {
            "cartoboost_lag": [0.4, 0.4],
            "cartoboost_auto_forecast": [0.5, 0.5],
        }

    monkeypatch.setattr(benchmark, "shared_candidate_validation_scores", fake_inner_scores)

    selected, timing = benchmark.apply_shared_candidate_selection(
        train,
        1,
        season_length=7,
        source="synthetic",
        raw_predictions=raw_predictions,
        model_timing={
            "cartoboost_auto_forecast": {
                "selected_candidate": "cartoboost_raw",
                "inner_raw_relative_rmse_gain": 0.75,
            }
        },
        cartoboost_config={
            "n_estimators": 1,
            "learning_rate": 0.06,
            "max_depth": 2,
            "min_samples_leaf": 1,
        },
        model_names=["cartoboost_lag", "cartoboost_auto_forecast"],
    )

    assert timing["native_auto_raw_keep"] is True
    assert timing["selected_candidates"]["cartoboost_auto_forecast"] == "cartoboost_lag"
    assert selected["cartoboost_auto_forecast"].to_list() == [1.0]


def test_forecasting_benchmark_autostats_candidate_targets_m4_and_m5():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "forecasting_library_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "forecasting_library_benchmark_autostats_gate",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    benchmark = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = benchmark
    spec.loader.exec_module(benchmark)

    assert benchmark.include_autostats_candidate(source="m4", season_length=12, horizon=18)
    assert benchmark.include_autostats_candidate(source="m4", season_length=4, horizon=8)
    assert not benchmark.include_autostats_candidate(source="m4", season_length=24, horizon=48)
    assert benchmark.include_autostats_candidate(source="m5", season_length=1, horizon=28)
    assert "shared_m5_phase14_total_reconciled_035" in benchmark.selectable_candidate_names(
        "cartoboost_auto_forecast",
        source="m5",
    )
    assert "shared_m5_point_autostats_phase14_blend" in benchmark.selectable_candidate_names(
        "cartoboost_auto_forecast",
        source="m5",
    )
    assert "shared_m5_wrmsse_autostats_blend" in benchmark.selectable_candidate_names(
        "cartoboost_auto_forecast",
        source="m5",
    )
    assert "shared_m5_total_reconciled_auto" in benchmark.selectable_candidate_names(
        "cartoboost_auto_forecast",
        source="m5",
    )
    assert "shared_m5_state_reconciled_auto" not in benchmark.selectable_candidate_names(
        "cartoboost_auto_forecast",
        source="m5",
    )
    assert "shared_m5_store_reconciled_auto" not in benchmark.selectable_candidate_names(
        "cartoboost_auto_forecast",
        source="m5",
    )
    assert (
        benchmark.candidate_choice_for_source(
            {
                "shared_m5_phase14_total_reconciled_035": 1.0,
                "shared_m5_point_autostats_phase14_blend": 1.015,
            },
            source="m5",
        )
        == "shared_m5_point_autostats_phase14_blend"
    )
    assert (
        benchmark.candidate_choice_for_source(
            {
                "shared_m5_phase14_total_reconciled_035": 1.0,
                "shared_m5_point_autostats_phase14_blend": 1.016,
            },
            source="m5",
        )
        == "shared_m5_phase14_total_reconciled_035"
    )
    assert (
        benchmark.candidate_choice_for_source(
            {
                "shared_m5_phase14_total_reconciled_035": 1.0,
                "shared_m5_wrmsse_autostats_blend": 1.0005,
            },
            source="m5",
            inner_origin_count=1,
        )
        == "shared_m5_wrmsse_autostats_blend"
    )
    assert (
        benchmark.candidate_choice_for_source(
            {
                "shared_m5_phase14_total_reconciled_035": 1.0,
                "shared_m5_wrmsse_autostats_blend": 1.0005,
            },
            source="m5",
            inner_origin_count=3,
        )
        == "shared_m5_phase14_total_reconciled_035"
    )


def test_forecasting_benchmark_shared_candidate_helpers_preserve_outputs():
    pl = pytest.importorskip("polars")
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "forecasting_library_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "forecasting_library_benchmark_shared_candidate_helpers",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    benchmark = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = benchmark
    spec.loader.exec_module(benchmark)

    start = datetime(2026, 1, 1)
    dates = [start + timedelta(days=offset) for offset in range(8)]
    train = pl.DataFrame(
        {
            "lane_id": ["a"] * 8 + ["b"] * 8,
            "date": [*dates, *dates],
            "loads": [*range(1, 9), *range(11, 19)],
            "pickup_zone": [1] * 16,
            "dropoff_zone": [2] * 16,
            "distance_miles": [3.5] * 16,
            "airport_lane": [0] * 16,
            "pickup_borough_code": [4] * 16,
        }
    ).with_columns(pl.col("date").cast(pl.Datetime("us")))

    seasonal = benchmark.seasonal_naive_forecast_frame(
        train,
        2,
        season_length=7,
        prediction_col="prediction",
    )
    calendar = benchmark.calendar_profile_forecast_frame(
        train,
        2,
        prediction_col="prediction",
        mode="phase14",
    )
    trend = benchmark.trend_forecast_frame(
        train,
        2,
        season_length=7,
        prediction_col="prediction",
        mode="half_drift",
    )

    assert list(seasonal.select("series_id", "horizon", "prediction").iter_rows()) == [
        ("a", 1, 2.0),
        ("b", 1, 12.0),
        ("a", 2, 3.0),
        ("b", 2, 13.0),
    ]
    assert list(calendar.select("series_id", "horizon", "prediction").iter_rows()) == [
        ("a", 1, 2.0),
        ("b", 1, 12.0),
        ("a", 2, 3.0),
        ("b", 2, 13.0),
    ]
    assert list(trend.select("series_id", "horizon", "prediction").iter_rows()) == [
        ("a", 1, pytest.approx(8.5)),
        ("b", 1, pytest.approx(18.5)),
        ("a", 2, pytest.approx(8.875)),
        ("b", 2, pytest.approx(18.875)),
    ]


def test_forecasting_benchmark_future_feature_builder_preserves_outputs():
    pl = pytest.importorskip("polars")
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "forecasting_library_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "forecasting_library_benchmark_future_feature_builder",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    benchmark = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = benchmark
    spec.loader.exec_module(benchmark)

    start = datetime(2026, 1, 1)
    history_dates = [start + timedelta(days=offset) for offset in range(8)]
    history = pl.DataFrame(
        {
            "lane_id": ["a"] * 8 + ["b"] * 8,
            "date": [*history_dates, *history_dates],
            "loads": [*range(1, 9), *range(11, 19)],
            "pickup_zone": [1] * 8 + [9] * 8,
            "dropoff_zone": [2] * 8 + [8] * 8,
            "distance_miles": [3.5] * 8 + [7.5] * 8,
            "airport_lane": [0] * 16,
            "pickup_borough_code": [4] * 8 + [6] * 8,
        }
    ).with_columns(pl.col("date").cast(pl.Datetime("us")))
    future = pl.DataFrame(
        {
            "lane_id": ["a", "b"],
            "date": [start + timedelta(days=8), start + timedelta(days=8)],
            "pickup_zone": [1, 9],
            "dropoff_zone": [2, 8],
            "distance_miles": [3.5, 7.5],
            "airport_lane": [0, 0],
            "pickup_borough_code": [4, 6],
        }
    ).with_columns(pl.col("date").cast(pl.Datetime("us")))

    features = benchmark.build_future_features(history, future, season_length=4).sort("lane_id")

    rows = features.select(
        "lane_id",
        "loads_lag_1",
        "loads_lag_3",
        "loads_lag_4",
        "loads_lag_8",
        "loads_roll_7",
        "loads_roll_4",
        "loads_delta_lag_3",
        "loads_delta_lag_4",
        "loads_roll_trend_4",
        "date_dayofweek",
        "date_day",
        "date_dayofyear",
        "date_month",
        "date_elapsed_days",
    ).iter_rows(named=True)
    assert list(rows) == [
        {
            "lane_id": "a",
            "loads_lag_1": 8.0,
            "loads_lag_3": 6.0,
            "loads_lag_4": 5.0,
            "loads_lag_8": 1.0,
            "loads_roll_7": pytest.approx(5.0),
            "loads_roll_4": pytest.approx(6.5),
            "loads_delta_lag_3": 2.0,
            "loads_delta_lag_4": 3.0,
            "loads_roll_trend_4": pytest.approx(4.0),
            "date_dayofweek": 5.0,
            "date_day": 9.0,
            "date_dayofyear": 9.0,
            "date_month": 1.0,
            "date_elapsed_days": 8.0,
        },
        {
            "lane_id": "b",
            "loads_lag_1": 18.0,
            "loads_lag_3": 16.0,
            "loads_lag_4": 15.0,
            "loads_lag_8": 11.0,
            "loads_roll_7": pytest.approx(15.0),
            "loads_roll_4": pytest.approx(16.5),
            "loads_delta_lag_3": 2.0,
            "loads_delta_lag_4": 3.0,
            "loads_roll_trend_4": pytest.approx(4.0),
            "date_dayofweek": 5.0,
            "date_day": 9.0,
            "date_dayofyear": 9.0,
            "date_month": 1.0,
            "date_elapsed_days": 8.0,
        },
    ]


def test_forecasting_benchmark_m5_selector_uses_raw_auto_without_nested_calibration(monkeypatch):
    pl = pytest.importorskip("polars")
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "forecasting_library_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "forecasting_library_benchmark_m5_selector_fast_path",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    benchmark = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = benchmark
    spec.loader.exec_module(benchmark)

    calls: list[tuple[str, str]] = []
    timestamp = datetime(2026, 1, 1)

    def forecast_frame(prediction_col: str, value: float):
        return pl.DataFrame(
            {
                "series_id": ["PU1->DO1"],
                "timestamp": [timestamp],
                "horizon": [1],
                prediction_col: [value],
            }
        ).with_columns(pl.col("timestamp").cast(pl.Datetime("us")))

    def fake_raw_forecast(*_args, prediction_col: str, **_kwargs):
        calls.append(("raw", prediction_col))
        return forecast_frame(prediction_col, 1.0), {"total_seconds": 0.01}

    def fake_auto_forecast(*_args, prediction_col: str, **_kwargs):
        calls.append(("auto", prediction_col))
        return forecast_frame(prediction_col, 2.0), {"total_seconds": 99.0}

    def fake_autostats_forecast(*_args, prediction_col: str, **_kwargs):
        calls.append(("autostats", prediction_col))
        return forecast_frame(prediction_col, 3.0), {"total_seconds": 0.02}

    monkeypatch.setattr(benchmark, "cartoboost_raw_forecast", fake_raw_forecast)
    monkeypatch.setattr(benchmark, "cartoboost_forecast", fake_auto_forecast)
    monkeypatch.setattr(benchmark, "cartoboost_autostats_forecast", fake_autostats_forecast)

    train = pl.DataFrame(
        {
            "lane_id": ["PU1->DO1"],
            "date": [timestamp],
            "loads": [1.0],
        }
    ).with_columns(pl.col("date").cast(pl.Datetime("us")))

    predictions, timing = benchmark.candidate_selection_forecast_roster(
        train,
        1,
        season_length=1,
        cartoboost_config={
            "n_estimators": 1,
            "learning_rate": 0.06,
            "max_depth": 2,
            "min_samples_leaf": 1,
        },
        model_names=["cartoboost_lag", "cartoboost_auto_forecast"],
        source="m5",
    )

    assert ("auto", "cartoboost_auto_forecast") not in calls
    assert ("raw", "cartoboost_lag") in calls
    assert ("raw", "cartoboost_auto_forecast") not in calls
    assert ("autostats", "cartoboost_autostats_bank") in calls
    assert "cartoboost_autostats_bank" in predictions.columns
    assert (
        timing["models"]["cartoboost_auto_forecast"]["selector_mode"]
        == "m5_inner_validation_skips_raw_auto"
    )


def test_forecasting_benchmark_outer_candidates_only_build_required_columns():
    pl = pytest.importorskip("polars")
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "forecasting_library_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "forecasting_library_benchmark_required_outer_columns",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    benchmark = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = benchmark
    spec.loader.exec_module(benchmark)

    timestamp = datetime(2026, 1, 8)
    train = pl.DataFrame(
        {
            "lane_id": ["PU1->DO1"] * 7,
            "date": [datetime(2026, 1, day) for day in range(1, 8)],
            "loads": [1.0, 2.0, 1.0, 3.0, 2.0, 4.0, 3.0],
        }
    ).with_columns(pl.col("date").cast(pl.Datetime("us")))
    predictions = pl.DataFrame(
        {
            "series_id": ["PU1->DO1"],
            "timestamp": [timestamp],
            "horizon": [1],
            "cartoboost_lag": [3.0],
            "cartoboost_auto_forecast": [2.0],
        }
    ).with_columns(pl.col("timestamp").cast(pl.Datetime("us")))

    selected = benchmark.add_shared_candidate_columns(
        train,
        1,
        season_length=7,
        predictions=predictions,
        source="m5",
        required_columns={"cartoboost_lag"},
    )

    assert selected.columns == predictions.columns


def test_forecasting_benchmark_m6_point_candidate_uses_raw_auto(monkeypatch):
    pl = pytest.importorskip("polars")
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "forecasting_library_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "forecasting_library_benchmark_m6_point_fast_path",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    benchmark = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = benchmark
    spec.loader.exec_module(benchmark)

    calls: list[tuple[str, str]] = []
    timestamp = datetime(2026, 1, 1)

    def forecast_frame(prediction_col: str, value: float):
        return pl.DataFrame(
            {
                "series_id": ["AAA"],
                "timestamp": [timestamp],
                "horizon": [1],
                prediction_col: [value],
            }
        ).with_columns(pl.col("timestamp").cast(pl.Datetime("us")))

    def fake_raw_forecast(*_args, prediction_col: str, **_kwargs):
        calls.append(("raw", prediction_col))
        return forecast_frame(prediction_col, 1.0), {"total_seconds": 0.01}

    def fake_auto_forecast(*_args, prediction_col: str, **_kwargs):
        calls.append(("auto", prediction_col))
        return forecast_frame(prediction_col, 2.0), {"total_seconds": 0.02}

    monkeypatch.setattr(benchmark, "cartoboost_raw_forecast", fake_raw_forecast)
    monkeypatch.setattr(benchmark, "cartoboost_forecast", fake_auto_forecast)

    train = pl.DataFrame(
        {
            "lane_id": ["AAA"],
            "date": [timestamp],
            "loads": [0.01],
        }
    ).with_columns(pl.col("date").cast(pl.Datetime("us")))

    predictions, timing = benchmark.forecast_model_roster(
        train,
        1,
        season_length=1,
        cartoboost_config={
            "n_estimators": 1,
            "learning_rate": 0.06,
            "max_depth": 2,
            "min_samples_leaf": 1,
        },
        model_names=["cartoboost_lag", "cartoboost_auto_forecast"],
        source="m6",
    )

    assert ("auto", "cartoboost_auto_forecast") not in calls
    assert ("raw", "cartoboost_auto_forecast") in calls
    assert ("auto", "cartoboost_m6_point_auto") not in calls
    assert ("raw", "cartoboost_m6_point_auto") not in calls
    assert "cartoboost_m6_point_auto" in predictions.columns
    assert (
        timing["models"]["cartoboost_auto_forecast"]["selector_mode"]
        == "m6_raw_auto_outer_no_nested_calibration"
    )


def test_forecasting_benchmark_m5_outer_selection_skips_raw_auto(monkeypatch):
    pl = pytest.importorskip("polars")
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "forecasting_library_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "forecasting_library_benchmark_m5_outer_selection_fast_path",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    benchmark = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = benchmark
    spec.loader.exec_module(benchmark)

    calls: list[tuple[str, str]] = []
    timestamp = datetime(2026, 1, 1)

    def forecast_frame(prediction_col: str, value: float):
        return pl.DataFrame(
            {
                "series_id": ["FOODS_1_CA_1"],
                "timestamp": [timestamp],
                "horizon": [1],
                prediction_col: [value],
            }
        ).with_columns(pl.col("timestamp").cast(pl.Datetime("us")))

    def fake_raw_forecast(*_args, prediction_col: str, **_kwargs):
        calls.append(("raw", prediction_col))
        return forecast_frame(prediction_col, 1.0), {"total_seconds": 0.01}

    def fake_auto_forecast(*_args, prediction_col: str, **_kwargs):
        calls.append(("auto", prediction_col))
        return forecast_frame(prediction_col, 2.0), {"total_seconds": 99.0}

    def fake_autostats_forecast(*_args, prediction_col: str, **_kwargs):
        calls.append(("autostats", prediction_col))
        return forecast_frame(prediction_col, 3.0), {"total_seconds": 0.02}

    monkeypatch.setattr(benchmark, "cartoboost_raw_forecast", fake_raw_forecast)
    monkeypatch.setattr(benchmark, "cartoboost_forecast", fake_auto_forecast)
    monkeypatch.setattr(benchmark, "cartoboost_autostats_forecast", fake_autostats_forecast)

    train = pl.DataFrame(
        {
            "lane_id": ["FOODS_1_CA_1"],
            "date": [timestamp],
            "loads": [1.0],
        }
    ).with_columns(pl.col("date").cast(pl.Datetime("us")))

    predictions, timing = benchmark.forecast_model_roster(
        train,
        1,
        season_length=1,
        cartoboost_config={
            "n_estimators": 1,
            "learning_rate": 0.06,
            "max_depth": 2,
            "min_samples_leaf": 1,
        },
        model_names=["cartoboost_lag", "cartoboost_auto_forecast"],
        source="m5",
        skip_m5_raw_auto_candidate=True,
    )

    assert ("raw", "cartoboost_lag") in calls
    assert ("auto", "cartoboost_auto_forecast") not in calls
    assert ("autostats", "cartoboost_autostats_bank") in calls
    assert "cartoboost_auto_forecast" in predictions.columns
    assert "cartoboost_autostats_bank" in predictions.columns
    assert predictions["cartoboost_auto_forecast"].to_list() == [3.0]
    assert (
        timing["models"]["cartoboost_auto_forecast"]["selector_mode"]
        == "m5_outer_maps_auto_to_autostats_candidate"
    )


def test_forecasting_benchmark_m6_outer_selection_skips_raw_auto(monkeypatch):
    pl = pytest.importorskip("polars")
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "forecasting_library_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "forecasting_library_benchmark_m6_outer_selection_fast_path",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    benchmark = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = benchmark
    spec.loader.exec_module(benchmark)

    calls: list[tuple[str, str]] = []
    timestamp = datetime(2026, 1, 1)

    def forecast_frame(prediction_col: str, value: float):
        return pl.DataFrame(
            {
                "series_id": ["AAA"],
                "timestamp": [timestamp],
                "horizon": [1],
                prediction_col: [value],
            }
        ).with_columns(pl.col("timestamp").cast(pl.Datetime("us")))

    def fake_raw_forecast(*_args, prediction_col: str, **_kwargs):
        calls.append(("raw", prediction_col))
        return forecast_frame(prediction_col, 1.0), {"total_seconds": 0.01}

    monkeypatch.setattr(benchmark, "cartoboost_raw_forecast", fake_raw_forecast)

    train = pl.DataFrame(
        {
            "lane_id": ["AAA"],
            "date": [timestamp],
            "loads": [0.01],
        }
    ).with_columns(pl.col("date").cast(pl.Datetime("us")))

    predictions, timing = benchmark.forecast_model_roster(
        train,
        1,
        season_length=1,
        cartoboost_config={
            "n_estimators": 1,
            "learning_rate": 0.06,
            "max_depth": 2,
            "min_samples_leaf": 1,
        },
        model_names=["cartoboost_lag", "cartoboost_auto_forecast"],
        source="m6",
        skip_m6_raw_auto_candidate=True,
    )

    assert ("raw", "cartoboost_lag") in calls
    assert ("raw", "cartoboost_auto_forecast") not in calls
    assert "cartoboost_auto_forecast" in predictions.columns
    assert "cartoboost_m6_point_auto" not in predictions.columns
    assert (
        timing["models"]["cartoboost_auto_forecast"]["selector_mode"]
        == "m6_outer_skips_raw_auto_candidate"
    )


def test_forecasting_benchmark_m4_inner_validation_skips_raw_auto(monkeypatch):
    pl = pytest.importorskip("polars")
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "forecasting_library_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "forecasting_library_benchmark_m4_inner_fast_path",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    benchmark = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = benchmark
    spec.loader.exec_module(benchmark)

    calls: list[tuple[str, str]] = []
    timestamp = datetime(2026, 1, 1)

    def forecast_frame(prediction_col: str, value: float):
        return pl.DataFrame(
            {
                "series_id": ["M4_Monthly_1"],
                "timestamp": [timestamp],
                "horizon": [1],
                prediction_col: [value],
            }
        ).with_columns(pl.col("timestamp").cast(pl.Datetime("us")))

    def fake_raw_forecast(*_args, prediction_col: str, **_kwargs):
        calls.append(("raw", prediction_col))
        return forecast_frame(prediction_col, 1.0), {"total_seconds": 0.01}

    monkeypatch.setattr(benchmark, "cartoboost_raw_forecast", fake_raw_forecast)

    train = pl.DataFrame(
        {
            "lane_id": ["M4_Monthly_1"],
            "date": [timestamp],
            "loads": [100.0],
        }
    ).with_columns(pl.col("date").cast(pl.Datetime("us")))

    predictions, timing = benchmark.candidate_selection_forecast_roster(
        train,
        1,
        season_length=12,
        cartoboost_config={
            "n_estimators": 1,
            "learning_rate": 0.06,
            "max_depth": 2,
            "min_samples_leaf": 1,
        },
        model_names=["cartoboost_lag", "cartoboost_auto_forecast"],
        source="m4",
    )

    assert ("raw", "cartoboost_lag") in calls
    assert ("raw", "cartoboost_auto_forecast") not in calls
    assert "cartoboost_lag" in predictions.columns
    assert (
        timing["models"]["cartoboost_auto_forecast"]["selector_mode"]
        == "m4_inner_validation_skips_raw_auto"
    )


def test_forecasting_benchmark_m6_inner_validation_skips_raw_auto(monkeypatch):
    pl = pytest.importorskip("polars")
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "forecasting_library_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "forecasting_library_benchmark_m6_inner_fast_path",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    benchmark = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = benchmark
    spec.loader.exec_module(benchmark)

    calls: list[tuple[str, str]] = []
    timestamp = datetime(2026, 1, 1)

    def forecast_frame(prediction_col: str, value: float):
        return pl.DataFrame(
            {
                "series_id": ["AAA"],
                "timestamp": [timestamp],
                "horizon": [1],
                prediction_col: [value],
            }
        ).with_columns(pl.col("timestamp").cast(pl.Datetime("us")))

    def fake_raw_forecast(*_args, prediction_col: str, **_kwargs):
        calls.append(("raw", prediction_col))
        return forecast_frame(prediction_col, 1.0), {"total_seconds": 0.01}

    monkeypatch.setattr(benchmark, "cartoboost_raw_forecast", fake_raw_forecast)

    train = pl.DataFrame(
        {
            "lane_id": ["AAA"],
            "date": [timestamp],
            "loads": [0.01],
        }
    ).with_columns(pl.col("date").cast(pl.Datetime("us")))

    predictions, timing = benchmark.candidate_selection_forecast_roster(
        train,
        1,
        season_length=1,
        cartoboost_config={
            "n_estimators": 1,
            "learning_rate": 0.06,
            "max_depth": 2,
            "min_samples_leaf": 1,
        },
        model_names=["cartoboost_lag", "cartoboost_auto_forecast"],
        source="m6",
    )

    assert ("raw", "cartoboost_lag") in calls
    assert ("raw", "cartoboost_auto_forecast") not in calls
    assert "cartoboost_lag" in predictions.columns
    assert (
        timing["models"]["cartoboost_auto_forecast"]["selector_mode"]
        == "m6_inner_validation_skips_raw_auto"
    )


def test_forecasting_benchmark_m5_inner_validation_skips_raw_auto(monkeypatch):
    pl = pytest.importorskip("polars")
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "forecasting_library_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "forecasting_library_benchmark_m5_inner_fast_path",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    benchmark = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = benchmark
    spec.loader.exec_module(benchmark)

    calls: list[tuple[str, str]] = []
    timestamp = datetime(2026, 1, 1)

    def forecast_frame(prediction_col: str, value: float):
        return pl.DataFrame(
            {
                "series_id": ["FOODS_1_CA_1"],
                "timestamp": [timestamp],
                "horizon": [1],
                prediction_col: [value],
            }
        ).with_columns(pl.col("timestamp").cast(pl.Datetime("us")))

    def fake_raw_forecast(*_args, prediction_col: str, **_kwargs):
        calls.append(("raw", prediction_col))
        return forecast_frame(prediction_col, 1.0), {"total_seconds": 0.01}

    def fake_autostats_forecast(*_args, prediction_col: str, **_kwargs):
        calls.append(("autostats", prediction_col))
        return forecast_frame(prediction_col, 2.0), {"total_seconds": 0.02}

    monkeypatch.setattr(benchmark, "cartoboost_raw_forecast", fake_raw_forecast)
    monkeypatch.setattr(benchmark, "cartoboost_autostats_forecast", fake_autostats_forecast)

    train = pl.DataFrame(
        {
            "lane_id": ["FOODS_1_CA_1"],
            "date": [timestamp],
            "loads": [1.0],
            "pickup_zone": [1.0],
            "dropoff_zone": [1.0],
            "distance_miles": [1.0],
            "airport_lane": [0.0],
            "pickup_borough_code": [1.0],
        }
    ).with_columns(pl.col("date").cast(pl.Datetime("us")))

    predictions, timing = benchmark.candidate_selection_forecast_roster(
        train,
        1,
        season_length=1,
        cartoboost_config={
            "n_estimators": 1,
            "learning_rate": 0.06,
            "max_depth": 2,
            "min_samples_leaf": 1,
        },
        model_names=["cartoboost_lag", "cartoboost_auto_forecast"],
        source="m5",
    )

    assert ("raw", "cartoboost_lag") in calls
    assert ("raw", "cartoboost_auto_forecast") not in calls
    assert ("autostats", "cartoboost_autostats_bank") in calls
    assert "cartoboost_auto_forecast" in predictions.columns
    assert "cartoboost_autostats_bank" in predictions.columns
    assert (
        timing["models"]["cartoboost_auto_forecast"]["selector_mode"]
        == "m5_inner_validation_skips_raw_auto"
    )


def test_forecasting_benchmark_static_covariates_are_source_gated():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "forecasting_library_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "forecasting_library_benchmark_covariate_gate",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    benchmark = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = benchmark
    spec.loader.exec_module(benchmark)

    base = {
        "n_estimators": 1,
        "learning_rate": 0.06,
        "max_depth": 2,
        "min_samples_leaf": 1,
    }

    synthetic = benchmark.cartoboost_source_config(base, source="synthetic")
    taxi = benchmark.cartoboost_source_config(base, source="nyc-taxi")
    m4 = benchmark.cartoboost_source_config(base, source="m4")
    m5 = benchmark.cartoboost_source_config(base, source="m5")
    m6 = benchmark.cartoboost_source_config(base, source="m6")

    assert synthetic["use_static_covariates"]
    assert synthetic["use_rich_calendar_features"]
    assert synthetic["use_native_rolling_stat_features"]
    assert not synthetic["use_native_partial_rolling_mean_features"]
    assert not synthetic["use_native_ewm_features"]
    assert synthetic["use_covariate_calendar_interactions"]
    assert taxi["use_static_covariates"]
    assert taxi["use_rich_calendar_features"]
    assert taxi["use_native_rolling_stat_features"]
    assert taxi["use_native_partial_rolling_mean_features"]
    assert not taxi["use_native_ewm_features"]
    assert taxi["use_covariate_calendar_interactions"]
    assert not m4["use_static_covariates"]
    assert not m4["use_rich_calendar_features"]
    assert not m4["use_native_rolling_stat_features"]
    assert not m4["use_native_partial_rolling_mean_features"]
    assert not m4["use_native_ewm_features"]
    assert not m4["use_covariate_calendar_interactions"]
    assert not m5["use_static_covariates"]
    assert not m5["use_rich_calendar_features"]
    assert not m5["use_native_rolling_stat_features"]
    assert not m5["use_native_partial_rolling_mean_features"]
    assert not m5["use_native_ewm_features"]
    assert not m5["use_covariate_calendar_interactions"]
    assert not m6["use_static_covariates"]
    assert not m6["use_rich_calendar_features"]
    assert not m6["use_native_rolling_stat_features"]
    assert not m6["use_native_partial_rolling_mean_features"]
    assert not m6["use_native_ewm_features"]
    assert not m6["use_covariate_calendar_interactions"]


def test_forecasting_benchmark_m4_lag_spine_targets_high_frequency_risk():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "forecasting_library_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "forecasting_library_benchmark_m4_lag_spine",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    benchmark = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = benchmark
    spec.loader.exec_module(benchmark)

    assert benchmark.m4_requires_lag_spine(season_length=24, horizon=48)
    assert benchmark.m4_requires_lag_spine(season_length=12, horizon=18)
    assert benchmark.m4_requires_lag_spine(season_length=1, horizon=13)
    assert not benchmark.m4_requires_lag_spine(season_length=1, horizon=6)
    assert not benchmark.m4_requires_lag_spine(season_length=4, horizon=8)
    assert not benchmark.requires_lag_spine(source="synthetic", season_length=7, horizon=14)
    assert benchmark.requires_lag_spine(source="m4", season_length=24, horizon=48)
    assert not benchmark.requires_lag_spine(source="m5", season_length=1, horizon=28)
    assert not benchmark.requires_lag_spine(source="m6", season_length=1, horizon=28)
    script = module_path.read_text(encoding="utf-8")
    assert 'source in {"synthetic", "m4"} and best_candidate == "cartoboost_lag"' not in script
    assert 'and best_candidate != "cartoboost_lag"' in script


def test_forecasting_benchmark_docs_match_committed_artifacts():
    repo_root = Path(__file__).resolve().parents[2]
    docs = (repo_root / "docs" / "benchmarks" / "forecasting.md").read_text(encoding="utf-8")
    artifacts = repo_root / "docs" / "assets" / "nyc_taxi_benchmarks"

    synthetic = json.loads((artifacts / "forecasting_overhaul_committed_suite.json").read_text())
    synthetic_quality = synthetic["aggregate_quality"]
    synthetic_lag = synthetic_quality["mean_rmse_ratio_to_problem_best"]["cartoboost_lag"]
    synthetic_auto = synthetic_quality["mean_rmse_ratio_to_problem_best"][
        "cartoboost_auto_forecast"
    ]
    assert f"Auto and lag tie at mean RMSE ratio {synthetic_auto:.6f}" in docs
    assert synthetic_auto == pytest.approx(synthetic_lag)
    assert synthetic_auto <= synthetic_lag

    m4 = json.loads((artifacts / "forecasting_overhaul_m4_committed.json").read_text())
    m4_quality = m4["aggregate_quality"]
    m4_lag = m4_quality["mean_rmse_ratio_to_problem_best"]["cartoboost_lag"]
    m4_auto = m4_quality["mean_rmse_ratio_to_problem_best"]["cartoboost_auto_forecast"]
    assert f"Auto wins/ties all 6 groups at mean RMSE ratio {m4_auto:.6f}" in docs
    assert f"| `cartoboost_lag` | 369.256653 | {m4_lag:.6f} |" in docs
    assert m4_auto < m4_lag

    m5 = json.loads((artifacts / "forecasting_overhaul_m5_committed.json").read_text())
    m5_auto = m5["metrics"]["cartoboost_auto_forecast"]
    m5_lag = m5["metrics"]["cartoboost_lag"]
    m5_wrmsse = m5["official_metrics"]["m5"]["model_scores"]["cartoboost_auto_forecast"]
    m5_auto_wrmsse = m5["official_metrics"]["m5"]["model_scores"]["cartoboost_auto_forecast"]
    assert (
        f"| `cartoboost_auto_forecast` | {m5_auto['rmse']:.6f} | "
        f"{m5_auto['mae']:.6f} | {m5_auto['wape']:.6f} | {m5_wrmsse:.6f} |"
    ) in docs
    assert m5_auto["rmse"] < m5_lag["rmse"]
    assert m5_auto_wrmsse < m5["official_metrics"]["m5"]["model_scores"]["cartoboost_lag"]

    m6 = json.loads((artifacts / "forecasting_overhaul_m6_committed.json").read_text())
    m6_auto = m6["metrics"]["cartoboost_auto_forecast"]
    m6_lag = m6["metrics"]["cartoboost_lag"]
    m6_rps = m6["official_metrics"]["m6"]["models"]
    assert (
        f"| `cartoboost_auto_forecast` | {m6_auto['rmse']:.6f} | "
        f"{m6_auto['mae']:.6f} | {m6_auto['wape']:.6f} | "
        f"{m6_rps['cartoboost_auto_forecast']['mean_rps']:.6f} |"
    ) in docs
    assert (
        "WAPE is reported but should not be treated as\n"
        "  the model-quality reason on signed returns"
    ) in docs
    assert m6_auto["rmse"] < m6_lag["rmse"]
    assert f"{m6_rps['cartoboost_lag']['mean_rps']:.6f}" in docs

    full_roster = json.loads(
        (artifacts / "forecasting_overhaul_committed_suite_full_roster.json").read_text()
    )
    full_roster_quality = full_roster["aggregate_quality"]
    full_winner = full_roster_quality["mean_rmse_ratio_ranking"][0]
    full_winner_ratio = full_roster_quality["mean_rmse_ratio_to_problem_best"][full_winner]
    full_auto_ratio = full_roster_quality["mean_rmse_ratio_to_problem_best"][
        "cartoboost_auto_forecast"
    ]
    assert f"`{full_winner}` remains first at mean RMSE ratio {full_winner_ratio:.6f}" in docs
    assert full_auto_ratio > full_winner_ratio

    generalization = json.loads(
        (artifacts / "forecasting_generalization_scalable_synthetic.json").read_text()
    )
    generalization_quality = generalization["aggregate_quality"]
    generalization_ratios = generalization_quality["mean_rmse_ratio_to_problem_best"]
    generalization_auto_ratio = generalization_ratios["cartoboost_auto_forecast"]
    generalization_lag_ratio = generalization_ratios["cartoboost_lag"]
    generalization_lightgbm_ratio = generalization_ratios["lightgbm_lag"]
    generalization_xgboost_ratio = generalization_ratios["xgboost_lag"]
    assert f"| `cartoboost_auto_forecast` | {generalization_auto_ratio:.6f} | 4 | 4 |" in docs
    assert f"| `cartoboost_lag` | {generalization_lag_ratio:.6f} | 4 | 4 |" in docs
    assert f"| `lightgbm_lag` | {generalization_lightgbm_ratio:.6f} | 0 | 3 |" in docs
    assert f"| `xgboost_lag` | {generalization_xgboost_ratio:.6f} | 0 | 0 |" in docs
    assert generalization_auto_ratio == pytest.approx(generalization_lag_ratio)
    assert generalization_auto_ratio < generalization_lightgbm_ratio
    assert generalization_auto_ratio < generalization_xgboost_ratio
    assert (
        f"The run completed in {generalization['timing']['total_seconds']:.3f} seconds "
        f"with peak RSS {generalization['resource_usage']['peak_rss_mb']:.3f} MB"
    ) in docs

    m5_sample = json.loads((artifacts / "forecasting_m5_full_roster_sample.json").read_text())
    m5_sample_winner = m5_sample["quality"]["winner"]
    m5_sample_auto = m5_sample["metrics"]["cartoboost_auto_forecast"]
    m5_sample_cartoboost_rmse = m5_sample["metrics"]["cartoboost_auto_forecast"]["rmse"]
    m5_sample_rmse_ranking = sorted(
        m5_sample["metrics"],
        key=lambda model: m5_sample["metrics"][model]["rmse"],
    )
    m5_sample_second_rmse_model = m5_sample_rmse_ranking[1]
    m5_sample_second_rmse = m5_sample["metrics"][m5_sample_second_rmse_model]["rmse"]
    m5_wrmsse = m5_sample["official_metrics"]["m5"]
    m5_wrmsse_winner = m5_wrmsse["ranking"][0]
    m5_wrmsse_winner_score = m5_wrmsse["model_scores"][m5_wrmsse_winner]
    m5_cartoboost_wrmsse = m5_wrmsse["model_scores"]["cartoboost_auto_forecast"]
    assert f"| Best RMSE | {m5_sample_cartoboost_rmse:.6f} |" in docs
    assert f"| CartoBoost MAE | {m5_sample_auto['mae']:.6f} |" in docs
    assert f"| CartoBoost WAPE | {m5_sample_auto['wape']:.6f} |" in docs
    assert f"| Best WRMSSE | `{m5_wrmsse_winner}`, {m5_wrmsse_winner_score:.6f} |" in docs
    assert f"| CartoBoost WRMSSE | {m5_cartoboost_wrmsse:.6f} |" in docs
    assert m5_sample_second_rmse < float("inf")
    assert m5_sample_winner == "cartoboost_auto_forecast"

    m6_full = json.loads((artifacts / "forecasting_m6_full.json").read_text())
    m6_full_winner = m6_full["quality"]["winner"]
    m6_full_auto = m6_full["metrics"]["cartoboost_auto_forecast"]
    m6_full_cartoboost_rmse = m6_full_auto["rmse"]
    m6_full_rmse_ranking = sorted(
        m6_full["metrics"],
        key=lambda model: m6_full["metrics"][model]["rmse"],
    )
    m6_full_second_rmse_model = m6_full_rmse_ranking[1]
    m6_full_second_rmse = m6_full["metrics"][m6_full_second_rmse_model]["rmse"]
    m6_full_rps = m6_full["official_metrics"]["m6"]
    m6_full_rps_winner = m6_full_rps["ranking"][0]
    m6_full_rps_winner_score = m6_full_rps["models"][m6_full_rps_winner]["mean_rps"]
    m6_full_auto_rps = m6_full_rps["models"]["cartoboost_auto_forecast"]["mean_rps"]
    assert (
        f"CartoBoost auto wins point quality: RMSE {m6_full_cartoboost_rmse:.6f}, "
        f"MAE {m6_full_auto['mae']:.6f}"
    ) not in docs
    assert (f"CartoBoost RMSE | {m6_full_cartoboost_rmse:.6f}") in docs
    assert (f"CartoBoost MAE | {m6_full_auto['mae']:.6f}") in docs
    assert (
        f"CartoBoost WAPE | {m6_full_auto['wape']:.6f}, reported as a signed-return diagnostic"
    ) in docs
    assert f"`{m6_full_second_rmse_model}`" in docs
    assert f"{m6_full_second_rmse:.6f}" in docs
    assert f"{m6_full_rps_winner_score:.6f}" in docs
    assert f"{m6_full_auto_rps:.6f}" in docs
    assert m6_full_winner == "cartoboost_auto_forecast"
    assert m6_full_rps_winner != "cartoboost_auto_forecast"


def test_forecasting_competition_readiness_catalog_is_documented():
    repo_root = Path(__file__).resolve().parents[2]
    catalog_path = (
        repo_root
        / "docs"
        / "assets"
        / "nyc_taxi_benchmarks"
        / "forecasting_competition_catalog.json"
    )
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    docs = (repo_root / "docs" / "benchmarks" / "forecasting-competition-readiness.md").read_text(
        encoding="utf-8"
    )
    benchmark_index = (repo_root / "docs" / "benchmarks" / "index.md").read_text(encoding="utf-8")
    sidebar = (repo_root / "sidebars.ts").read_text(encoding="utf-8")

    assert catalog["status"] == "catalog_only"
    assert "not CartoBoost quality evidence" in catalog["claim_boundary"]
    assert len(catalog["datasets"]) >= 6
    dataset_ids = {dataset["id"] for dataset in catalog["datasets"]}
    assert {
        "monash_tourism",
        "nn5_daily",
        "kaggle_store_sales",
        "kaggle_web_traffic",
        "gefcom2012_load",
        "kdd_cup_2018_air",
    }.issubset(dataset_ids)

    for dataset in catalog["datasets"]:
        assert dataset["source_url"].startswith("https://")
        assert dataset["adapter_status"] == "not_implemented"
        assert dataset["expected_metric"]
        assert dataset["ship_gate"]

    assert "Forecasting Competition Readiness" in benchmark_index
    assert "forecasting-competition-readiness" in sidebar
    assert "These datasets are adapter\ntargets, not quality claims" in docs
    assert "No dataset on this page is currently claimed as CartoBoost benchmark evidence" in docs


def test_committed_forecasting_artifacts_include_provenance_fields():
    repo_root = Path(__file__).resolve().parents[2]
    artifacts = repo_root / "docs" / "assets" / "nyc_taxi_benchmarks"
    names = [
        "forecasting_overhaul_committed_suite.json",
        "forecasting_overhaul_m4_committed.json",
        "forecasting_overhaul_m5_committed.json",
        "forecasting_overhaul_m6_committed.json",
    ]
    for name in names:
        artifact = json.loads((artifacts / name).read_text())
        assert artifact["git_commit"]
        assert artifact["dataset_hash"]
        assert "source_file_hashes" in artifact
        assert artifact["benchmark_integrity"]["no_hyperopt"] is True
        assert artifact["benchmark_integrity"]["seed"] == 42
        assert artifact["resource_usage"]["peak_rss_mb"] > 0.0
        assert artifact["resource_usage"]["process_cpu_seconds"] >= 0.0


def test_forecasting_benchmark_m5_requires_real_local_files(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "forecasting_library_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "forecasting_library_benchmark_missing",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    benchmark = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = benchmark
    spec.loader.exec_module(benchmark)

    try:
        benchmark.load_m5_fixture(
            types.SimpleNamespace(
                m5_data_dir=tmp_path,
                m5_series_limit=0,
                m5_history_days=0,
                no_download=True,
            )
        )
    except FileNotFoundError as exc:
        assert "sales_train_evaluation.csv or sales_train_validation.csv" in str(exc)
    else:
        raise AssertionError("missing M5 files should hard-fail")


def test_forecasting_benchmark_allows_bounded_m5_full_roster():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "forecasting_library_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "forecasting_library_benchmark_m5_roster",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    benchmark = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = benchmark
    spec.loader.exec_module(benchmark)

    benchmark.validate_args(
        types.SimpleNamespace(
            lanes=1,
            horizon=28,
            source="m5",
            days=120,
            suite=False,
            m4_suite=False,
            m4_series_limit=0,
            m5_series_limit=100,
            m5_history_days=90,
            m6_series_limit=0,
            m6_horizon=28,
            model_roster="full",
            allow_full_m5_roster=False,
            cartoboost_n_estimators=1,
            cartoboost_max_depth=3,
            cartoboost_min_samples_leaf=8,
            suite_folds=1,
        )
    )

    try:
        benchmark.validate_args(
            types.SimpleNamespace(
                lanes=1,
                horizon=28,
                source="m5",
                days=120,
                suite=False,
                m4_suite=False,
                m4_series_limit=0,
                m5_series_limit=0,
                m5_history_days=90,
                m6_series_limit=0,
                m6_horizon=28,
                model_roster="full",
                allow_full_m5_roster=False,
                cartoboost_n_estimators=1,
                cartoboost_max_depth=3,
                cartoboost_min_samples_leaf=8,
                suite_folds=1,
            )
        )
    except ValueError as exc:
        assert "requires a positive --m5-series-limit" in str(exc)
    else:
        raise AssertionError("unbounded M5 full-roster runs should require explicit opt-in")


def test_forecasting_benchmark_auto_config_is_fixed_stronger_policy():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "forecasting_library_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "forecasting_library_benchmark_auto_policy",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    benchmark = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = benchmark
    spec.loader.exec_module(benchmark)

    config = {
        "n_estimators": 180,
        "auto_n_estimators": None,
        "learning_rate": 0.06,
        "max_depth": 4,
        "min_samples_leaf": 8,
    }
    auto = benchmark.cartoboost_auto_config(config, season_length=7, horizon=28)

    assert auto["n_estimators"] == 360
    assert auto["learning_rate"] == 0.06
    assert auto["max_depth"] == 6
    assert auto["min_samples_leaf"] == 4
    assert config["n_estimators"] == 180
    assert config["auto_n_estimators"] is None

    override = benchmark.cartoboost_auto_config(
        {
            **config,
            "n_estimators": 60,
            "auto_n_estimators": 72,
        },
        season_length=7,
        horizon=28,
    )
    assert override["n_estimators"] == 72


def test_forecasting_benchmark_model_settings_include_auto_policy():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "forecasting_library_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "forecasting_library_benchmark_model_settings_policy",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    benchmark = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = benchmark
    spec.loader.exec_module(benchmark)

    settings = benchmark.cartoboost_model_settings(
        {
            "n_estimators": 60,
            "auto_n_estimators": 72,
            "learning_rate": 0.06,
            "max_depth": 4,
            "min_samples_leaf": 8,
        }
    )

    assert set(settings) == {"cartoboost_lag", "cartoboost_auto_forecast"}
    assert settings["cartoboost_lag"]["n_estimators"] == 60
    assert settings["cartoboost_auto_forecast"]["auto_n_estimators"] == 72
    assert (
        settings["cartoboost_auto_forecast"]["auto_n_estimators_policy"]
        == "explicit --cartoboost-auto-n-estimators override"
    )
    assert (
        "covariate-calendar interactions" in settings["cartoboost_lag"]["native_covariate_policy"]
    )
    assert (
        "non-M inner validation skips raw auto"
        in settings["cartoboost_auto_forecast"]["auto_selector_policy"]
    )
    assert (
        "cache identical inner validation cutoffs"
        in settings["cartoboost_auto_forecast"]["auto_selector_policy"]
    )


def test_forecasting_benchmark_auto_ensemble_weights_are_stable_and_normalized():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "forecasting_library_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "forecasting_library_benchmark_auto_weights",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    benchmark = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = benchmark
    spec.loader.exec_module(benchmark)

    weights = benchmark.validation_ensemble_weights(
        {
            "worse": 4.0,
            "best": 1.0,
            "nan": float("nan"),
            "second": 2.0,
            "third": 3.0,
            "fourth": 3.5,
            "fifth": 3.6,
        }
    )

    assert list(weights) == ["best", "second", "third", "fourth"]
    assert sum(weights.values()) == pytest.approx(1.0)
    assert weights["best"] > weights["second"] > weights["third"] > weights["fourth"]


def test_forecasting_benchmark_loads_m6_assets_file(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "forecasting_library_benchmark.py"
    spec = importlib.util.spec_from_file_location("forecasting_library_benchmark_m6", module_path)
    assert spec is not None
    assert spec.loader is not None
    benchmark = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = benchmark
    spec.loader.exec_module(benchmark)

    path = tmp_path / "assets_m6.csv"
    lines = ["symbol,date,price"]
    start = date(2020, 1, 1)
    for day in range(90):
        current = start + timedelta(days=day)
        lines.append(f"AAA,{current.isoformat()},{100.0 + day}")
        lines.append(f"BBB,{current.isoformat()},{200.0 + day * 2}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    table, dataset = benchmark.load_m6_fixture(
        types.SimpleNamespace(
            m6_assets_path=path,
            m6_series_limit=0,
            m6_horizon=28,
            no_download=True,
        )
    )

    assert dataset["series"] == 2
    assert dataset["horizon"] == 28
    assert dataset["days"] == 90
    assert table.height == 180
    assert {"lane_id", "date", "loads", *benchmark.STATIC_COVARIATES} <= set(table.columns)
    assert "shared_m6_phase14_rank_blend" in benchmark.selectable_candidate_names(
        "cartoboost_auto_forecast",
        source="m6",
    )
    assert "shared_m6_market_neutral_zero" in benchmark.selectable_candidate_names(
        "cartoboost_auto_forecast",
        source="m6",
    )
    assert "shared_m6_phase14_rank_blend" not in benchmark.selectable_candidate_names(
        "cartoboost_auto_forecast",
        source="m5",
    )


def test_forecasting_benchmark_emits_m5_wrmsse_artifact():
    pl = pytest.importorskip("polars")
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "forecasting_library_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "forecasting_library_benchmark_m5_artifact",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    benchmark = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = benchmark
    spec.loader.exec_module(benchmark)

    rows = []
    start = date(2026, 1, 1)
    for lane, base, pickup_zone in [("PU1->DO1", 10.0, 1.0), ("PU2->DO2", 20.0, 2.0)]:
        for day in range(8):
            rows.append(
                {
                    "lane_id": lane,
                    "date": start + timedelta(days=day),
                    "loads": base + day,
                    "pickup_zone": pickup_zone,
                    "dropoff_zone": pickup_zone + 10.0,
                    "distance_miles": 1.0,
                    "airport_lane": 0.0,
                    "pickup_borough_code": pickup_zone,
                }
            )
    table = pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Datetime("us")))
    scored = pl.DataFrame(
        [
            {
                "series_id": lane,
                "timestamp": start + timedelta(days=day),
                "horizon": day - 5,
                "actual": base + day,
                "cartoboost_lag": base + day - 1.0,
                "cartoboost_auto_forecast": base + day,
            }
            for lane, base in [("PU1->DO1", 10.0), ("PU2->DO2", 20.0)]
            for day in range(6, 8)
        ]
    ).with_columns(pl.col("timestamp").cast(pl.Datetime("us")))

    artifact = benchmark.benchmark_objective_artifacts(
        "m5",
        train_table=table,
        scored=scored,
        model_names=["cartoboost_lag", "cartoboost_auto_forecast"],
        season_length=1,
    )

    m5 = artifact["m5"]
    assert artifact["primary_metric"] == "wrmsse"
    assert m5["ranking"] == ["cartoboost_auto_forecast", "cartoboost_lag"]
    assert set(m5["levels"]) == {"total", "state", "store", "item", "item_store"}
    total = m5["levels"]["total"]["models"]
    assert total["cartoboost_auto_forecast"]["wrmsse"] == pytest.approx(0.0)
    assert total["cartoboost_lag"]["wrmsse"] > 0.0


def test_forecasting_benchmark_exposes_lag_as_traceable_auto_candidate():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "forecasting_library_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "forecasting_library_benchmark_no_auto_protection",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    benchmark = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = benchmark
    spec.loader.exec_module(benchmark)

    assert not hasattr(benchmark, "validation_auto_lag_protection")
    assert "cartoboost_lag" in benchmark.selectable_candidate_names(
        "cartoboost_auto_forecast",
        source="m4",
    )
    assert benchmark.candidate_complexity_rank("cartoboost_lag") < (
        benchmark.candidate_complexity_rank("cartoboost_auto_forecast")
    )


def test_forecasting_benchmark_emits_m6_rps_artifact():
    pl = pytest.importorskip("polars")
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "forecasting_library_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "forecasting_library_benchmark_m6_artifact",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    benchmark = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = benchmark
    spec.loader.exec_module(benchmark)

    scored = pl.DataFrame(
        [
            {
                "series_id": symbol,
                "timestamp": date(2026, 1, 1) + timedelta(days=horizon),
                "horizon": horizon + 1,
                "actual": actual,
                "cartoboost_lag": lag,
                "cartoboost_auto_forecast": actual,
            }
            for symbol, actual, lag in [
                ("AAA", -0.03, 0.04),
                ("BBB", -0.01, 0.03),
                ("CCC", 0.01, 0.02),
                ("DDD", 0.03, 0.01),
                ("EEE", 0.05, -0.02),
            ]
            for horizon in range(2)
        ]
    ).with_columns(pl.col("timestamp").cast(pl.Datetime("us")))

    artifact = benchmark.benchmark_objective_artifacts(
        "m6",
        train_table=scored,
        scored=scored,
        model_names=["cartoboost_lag", "cartoboost_auto_forecast"],
        season_length=7,
    )

    m6 = artifact["m6"]
    assert artifact["primary_metric"] == "rank_probability_score"
    auto = m6["models"]["cartoboost_auto_forecast"]
    assert auto["asset_count"] == 5
    assert auto["rank_probability_calibration"]["fallback"] == "uniform_when_no_validation_support"
    assert auto["mean_rps"] == pytest.approx(m6["models"]["cartoboost_lag"]["mean_rps"])
    assert sum(row["weight"] for row in auto["decisions"]) == pytest.approx(0.0)


def test_forecasting_benchmark_m6_rps_uses_validation_calibration():
    pl = pytest.importorskip("polars")
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "forecasting_library_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "forecasting_library_benchmark_m6_calibrated_artifact",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    benchmark = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = benchmark
    spec.loader.exec_module(benchmark)

    rows = [
        ("AAA", -0.05),
        ("BBB", -0.02),
        ("CCC", 0.00),
        ("DDD", 0.02),
        ("EEE", 0.05),
    ]
    scored = pl.DataFrame(
        [
            {
                "series_id": symbol,
                "timestamp": date(2026, 2, 1) + timedelta(days=horizon),
                "horizon": horizon + 1,
                "actual": actual,
                "cartoboost_auto_forecast": actual,
                "cartoboost_lag": actual,
            }
            for symbol, actual in rows
            for horizon in range(2)
        ]
    ).with_columns(pl.col("timestamp").cast(pl.Datetime("us")))
    calibration_scored = pl.DataFrame(
        [
            {
                "series_id": symbol,
                "timestamp": date(2026, 1, 1) + timedelta(days=horizon),
                "horizon": horizon + 1,
                "actual": actual,
                "cartoboost_auto_forecast": actual,
                "cartoboost_lag": -actual,
            }
            for symbol, actual in rows
            for horizon in range(2)
        ]
    ).with_columns(pl.col("timestamp").cast(pl.Datetime("us")))

    artifact = benchmark.m6_rps_artifact(
        scored,
        model_names=["cartoboost_lag", "cartoboost_auto_forecast"],
        calibration_scored=calibration_scored,
    )

    auto = artifact["models"]["cartoboost_auto_forecast"]
    lag = artifact["models"]["cartoboost_lag"]
    assert auto["rank_probability_calibration"]["validation_support"] == 5
    assert auto["rank_probability_calibration"]["shrinkage_to_confusion"] > 0.0
    assert auto["mean_rps"] < lag["mean_rps"]
    assert artifact["ranking"][0] == "cartoboost_auto_forecast"


def test_model_benchmark_suite_graph_families_smoke(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = tmp_path / "model_benchmarks_graph"

    subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "run_model_benchmark_suite.py"),
            "--output-dir",
            str(output_dir),
            "--datasets",
            "graph",
            "--models",
            (
                "cartoboost_graph_node2vec,cartoboost_graph_graphsage,"
                "cartoboost_graph_hetero_graphsage,cartoboost_graph_hinsage"
            ),
            "--n-rows",
            "96",
            "--graph-dim",
            "2",
            "--graph-epochs",
            "1",
            "--n-estimators",
            "4",
            "--no-plots",
        ],
        cwd=repo_root,
        check=True,
    )

    results = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))
    models = results["workloads"]["graph"]["splits"]["random"]["models"]
    for model_name, family in {
        "cartoboost_graph_node2vec": "node2vec",
        "cartoboost_graph_graphsage": "graphsage",
        "cartoboost_graph_hetero_graphsage": "hetero_graphsage",
        "cartoboost_graph_hinsage": "hinsage",
    }.items():
        assert models[model_name]["status"] == "ok"
        assert models[model_name]["config"]["graph_family"] == family


def test_nyc_taxi_benchmark_graph_family_switch_smoke(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]

    for family in ["node2vec", "graphsage", "hetero_graphsage", "hinsage"]:
        output_dir = tmp_path / f"nyc_taxi_{family}"
        subprocess.run(
            [
                sys.executable,
                str(repo_root / "scripts" / "run_nyc_taxi_quality_benchmarks.py"),
                "--synthetic-smoke",
                "--output-dir",
                str(output_dir),
                "--tasks",
                "duration",
                "--models",
                "cartoboost_graph",
                "--graph-family",
                family,
                "--graph-dim",
                "2",
                "--graph-epochs",
                "1",
                "--cartoboost-n-estimators",
                "4",
            ],
            cwd=repo_root,
            check=True,
        )

        results = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))
        assert results["model_config"]["graph_family"] == family
        model = results["tasks"]["duration"]["splits"]["random"]["models"]["cartoboost_graph"]
        assert model["status"] == "ok"
        assert model["config"]["graph_family"] == family
