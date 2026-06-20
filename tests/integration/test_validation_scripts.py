from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import types
from datetime import date, timedelta
from pathlib import Path

import pytest


def ordinal_word(rank: int) -> str:
    words = {
        1: "first",
        2: "second",
        3: "third",
        4: "fourth",
        5: "fifth",
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
    assert benchmark.auto_selection_objective("m5") == "wrmsse"
    assert benchmark.auto_selection_objective("m6") == "rank_probability_score_then_rmse"
    assert benchmark.auto_selection_objective("nyc-taxi") == "rmse"


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
    assert benchmark.shared_candidate_validation_cutoffs(list(range(62)), horizon=28) == [34]
    assert benchmark.shared_candidate_validation_cutoffs(list(range(20)), horizon=14) == []


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
                "cartoboost_auto_forecast": 1.00,
                "shared_seasonal_base": 1.03,
                "shared_calendar_dom": 0.99,
            },
            source="synthetic",
        )
        == "shared_seasonal_base"
    )


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
    assert benchmark.requires_lag_spine(source="synthetic", season_length=7, horizon=14)
    assert benchmark.requires_lag_spine(source="m4", season_length=24, horizon=48)
    assert not benchmark.requires_lag_spine(source="m5", season_length=1, horizon=28)
    assert not benchmark.requires_lag_spine(source="m6", season_length=1, horizon=28)
    script = module_path.read_text(encoding="utf-8")
    assert 'source in {"synthetic", "m4"} and best_candidate == "cartoboost_lag"' in script


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
    assert (
        f"`cartoboost_auto_forecast` and `cartoboost_lag` tied with mean RMSE "
        f"ratio {synthetic_auto:.6f} and "
        f"{synthetic_quality['wins_or_ties']['cartoboost_lag']}/4 wins-or-ties"
    ) in docs
    assert synthetic_auto == pytest.approx(synthetic_lag)

    m4 = json.loads((artifacts / "forecasting_overhaul_m4_committed.json").read_text())
    m4_quality = m4["aggregate_quality"]
    m4_lag = m4_quality["mean_rmse_ratio_to_problem_best"]["cartoboost_lag"]
    m4_auto = m4_quality["mean_rmse_ratio_to_problem_best"]["cartoboost_auto_forecast"]
    assert (
        f"`cartoboost_auto_forecast` ranked first with mean RMSE ratio {m4_auto:.6f} "
        f"and {m4_quality['wins_or_ties']['cartoboost_auto_forecast']}/6 wins-or-ties; "
        f"`cartoboost_lag` had mean RMSE ratio {m4_lag:.6f}"
    ) in docs
    assert m4_auto < m4_lag

    m5 = json.loads((artifacts / "forecasting_overhaul_m5_committed.json").read_text())
    m5_auto = m5["metrics"]["cartoboost_auto_forecast"]
    m5_lag = m5["metrics"]["cartoboost_lag"]
    m5_wrmsse = m5["official_metrics"]["m5"]["model_scores"]["cartoboost_auto_forecast"]
    m5_auto_wrmsse = m5["official_metrics"]["m5"]["model_scores"]["cartoboost_auto_forecast"]
    assert (
        f"RMSE {m5_auto['rmse']:.6f}, MAE {m5_auto['mae']:.6f}, "
        f"WAPE {m5_auto['wape']:.6f}, WRMSSE {m5_wrmsse:.6f}"
    ) in docs
    assert m5_auto["rmse"] < m5_lag["rmse"]
    assert m5_auto_wrmsse < m5["official_metrics"]["m5"]["model_scores"]["cartoboost_lag"]

    m6 = json.loads((artifacts / "forecasting_overhaul_m6_committed.json").read_text())
    m6_auto = m6["metrics"]["cartoboost_auto_forecast"]
    m6_rps = m6["official_metrics"]["m6"]["models"]
    assert (
        f"RMSE {m6_auto['rmse']:.6f}, MAE {m6_auto['mae']:.6f}, "
        f"WAPE {m6_auto['wape']:.6f}, RPS "
        f"{m6_rps['cartoboost_lag']['mean_rps']:.6f}"
    ) in docs
    assert m6_auto["rmse"] < m6["metrics"]["cartoboost_lag"]["rmse"]
    assert (
        f"`cartoboost_auto_forecast` and `cartoboost_lag` tied calibrated RPS "
        f"at {m6_rps['cartoboost_lag']['mean_rps']:.6f}"
    ) in docs

    full_roster = json.loads(
        (artifacts / "forecasting_overhaul_committed_suite_full_roster.json").read_text()
    )
    full_roster_quality = full_roster["aggregate_quality"]
    full_winner = full_roster_quality["mean_rmse_ratio_ranking"][0]
    full_winner_ratio = full_roster_quality["mean_rmse_ratio_to_problem_best"][full_winner]
    full_auto_ratio = full_roster_quality["mean_rmse_ratio_to_problem_best"][
        "cartoboost_auto_forecast"
    ]
    assert (
        f"`{full_winner}` remains the maintained winner at mean RMSE ratio "
        f"{full_winner_ratio:.6f}; CartoBoost auto is {full_auto_ratio:.6f}"
    ) in docs

    m5_sample = json.loads((artifacts / "forecasting_m5_full_roster_sample.json").read_text())
    m5_sample_winner = m5_sample["quality"]["winner"]
    m5_sample_winner_rmse = m5_sample["metrics"][m5_sample_winner]["rmse"]
    m5_sample_cartoboost_rmse = m5_sample["metrics"]["cartoboost_auto_forecast"]["rmse"]
    m5_sample_rmse_ranking = sorted(
        m5_sample["metrics"],
        key=lambda model: m5_sample["metrics"][model]["rmse"],
    )
    m5_sample_cartoboost_rank = ordinal_word(
        m5_sample_rmse_ranking.index("cartoboost_auto_forecast") + 1
    )
    m5_wrmsse = m5_sample["official_metrics"]["m5"]
    m5_wrmsse_winner = m5_wrmsse["ranking"][0]
    m5_wrmsse_winner_score = m5_wrmsse["model_scores"][m5_wrmsse_winner]
    m5_cartoboost_wrmsse = m5_wrmsse["model_scores"]["cartoboost_auto_forecast"]
    m5_cartoboost_wrmsse_rank = ordinal_word(
        m5_wrmsse["ranking"].index("cartoboost_auto_forecast") + 1
    )
    assert (
        f"`{m5_sample_winner}` remains the point-metric winner at RMSE "
        f"{m5_sample_winner_rmse:.6f}; CartoBoost auto is {m5_sample_cartoboost_rank} "
        f"by RMSE at "
        f"{m5_sample_cartoboost_rmse:.6f}. `{m5_wrmsse_winner}` leads official WRMSSE at "
        f"{m5_wrmsse_winner_score:.6f}; CartoBoost auto is {m5_cartoboost_wrmsse_rank} at "
        f"{m5_cartoboost_wrmsse:.6f}."
    ) in docs

    m6_full = json.loads((artifacts / "forecasting_m6_full.json").read_text())
    m6_full_winner = m6_full["quality"]["winner"]
    m6_full_winner_rmse = m6_full["metrics"][m6_full_winner]["rmse"]
    m6_full_cartoboost_rmse = m6_full["metrics"]["cartoboost_lag"]["rmse"]
    assert (
        f"`{m6_full_winner}` remains the maintained winner at RMSE "
        f"{m6_full_winner_rmse:.6f}; CartoBoost lag RMSE is "
        f"{m6_full_cartoboost_rmse:.6f}"
    ) in docs


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
