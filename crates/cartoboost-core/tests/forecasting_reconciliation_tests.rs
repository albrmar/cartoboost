use cartoboost_core::forecasting::{
    HierarchySpec, Reconciler, ReconciliationMethod, TemporalAggregation, TemporalHierarchy,
};
use cartoboost_core::metrics::{wrmsse, WrmsseSeries};

fn taxi_hierarchy() -> HierarchySpec {
    HierarchySpec::from_edges(vec![
        ("all_taxi", None),
        ("PU1", Some("all_taxi")),
        ("PU2", Some("all_taxi")),
        ("PU1_DO1", Some("PU1")),
        ("PU1_DO2", Some("PU1")),
        ("PU2_DO1", Some("PU2")),
    ])
    .expect("valid hierarchy")
}

#[test]
fn sparse_hierarchy_aggregates_bottom_taxi_series() {
    let hierarchy = taxi_hierarchy();
    let aggregated = hierarchy
        .aggregate_bottom_values(&[10.0, 15.0, 20.0])
        .expect("aggregate");

    assert_eq!(
        hierarchy.bottom_node_ids(),
        vec!["PU1_DO1", "PU1_DO2", "PU2_DO1"]
    );
    assert_eq!(aggregated, vec![45.0, 25.0, 20.0, 10.0, 15.0, 20.0]);
    assert!(hierarchy
        .is_coherent(&aggregated, 1e-12)
        .expect("coherence check"));
}

#[test]
fn bottom_up_and_top_down_reconciliation_return_coherent_forecasts() {
    let hierarchy = taxi_hierarchy();
    let base = vec![
        vec![100.0, 120.0],
        vec![70.0, 80.0],
        vec![40.0, 45.0],
        vec![20.0, 25.0],
        vec![30.0, 35.0],
        vec![50.0, 55.0],
    ];

    let bottom_up = Reconciler::new(hierarchy.clone(), ReconciliationMethod::BottomUp)
        .reconcile(&base)
        .expect("bottom-up");
    assert_eq!(bottom_up[0], vec![100.0, 115.0]);
    assert_eq!(bottom_up[1], vec![50.0, 60.0]);
    assert_eq!(bottom_up[2], vec![50.0, 55.0]);

    let top_down = Reconciler::new(hierarchy.clone(), ReconciliationMethod::TopDown)
        .reconcile(&base)
        .expect("top-down");
    assert_eq!(top_down[0], vec![100.0, 120.0]);
    assert!(hierarchy
        .is_coherent(
            &top_down.iter().map(|row| row[0]).collect::<Vec<_>>(),
            1e-10
        )
        .expect("coherence"));
    assert_eq!(top_down[3][0], 20.0);
    assert_eq!(top_down[4][0], 30.0);
    assert_eq!(top_down[5][0], 50.0);
}

#[test]
fn middle_out_uses_requested_level_and_ols_projects_to_coherent_space() {
    let hierarchy = taxi_hierarchy();
    let base = vec![
        vec![90.0],
        vec![60.0],
        vec![50.0],
        vec![20.0],
        vec![40.0],
        vec![55.0],
    ];

    let middle = Reconciler::new(
        hierarchy.clone(),
        ReconciliationMethod::MiddleOut { level: 1 },
    )
    .reconcile(&base)
    .expect("middle-out");
    assert_eq!(middle[0][0], 110.0);
    assert_eq!(middle[1][0], 60.0);
    assert_eq!(middle[2][0], 50.0);
    assert_eq!(middle[3][0], 20.0);
    assert_eq!(middle[4][0], 40.0);
    assert_eq!(middle[5][0], 50.0);

    let ols = Reconciler::new(hierarchy.clone(), ReconciliationMethod::Ols)
        .reconcile(&base)
        .expect("ols");
    let first_horizon = ols.iter().map(|row| row[0]).collect::<Vec<_>>();
    assert!(hierarchy
        .is_coherent(&first_horizon, 1e-10)
        .expect("coherence"));
    assert!(ols[0][0] > 90.0);
    assert!(ols[0][0] < 110.0);
}

#[test]
fn wls_and_mint_shrink_accept_small_reference_inputs() {
    let hierarchy = taxi_hierarchy();
    let base = vec![
        vec![90.0],
        vec![60.0],
        vec![50.0],
        vec![20.0],
        vec![40.0],
        vec![55.0],
    ];

    let wls = Reconciler::new(
        hierarchy.clone(),
        ReconciliationMethod::Wls {
            variances: vec![4.0, 2.0, 2.0, 1.0, 1.0, 1.0],
        },
    )
    .reconcile(&base)
    .expect("wls");
    assert!(hierarchy
        .is_coherent(&wls.iter().map(|row| row[0]).collect::<Vec<_>>(), 1e-10)
        .expect("coherence"));

    let residuals = vec![
        vec![2.0, -2.0, 1.0],
        vec![1.0, -1.0, 0.5],
        vec![1.5, -1.5, 0.25],
        vec![0.5, -0.5, 0.25],
        vec![0.7, -0.7, 0.25],
        vec![0.9, -0.9, 0.25],
    ];
    let mint = Reconciler::new(
        hierarchy.clone(),
        ReconciliationMethod::MinTShrink {
            residuals,
            shrinkage: 0.25,
        },
    )
    .reconcile(&base)
    .expect("mint shrink");
    assert!(hierarchy
        .is_coherent(&mint.iter().map(|row| row[0]).collect::<Vec<_>>(), 1e-10)
        .expect("coherence"));
}

#[test]
fn temporal_hierarchy_aggregates_non_overlapping_windows() {
    let hierarchy = TemporalHierarchy::new(
        "hour",
        vec![
            TemporalAggregation::new("six_hour", 6).expect("six hour"),
            TemporalAggregation::new("day", 24).expect("day"),
        ],
    )
    .expect("temporal hierarchy");

    let values = (1..=24).map(f64::from).collect::<Vec<_>>();
    let levels = hierarchy.aggregate_series(&values).expect("aggregate");

    assert_eq!(levels[0].values.len(), 24);
    assert_eq!(levels[1].values, vec![21.0, 57.0, 93.0, 129.0]);
    assert_eq!(levels[2].values, vec![300.0]);
}

#[test]
fn wrmsse_matches_manual_m5_style_reference() {
    let rows = vec![
        WrmsseSeries::new(
            "PULocationID=1",
            vec![10.0, 12.0, 14.0, 16.0],
            vec![18.0, 20.0],
            vec![17.0, 23.0],
            2.0,
        ),
        WrmsseSeries::new(
            "PULocationID=2",
            vec![20.0, 21.0, 23.0, 26.0],
            vec![30.0, 34.0],
            vec![32.0, 31.0],
            1.0,
        ),
    ];

    let score = wrmsse(&rows, 1).expect("wrmsse");

    assert!((score.series[0].scale - 4.0).abs() < 1e-12);
    assert!((score.series[1].scale - (14.0 / 3.0)).abs() < 1e-12);
    assert!((score.series[0].rmsse - (10.0_f64 / 8.0).sqrt()).abs() < 1e-12);
    assert!((score.series[1].rmsse - (13.0_f64 / (2.0 * 14.0 / 3.0)).sqrt()).abs() < 1e-12);
    assert!((score.score - 1.138_753_888_734_651_6).abs() < 1e-12);
}
