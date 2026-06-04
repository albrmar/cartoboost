use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

pub const CONTEXT: usize = 0;
pub const RESIDUAL: usize = 1;
pub const TREE_FIT: usize = 2;
pub const HISTOGRAM: usize = 3;
pub const MATERIALIZE: usize = 4;
pub const LEAF: usize = 5;
pub const PRED_UPDATE: usize = 6;
pub const HIST_ACCUMULATE: usize = 7;
pub const HIST_SCORE: usize = 8;
pub const PARENT_SSE: usize = 9;
pub const HIST_PREPARE: usize = 10;
pub const MATERIALIZE_PARTITION: usize = 11;
pub const MATERIALIZE_CHILD_HIST: usize = 12;

const NAMES: [&str; 13] = [
    "context",
    "residual",
    "tree_fit",
    "histogram",
    "materialize",
    "leaf",
    "pred_update",
    "hist_accumulate",
    "hist_score",
    "parent_sse",
    "hist_prepare",
    "materialize_partition",
    "materialize_child_hist",
];

static COUNTERS: [AtomicU64; 13] = [
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
];

pub fn enabled() -> bool {
    std::env::var_os("GEOBOOST_PROFILE_FIT").is_some()
}

pub fn reset() {
    for counter in &COUNTERS {
        counter.store(0, Ordering::Relaxed);
    }
}

pub fn add(bucket: usize, elapsed: Duration) {
    if !enabled() {
        return;
    }
    COUNTERS[bucket].fetch_add(elapsed.as_nanos() as u64, Ordering::Relaxed);
}

pub fn timed<T>(bucket: usize, f: impl FnOnce() -> T) -> T {
    if !enabled() {
        return f();
    }
    let started = Instant::now();
    let value = f();
    add(bucket, started.elapsed());
    value
}

pub fn report(label: &str, total: Duration) {
    if !enabled() {
        return;
    }
    let total_ms = total.as_secs_f64() * 1000.0;
    eprint!("geoboost_fit_profile label={label} total_ms={total_ms:.3}");
    for (name, counter) in NAMES.iter().zip(&COUNTERS) {
        let ms = counter.load(Ordering::Relaxed) as f64 / 1_000_000.0;
        eprint!(" {name}_ms={ms:.3}");
    }
    eprintln!();
}
