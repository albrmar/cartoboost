use geoboost_core::tree::SplitterKind;
use geoboost_core::{Booster, BoosterConfig, Dataset, Model as CoreModel};
use std::collections::BTreeMap;
use std::env;
use std::error::Error;
use std::fs;
use std::path::Path;

type CliResult<T> = Result<T, Box<dyn Error>>;

fn main() {
    if let Err(err) = run() {
        eprintln!("geoboost: {err}");
        std::process::exit(1);
    }
}

fn run() -> CliResult<()> {
    let mut args = env::args().skip(1);
    let Some(command) = args.next() else {
        print_help();
        return Ok(());
    };

    if command == "-h" || command == "--help" {
        print_help();
        return Ok(());
    }

    let opts = parse_options(args.collect())?;
    if opts.contains_key("help") {
        print_help();
        return Ok(());
    }

    match command.as_str() {
        "train" => train(opts),
        "predict" => predict(opts),
        "eval" => evaluate(opts),
        "inspect" => inspect(opts),
        other => Err(format!("unknown command '{other}'").into()),
    }
}

fn print_help() {
    println!(
        "geoboost <command> [options]\n\nCommands:\n  train    --data <csv> [--config <toml>] [--model-out <path>] [--output json|csv]\n  predict  --model <path> --input <csv> [--predictions-out <path>] [--output json|csv]\n  eval     --model <path> --data <csv> [--output json|csv]\n  inspect  [--model <path>] [--config <toml>] [--data <csv>] [--output json|csv]"
    );
}

fn parse_options(raw: Vec<String>) -> CliResult<BTreeMap<String, String>> {
    let mut opts = BTreeMap::new();
    let mut i = 0;
    while i < raw.len() {
        let key = &raw[i];
        if !key.starts_with("--") {
            return Err(format!("expected option, got '{key}'").into());
        }
        let name = key.trim_start_matches("--").to_string();
        if name == "help" {
            opts.insert(name, "true".to_string());
            i += 1;
            continue;
        }
        let Some(value) = raw.get(i + 1) else {
            return Err(format!("missing value for '{key}'").into());
        };
        opts.insert(name, value.clone());
        i += 2;
    }
    Ok(opts)
}

fn train(opts: BTreeMap<String, String>) -> CliResult<()> {
    let data_path = required(&opts, "data")?;
    let output = output_format(&opts)?;
    let rows = read_csv(data_path)?;
    let cfg = opts
        .get("config")
        .map(|path| read_config(path))
        .transpose()?
        .unwrap_or_default();
    let target = cfg
        .target
        .clone()
        .or_else(|| rows.headers.last().cloned())
        .unwrap_or_else(|| "target".to_string());
    let (dataset, y, feature_count) = numeric_dataset_and_target(&rows, &target)?;
    let model_path = opts
        .get("model-out")
        .cloned()
        .unwrap_or_else(|| "geoboost-model.json".to_string());
    let mut model = Booster::new(cfg.booster_config()).fit(&dataset, &y, None)?;
    model.target_name = Some(target);
    model.save(&model_path)?;

    match output.as_str() {
        "csv" => println!(
            "ok,command,rows,features,model_path\ntrue,train,{},{},{}",
            rows.records.len(),
            feature_count,
            csv_cell(&model_path)
        ),
        _ => println!(
            "{{\"ok\":true,\"command\":\"train\",\"rows\":{},\"features\":{},\"model_path\":\"{}\",\"trees\":{}}}",
            rows.records.len(),
            feature_count,
            json_escape(&model_path),
            model.trees.len()
        ),
    }
    Ok(())
}

fn predict(opts: BTreeMap<String, String>) -> CliResult<()> {
    let model_path = required(&opts, "model")?;
    let input_path = required(&opts, "input")?;
    let output = output_format(&opts)?;
    let rows = read_csv(input_path)?;
    if let Ok(model) = CoreModel::load(model_path) {
        let dataset = numeric_dataset_without_target(&rows, model.feature_count)?;
        let predictions = model.predict(&dataset);
        let mut csv = String::from("row,prediction\n");
        for (idx, prediction) in predictions.iter().enumerate() {
            csv.push_str(&format!("{idx},{}\n", format_float(*prediction)));
        }
        if let Some(path) = opts.get("predictions-out") {
            fs::write(path, &csv)?;
        }

        match output.as_str() {
            "csv" => print!("{csv}"),
            _ => println!(
                "{{\"ok\":true,\"command\":\"predict\",\"rows\":{},\"predictions_out\":{}}}",
                predictions.len(),
                optional_json_string(opts.get("predictions-out"))
            ),
        }
        return Ok(());
    }

    let model = read_model(model_path)?;
    let mut csv = String::from("row,prediction\n");
    for idx in 0..rows.records.len() {
        csv.push_str(&format!(
            "{},{}\n",
            idx,
            csv_cell(&model.baseline.as_string())
        ));
    }
    if let Some(path) = opts.get("predictions-out") {
        fs::write(path, &csv)?;
    }

    match output.as_str() {
        "csv" => print!("{csv}"),
        _ => println!(
            "{{\"ok\":true,\"command\":\"predict\",\"rows\":{},\"prediction\":{},\"predictions_out\":{}}}",
            rows.records.len(),
            model.baseline.to_json_value(),
            optional_json_string(opts.get("predictions-out"))
        ),
    }
    Ok(())
}

fn evaluate(opts: BTreeMap<String, String>) -> CliResult<()> {
    let model_path = required(&opts, "model")?;
    let data_path = required(&opts, "data")?;
    let output = output_format(&opts)?;
    let rows = read_csv(data_path)?;
    if let Ok(model) = CoreModel::load(model_path) {
        let target = model
            .target_name
            .clone()
            .or_else(|| rows.headers.last().cloned())
            .unwrap_or_else(|| "target".to_string());
        let (dataset, y, _) = numeric_dataset_and_target(&rows, &target)?;
        let predictions = model.predict(&dataset);
        let mae = y
            .iter()
            .zip(&predictions)
            .map(|(actual, prediction)| (actual - prediction).abs())
            .sum::<f64>()
            / y.len().max(1) as f64;

        match output.as_str() {
            "csv" => println!(
                "ok,command,rows,target,mae\ntrue,eval,{},{},{}",
                rows.records.len(),
                csv_cell(&target),
                format_float(mae)
            ),
            _ => println!(
                "{{\"ok\":true,\"command\":\"eval\",\"rows\":{},\"target\":\"{}\",\"mae\":{}}}",
                rows.records.len(),
                json_escape(&target),
                format_float(mae)
            ),
        }
        return Ok(());
    }

    let model = read_model(model_path)?;
    let target_idx = rows.headers.iter().position(|name| name == &model.target);
    let mae = target_idx.and_then(|idx| mean_absolute_error(&rows, idx, model.baseline.numeric()));

    match output.as_str() {
        "csv" => println!(
            "ok,command,rows,target,mae\ntrue,eval,{},{},{}",
            rows.records.len(),
            csv_cell(&model.target),
            mae.map(format_float).unwrap_or_else(|| "".to_string())
        ),
        _ => println!(
            "{{\"ok\":true,\"command\":\"eval\",\"rows\":{},\"target\":\"{}\",\"mae\":{}}}",
            rows.records.len(),
            json_escape(&model.target),
            mae.map(format_float).unwrap_or_else(|| "null".to_string())
        ),
    }
    Ok(())
}

fn inspect(opts: BTreeMap<String, String>) -> CliResult<()> {
    let output = output_format(&opts)?;
    let model = opts.get("model").map(|path| read_model(path)).transpose()?;
    let config = opts
        .get("config")
        .map(|path| read_config(path))
        .transpose()?;
    let data = opts.get("data").map(|path| read_csv(path)).transpose()?;

    match output.as_str() {
        "csv" => {
            println!("kind,path,rows,columns,target");
            if let Some(path) = opts.get("model") {
                let Some(model) = &model else { unreachable!() };
                println!(
                    "model,{},{},{},{}",
                    csv_cell(path),
                    model.rows,
                    model.feature_count + 1,
                    csv_cell(&model.target)
                );
            }
            if let Some(path) = opts.get("config") {
                let target = config
                    .as_ref()
                    .and_then(|cfg| cfg.target.as_ref())
                    .cloned()
                    .unwrap_or_default();
                println!("config,{},,,{}", csv_cell(path), csv_cell(&target));
            }
            if let Some(path) = opts.get("data") {
                let Some(data) = &data else { unreachable!() };
                println!(
                    "data,{},{},{},",
                    csv_cell(path),
                    data.records.len(),
                    data.headers.len()
                );
            }
        }
        _ => {
            let model_json = model
                .as_ref()
                .map(Model::summary_json)
                .unwrap_or_else(|| "null".to_string());
            let config_json = config
                .as_ref()
                .map(Config::summary_json)
                .unwrap_or_else(|| "null".to_string());
            let data_json = data
                .as_ref()
                .map(CsvData::summary_json)
                .unwrap_or_else(|| "null".to_string());
            println!("{{\"ok\":true,\"command\":\"inspect\",\"model\":{model_json},\"config\":{config_json},\"data\":{data_json}}}");
        }
    }
    Ok(())
}

fn required<'a>(opts: &'a BTreeMap<String, String>, name: &str) -> CliResult<&'a str> {
    opts.get(name)
        .map(String::as_str)
        .ok_or_else(|| format!("missing required option '--{name}'").into())
}

fn output_format(opts: &BTreeMap<String, String>) -> CliResult<String> {
    let output = opts
        .get("output")
        .cloned()
        .unwrap_or_else(|| "json".to_string());
    match output.as_str() {
        "json" | "csv" => Ok(output),
        other => {
            Err(format!("unsupported output format '{other}', expected 'json' or 'csv'").into())
        }
    }
}

#[derive(Default)]
struct Config {
    target: Option<String>,
    n_estimators: Option<usize>,
    learning_rate: Option<f64>,
    max_depth: Option<usize>,
    min_samples_leaf: Option<usize>,
    min_gain: Option<f64>,
    splitter: Option<String>,
}

impl Config {
    fn summary_json(&self) -> String {
        format!(
            "{{\"target\":{}}}",
            optional_json_string(self.target.as_ref())
        )
    }

    fn booster_config(&self) -> BoosterConfig {
        let defaults = BoosterConfig::default();
        BoosterConfig {
            n_estimators: self.n_estimators.unwrap_or(defaults.n_estimators),
            learning_rate: self.learning_rate.unwrap_or(defaults.learning_rate),
            max_depth: self.max_depth.unwrap_or(defaults.max_depth),
            min_samples_leaf: self.min_samples_leaf.unwrap_or(defaults.min_samples_leaf),
            min_gain: self.min_gain.unwrap_or(defaults.min_gain),
            splitters: self
                .splitter
                .as_deref()
                .map(cli_splitters)
                .unwrap_or(defaults.splitters),
        }
    }
}

fn read_config(path: &str) -> CliResult<Config> {
    let text = fs::read_to_string(path)?;
    let mut config = Config::default();
    for line in text.lines() {
        let line = line.trim();
        if line.starts_with('#') || line.is_empty() {
            continue;
        }
        if let Some((key, value)) = line.split_once('=') {
            match key.trim() {
                "target" | "target_column" => config.target = Some(trim_toml_string(value)),
                "n_estimators" => config.n_estimators = trim_toml_string(value).parse().ok(),
                "learning_rate" => config.learning_rate = trim_toml_string(value).parse().ok(),
                "max_depth" => config.max_depth = trim_toml_string(value).parse().ok(),
                "min_samples_leaf" => {
                    config.min_samples_leaf = trim_toml_string(value).parse().ok()
                }
                "min_gain" => config.min_gain = trim_toml_string(value).parse().ok(),
                "splitter" | "splitters" => config.splitter = Some(trim_toml_string(value)),
                _ => {}
            }
        }
    }
    Ok(config)
}

fn cli_splitters(value: &str) -> Vec<SplitterKind> {
    let splitters = value
        .split(',')
        .map(str::trim)
        .filter_map(|name| match name {
            "axis" => Some(SplitterKind::Axis),
            "diagonal_2d" | "diagonal2d" => Some(SplitterKind::Diagonal2D),
            "gaussian_2d" | "gaussian2d" | "radial" => Some(SplitterKind::Gaussian2D),
            "periodic_time" | "periodic_24" => Some(SplitterKind::Periodic { period: 24.0 }),
            _ => None,
        })
        .collect::<Vec<_>>();
    if splitters.is_empty() {
        vec![SplitterKind::Axis]
    } else {
        splitters
    }
}

fn trim_toml_string(value: &str) -> String {
    value
        .trim()
        .trim_matches('"')
        .trim_matches('\'')
        .to_string()
}

struct CsvData {
    headers: Vec<String>,
    records: Vec<Vec<String>>,
}

impl CsvData {
    fn summary_json(&self) -> String {
        format!(
            "{{\"rows\":{},\"columns\":{}}}",
            self.records.len(),
            self.headers.len()
        )
    }
}

fn read_csv(path: &str) -> CliResult<CsvData> {
    let text = fs::read_to_string(path)?;
    let mut lines = text.lines().filter(|line| !line.trim().is_empty());
    let headers = lines
        .next()
        .map(parse_csv_line)
        .ok_or_else(|| format!("CSV file '{}' is empty", Path::new(path).display()))?;
    let records = lines.map(parse_csv_line).collect();
    Ok(CsvData { headers, records })
}

fn numeric_dataset_and_target(
    rows: &CsvData,
    target: &str,
) -> CliResult<(Dataset, Vec<f64>, usize)> {
    let target_idx = rows
        .headers
        .iter()
        .position(|name| name == target)
        .ok_or_else(|| format!("target column '{target}' not found"))?;
    let feature_indices = rows
        .headers
        .iter()
        .enumerate()
        .filter_map(|(idx, _)| (idx != target_idx).then_some(idx))
        .collect::<Vec<_>>();
    let mut x = Vec::with_capacity(rows.records.len());
    let mut y = Vec::with_capacity(rows.records.len());
    for record in &rows.records {
        x.push(parse_numeric_row(record, &feature_indices)?);
        y.push(parse_numeric_cell(record, target_idx)?);
    }
    let feature_count = feature_indices.len();
    Ok((Dataset::from_rows(x)?, y, feature_count))
}

fn numeric_dataset_without_target(rows: &CsvData, feature_count: usize) -> CliResult<Dataset> {
    let feature_indices = (0..feature_count.min(rows.headers.len())).collect::<Vec<_>>();
    if feature_indices.len() != feature_count {
        return Err(format!(
            "input has {} columns but model expects {feature_count} features",
            rows.headers.len()
        )
        .into());
    }
    let x = rows
        .records
        .iter()
        .map(|record| parse_numeric_row(record, &feature_indices))
        .collect::<CliResult<Vec<_>>>()?;
    Ok(Dataset::from_rows(x)?)
}

fn parse_numeric_row(record: &[String], indices: &[usize]) -> CliResult<Vec<f64>> {
    indices
        .iter()
        .map(|idx| parse_numeric_cell(record, *idx))
        .collect()
}

fn parse_numeric_cell(record: &[String], idx: usize) -> CliResult<f64> {
    record
        .get(idx)
        .ok_or_else(|| format!("row is missing column index {idx}").into())
        .and_then(|value| {
            value
                .parse::<f64>()
                .map_err(|err| format!("failed to parse numeric value '{value}': {err}").into())
        })
}

fn parse_csv_line(line: &str) -> Vec<String> {
    let mut values = Vec::new();
    let mut cell = String::new();
    let mut quoted = false;
    let mut chars = line.chars().peekable();
    while let Some(ch) = chars.next() {
        match ch {
            '"' if quoted && chars.peek() == Some(&'"') => {
                cell.push('"');
                chars.next();
            }
            '"' => quoted = !quoted,
            ',' if !quoted => {
                values.push(cell.trim().to_string());
                cell.clear();
            }
            _ => cell.push(ch),
        }
    }
    values.push(cell.trim().to_string());
    values
}

#[derive(Clone)]
enum Baseline {
    Numeric(f64),
    Text(String),
}

impl Baseline {
    fn as_string(&self) -> String {
        match self {
            Baseline::Numeric(value) => format_float(*value),
            Baseline::Text(value) => value.clone(),
        }
    }

    fn numeric(&self) -> Option<f64> {
        match self {
            Baseline::Numeric(value) => Some(*value),
            Baseline::Text(_) => None,
        }
    }

    fn to_json_value(&self) -> String {
        match self {
            Baseline::Numeric(value) => format_float(*value),
            Baseline::Text(value) => format!("\"{}\"", json_escape(value)),
        }
    }
}

struct Model {
    target: String,
    rows: usize,
    feature_count: usize,
    baseline: Baseline,
}

impl Model {
    fn summary_json(&self) -> String {
        format!(
            "{{\"model_type\":\"constant_baseline\",\"rows\":{},\"features\":{},\"target\":\"{}\",\"baseline\":{}}}",
            self.rows,
            self.feature_count,
            json_escape(&self.target),
            self.baseline.to_json_value()
        )
    }
}

fn mean_absolute_error(rows: &CsvData, target_idx: usize, prediction: Option<f64>) -> Option<f64> {
    let prediction = prediction?;
    let mut total = 0.0;
    let mut count = 0usize;
    for record in &rows.records {
        if let Some(actual) = record
            .get(target_idx)
            .and_then(|value| value.parse::<f64>().ok())
        {
            total += (actual - prediction).abs();
            count += 1;
        }
    }
    (count > 0).then_some(total / count as f64)
}

fn read_model(path: &str) -> CliResult<Model> {
    let text = fs::read_to_string(path)?;
    let target = json_string_field(&text, "target").unwrap_or_else(|| "target".to_string());
    let rows = json_usize_field(&text, "rows").unwrap_or(0);
    let feature_count = json_usize_field(&text, "features").unwrap_or(0);
    let baseline = if let Some(value) = json_number_field(&text, "baseline") {
        Baseline::Numeric(value)
    } else {
        Baseline::Text(json_string_field(&text, "baseline").unwrap_or_default())
    };
    Ok(Model {
        target,
        rows,
        feature_count,
        baseline,
    })
}

fn json_string_field(text: &str, field: &str) -> Option<String> {
    let key = format!("\"{field}\"");
    let after_key = text.split(&key).nth(1)?;
    let after_colon = after_key.split_once(':')?.1.trim_start();
    if !after_colon.starts_with('"') {
        return None;
    }
    let mut out = String::new();
    let mut escaped = false;
    for ch in after_colon[1..].chars() {
        if escaped {
            out.push(ch);
            escaped = false;
        } else if ch == '\\' {
            escaped = true;
        } else if ch == '"' {
            return Some(out);
        } else {
            out.push(ch);
        }
    }
    None
}

fn json_usize_field(text: &str, field: &str) -> Option<usize> {
    json_number_token(text, field)?.parse().ok()
}

fn json_number_field(text: &str, field: &str) -> Option<f64> {
    json_number_token(text, field)?.parse().ok()
}

fn json_number_token<'a>(text: &'a str, field: &str) -> Option<&'a str> {
    let key = format!("\"{field}\"");
    let after_key = text.split(&key).nth(1)?;
    let after_colon = after_key.split_once(':')?.1.trim_start();
    let end = after_colon
        .find(|ch: char| !(ch.is_ascii_digit() || ch == '.' || ch == '-'))
        .unwrap_or(after_colon.len());
    (end > 0).then_some(&after_colon[..end])
}

fn json_escape(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
}

fn optional_json_string(value: Option<&String>) -> String {
    value
        .map(|value| format!("\"{}\"", json_escape(value)))
        .unwrap_or_else(|| "null".to_string())
}

fn csv_cell(value: &str) -> String {
    if value.contains(',') || value.contains('"') || value.contains('\n') {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_string()
    }
}

fn format_float(value: f64) -> String {
    let formatted = format!("{value:.6}");
    formatted
        .trim_end_matches('0')
        .trim_end_matches('.')
        .to_string()
}
