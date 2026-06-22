import {useCallback, useEffect, useMemo, useState} from 'react';
import useBaseUrl from '@docusaurus/useBaseUrl';

import styles from './styles.module.css';

type ParsedTable = {
  columns: string[];
  rows: Record<string, string>[];
  fileName: string;
};

type ForecastRecord = {
  series_id: string;
  timestamp: string;
  horizon: number;
  model: string;
  prediction: number;
};

type WasmModule = {
  default: (
    input?: string | URL | Request | Response | BufferSource | WebAssembly.Module | {module_or_path: string},
  ) => Promise<unknown>;
  runForecast: (request: unknown) => ForecastResponse;
  runRegressionModel: (request: unknown) => RegressionResponse;
  runNeuralModel: (request: unknown) => RegressionResponse;
  availableForecastModels?: () => WasmModelMetadata[];
};

type ParquetWasmModule = {
  default: () => Promise<unknown>;
  readParquet: (data: Uint8Array) => {intoIPCStream: () => Uint8Array; free?: () => void};
};

type ForecastResponse = {
  metadata: {
    model: string;
    input: {
      n_rows: number;
      is_panel: boolean;
      series_ids: string[];
      frequency: string;
    };
  };
  forecast: {
    records: ForecastRecord[];
  };
};

type ComparisonResult = {
  requestedModel: string;
  label: string;
  pipeline: string;
  response?: ForecastResponse;
  error?: string;
};

type BacktestResult = {
  requestedModel: string;
  label: string;
  pipeline: string;
  rmse?: number;
  mae?: number;
  wape?: number;
  comparedRows?: number;
  response?: ForecastResponse;
  error?: string;
};

type RegressionResponse = {
  metadata: {
    model: string;
    featureNames?: string[];
    sparseFeatureNames?: string[];
    denseFeatureNames?: string[];
    pipeline?: string;
    splitterMode?: string;
    loss?: string;
    treeCount: number;
  };
  metrics: {
    rmse: number;
    mae: number;
    r2: number;
    trainRows: number;
    holdoutRows: number;
  };
  predictions: {
    rowIndex: number;
    actual: number;
    prediction: number;
    lowerPrediction?: number;
    upperPrediction?: number;
    residual: number;
  }[];
  featureImportance: {
    feature: string;
    splitCount: number;
  }[];
};

type ModelOption = {
  value: string;
  label: string;
  group: string;
};

type WasmModelMetadata = {
  name: string;
  label: string;
  pipeline: string;
};

const fallbackModelOptions: ModelOption[] = [
  {value: 'auto_forecast', label: 'CartoBoost AutoForecast', group: 'global'},
  {value: 'cartoboost_lag', label: 'CartoBoost Lag', group: 'global'},
  {value: 'cartoboost_direct', label: 'CartoBoost Direct', group: 'global'},
  {value: 'rectified_recursive', label: 'Rectified Recursive', group: 'global'},
  {value: 'lag_plus', label: 'Lag Plus', group: 'global'},
  {value: 'scaled_cartoboost_lag', label: 'Scaled CartoBoost Lag', group: 'transform'},
  {value: 'log1p_cartoboost_lag', label: 'Log1p CartoBoost Lag', group: 'transform'},
  {value: 'classical_expert_bank', label: 'Classical Expert Bank', group: 'selection'},
  {value: 'autostats_bank', label: 'AutoStats Bank', group: 'selection'},
  {value: 'intermittent_demand', label: 'Intermittent Demand', group: 'demand'},
  {value: 'stl_cartoboost', label: 'STL + ARIMA', group: 'decomposition'},
  {value: 'mstl_cartoboost', label: 'MSTL + ARIMA', group: 'decomposition'},
  {value: 'seasonal_naive', label: 'Seasonal Naive', group: 'local'},
  {value: 'window_average', label: 'Window Average', group: 'local'},
  {value: 'seasonal_window_average', label: 'Seasonal Window Average', group: 'local'},
  {value: 'theta', label: 'Theta', group: 'local'},
  {value: 'auto_ets', label: 'Auto ETS', group: 'local'},
  {value: 'ets', label: 'ETS', group: 'local'},
  {value: 'seasonal_ets', label: 'Seasonal ETS', group: 'local'},
  {value: 'auto_arima', label: 'Auto ARIMA', group: 'local'},
  {value: 'arima', label: 'ARIMA', group: 'local'},
  {value: 'kalman', label: 'Kalman', group: 'local'},
  {value: 'local_level_kalman', label: 'Local Level Kalman', group: 'local'},
  {value: 'auto_kalman', label: 'Auto Kalman', group: 'local'},
  {value: 'auto_local_level_kalman', label: 'Auto Local Level Kalman', group: 'local'},
  {value: 'kriging', label: 'Kriging', group: 'spatial'},
  {value: 'optimized_theta', label: 'Optimized Theta', group: 'local'},
  {value: 'naive', label: 'Naive', group: 'local'},
];

const forecastPipelineLabels: Record<string, string> = {
  global: 'CartoBoost native and global',
  transform: 'CartoBoost transformed',
  selection: 'Selector banks',
  demand: 'Sparse demand',
  decomposition: 'Decomposition',
  spatial: 'Geographic',
  local: 'Local statistical',
};

const neuralPipelineLabels: Record<string, string> = {
  embedding: 'Embedding regressor',
  node2vec: 'Node2Vec graph',
  graphsage: 'GraphSAGE graph',
  hetero_graphsage: 'HeteroGraphSAGE typed graph',
  hinsage: 'HinSAGE source-target graph',
};

const graphNeuralPipelines = new Set(['node2vec', 'graphsage', 'hetero_graphsage', 'hinsage']);

const sampleCsv = buildSampleCsv();

export default function ForecastLabClient(): React.ReactElement {
  const wasmJsUrl = useBaseUrl('/wasm/cartoboost/cartoboost_wasm.js');
  const wasmBinaryUrl = useBaseUrl('/wasm/cartoboost/cartoboost_wasm_bg.wasm');
  const [table, setTable] = useState<ParsedTable | null>(null);
  const [timestampCol, setTimestampCol] = useState('timestamp');
  const [targetCol, setTargetCol] = useState('target');
  const [seriesCol, setSeriesCol] = useState('series_id');
  const [frequency, setFrequency] = useState('daily');
  const [model, setModel] = useState('auto_forecast');
  const [horizon, setHorizon] = useState(14);
  const [seasonLength, setSeasonLength] = useState(7);
  const [result, setResult] = useState<ForecastResponse | null>(null);
  const [comparisonResults, setComparisonResults] = useState<ComparisonResult[]>([]);
  const [backtestResults, setBacktestResults] = useState<BacktestResult[]>([]);
  const [regressionResult, setRegressionResult] = useState<RegressionResponse | null>(null);
  const [neuralResult, setNeuralResult] = useState<RegressionResponse | null>(null);
  const [featureCols, setFeatureCols] = useState<string[]>([]);
  const [sparseFeatureCols, setSparseFeatureCols] = useState<string[]>([]);
  const [modelingMode, setModelingMode] = useState('full');
  const [modelingLoss, setModelingLoss] = useState('l2');
  const [neuralPipeline, setNeuralPipeline] = useState('embedding');
  const [neuralIdCol, setNeuralIdCol] = useState('');
  const [graphSourceCol, setGraphSourceCol] = useState('');
  const [graphTargetCol, setGraphTargetCol] = useState('');
  const [graphWeightCol, setGraphWeightCol] = useState('');
  const [status, setStatus] = useState('Drop a CSV, TSV, or Parquet file to start.');
  const [isRunning, setIsRunning] = useState(false);
  const [modelOptions, setModelOptions] = useState<ModelOption[]>(fallbackModelOptions);

  const previewRows = table?.rows.slice(0, 6) ?? [];
  const selectedColumnsReady =
    table !== null && timestampCol !== '' && targetCol !== '' && table.columns.includes(timestampCol) && table.columns.includes(targetCol);

  useEffect(() => {
    let cancelled = false;
    void getForecastModelOptions(wasmJsUrl, wasmBinaryUrl)
      .then((options) => {
        if (cancelled) {
          return;
        }
        setModelOptions(options);
        setModel((current) => (options.some((option) => option.value === current) ? current : options[0]?.value ?? 'auto_forecast'));
      })
      .catch(() => {
        if (!cancelled) {
          setModelOptions(fallbackModelOptions);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [wasmBinaryUrl, wasmJsUrl]);

  const loadText = useCallback((text: string, fileName: string) => {
    const parsed = parseDelimited(text, fileName);
    const nextTimestampCol = guessColumn(parsed.columns, ['timestamp', 'pickup_datetime', 'date', 'ds', 'time']) ?? parsed.columns[0] ?? '';
    const nextTargetCol = guessColumn(parsed.columns, ['target', 'y', 'demand', 'trips', 'count', 'fare', 'duration']) ?? parsed.columns[1] ?? '';
    const nextSeriesCol = guessColumn(parsed.columns, ['series_id', 'unique_id', 'PULocationID', 'DOLocationID', 'zone', 'route']) ?? '';
    setTable(parsed);
    setResult(null);
    setComparisonResults([]);
    setBacktestResults([]);
    setRegressionResult(null);
    setNeuralResult(null);
    setTimestampCol(nextTimestampCol);
    setTargetCol(nextTargetCol);
    setSeriesCol(nextSeriesCol);
    setFeatureCols(defaultFeatureColumns(parsed, nextTargetCol, nextTimestampCol, nextSeriesCol));
    setSparseFeatureCols(defaultSparseFeatureColumns(parsed, nextTargetCol, nextTimestampCol, nextSeriesCol));
    setNeuralIdCol(guessColumn(parsed.columns, ['pickup_zone_id', 'PULocationID', 'zone_id', 'id']) ?? '');
    setGraphSourceCol(guessColumn(parsed.columns, ['pickup_zone_id', 'PULocationID', 'source', 'source_id']) ?? '');
    setGraphTargetCol(guessColumn(parsed.columns, ['dropoff_zone_id', 'DOLocationID', 'target_node', 'target_id', 'destination']) ?? '');
    setGraphWeightCol(guessColumn(parsed.columns, ['edge_weight', 'weight', 'trip_count']) ?? '');
    setStatus(`Loaded ${parsed.rows.length.toLocaleString()} rows from ${fileName}.`);
  }, []);

  const loadFile = useCallback(
    async (file: File) => {
      if (file.name.toLowerCase().endsWith('.parquet')) {
        const parsed = await parseParquet(file);
        const nextTimestampCol = guessColumn(parsed.columns, ['timestamp', 'pickup_datetime', 'date', 'ds', 'time']) ?? parsed.columns[0] ?? '';
        const nextTargetCol = guessColumn(parsed.columns, ['target', 'y', 'demand', 'trips', 'count', 'fare', 'duration']) ?? parsed.columns[1] ?? '';
        const nextSeriesCol = guessColumn(parsed.columns, ['series_id', 'unique_id', 'PULocationID', 'DOLocationID', 'zone', 'route']) ?? '';
        setTable(parsed);
        setResult(null);
        setComparisonResults([]);
        setBacktestResults([]);
        setRegressionResult(null);
        setNeuralResult(null);
        setTimestampCol(nextTimestampCol);
        setTargetCol(nextTargetCol);
        setSeriesCol(nextSeriesCol);
        setFeatureCols(defaultFeatureColumns(parsed, nextTargetCol, nextTimestampCol, nextSeriesCol));
        setSparseFeatureCols(defaultSparseFeatureColumns(parsed, nextTargetCol, nextTimestampCol, nextSeriesCol));
        setNeuralIdCol(guessColumn(parsed.columns, ['pickup_zone_id', 'PULocationID', 'zone_id', 'id']) ?? '');
        setGraphSourceCol(guessColumn(parsed.columns, ['pickup_zone_id', 'PULocationID', 'source', 'source_id']) ?? '');
        setGraphTargetCol(guessColumn(parsed.columns, ['dropoff_zone_id', 'DOLocationID', 'target_node', 'target_id', 'destination']) ?? '');
        setGraphWeightCol(guessColumn(parsed.columns, ['edge_weight', 'weight', 'trip_count']) ?? '');
        setStatus(`Loaded ${parsed.rows.length.toLocaleString()} rows from ${file.name}.`);
        return;
      }
      const text = await file.text();
      loadText(text, file.name);
    },
    [loadText],
  );

  const onDrop = useCallback(
    async (event: React.DragEvent<HTMLLabelElement>) => {
      event.preventDefault();
      const file = event.dataTransfer.files.item(0);
      if (file) {
        await loadFile(file);
      }
    },
    [loadFile],
  );

  const runForecast = useCallback(async () => {
    if (!table || !selectedColumnsReady) {
      setStatus('Choose timestamp and target columns before running a forecast.');
      return;
    }
    setIsRunning(true);
    setStatus('Loading CartoBoost WebAssembly and fitting the forecast.');
    try {
      const response = await runBrowserForecast({
        wasmJsUrl,
        wasmBinaryUrl,
        table,
        timestampCol,
        targetCol,
        seriesCol,
        frequency,
        horizon,
        model,
        seasonLength,
      });
      setResult(response);
      setComparisonResults([]);
      setBacktestResults([]);
      setRegressionResult(null);
      setNeuralResult(null);
      setStatus(`Forecast complete with ${response.forecast.records.length.toLocaleString()} prediction rows.`);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : String(error));
      setResult(null);
    } finally {
      setIsRunning(false);
    }
  }, [
    frequency,
    horizon,
    model,
    seasonLength,
    selectedColumnsReady,
    seriesCol,
    table,
    targetCol,
    timestampCol,
    wasmBinaryUrl,
    wasmJsUrl,
  ]);

  const runComparison = useCallback(async () => {
    if (!table || !selectedColumnsReady) {
      setStatus('Choose timestamp and target columns before running a comparison.');
      return;
    }
    setIsRunning(true);
    setResult(null);
    setBacktestResults([]);
    setRegressionResult(null);
    setNeuralResult(null);
    setComparisonResults([]);
    setStatus('Running native model roster.');
    const roster = modelOptions.filter((option) => option.value !== 'kriging' || hasCoordinateColumns(table.columns));
    const nextResults: ComparisonResult[] = [];
    for (const option of roster) {
      setStatus(`Running ${option.label}.`);
      try {
        const response = await runBrowserForecast({
          wasmJsUrl,
          wasmBinaryUrl,
          table,
          timestampCol,
          targetCol,
          seriesCol,
          frequency,
          horizon,
          model: option.value,
          seasonLength,
        });
        nextResults.push({
          requestedModel: option.value,
          label: option.label,
          pipeline: option.group,
          response,
        });
      } catch (error) {
        nextResults.push({
          requestedModel: option.value,
          label: option.label,
          pipeline: option.group,
          error: error instanceof Error ? error.message : String(error),
        });
      }
      setComparisonResults([...nextResults]);
    }
    const successes = nextResults.filter((item) => item.response).length;
    setStatus(`Model roster complete: ${successes.toLocaleString()} succeeded, ${(nextResults.length - successes).toLocaleString()} reported constraints.`);
    setIsRunning(false);
  }, [
    frequency,
    horizon,
    modelOptions,
    seasonLength,
    selectedColumnsReady,
    seriesCol,
    table,
    targetCol,
    timestampCol,
    wasmBinaryUrl,
    wasmJsUrl,
  ]);

  const runBacktest = useCallback(async () => {
    if (!table || !selectedColumnsReady) {
      setStatus('Choose timestamp and target columns before running a holdout backtest.');
      return;
    }
    setIsRunning(true);
    setResult(null);
    setComparisonResults([]);
    setBacktestResults([]);
    setRegressionResult(null);
    setNeuralResult(null);
    setStatus('Running holdout backtest across native roster.');
    try {
      const split = holdoutSplit(table, timestampCol, targetCol, seriesCol, horizon);
      const roster = modelOptions.filter((option) => option.value !== 'kriging' || hasCoordinateColumns(table.columns));
      const nextResults: BacktestResult[] = [];
      for (const option of roster) {
        setStatus(`Backtesting ${option.label}.`);
        try {
          const response = await runBrowserForecast({
            wasmJsUrl,
            wasmBinaryUrl,
            table: split.train,
            timestampCol,
            targetCol,
            seriesCol,
            frequency,
            horizon,
            model: option.value,
            seasonLength,
          });
          const metrics = evaluateHoldout(response.forecast.records, split.actuals);
          nextResults.push({
            requestedModel: option.value,
            label: option.label,
            pipeline: option.group,
            response,
            ...metrics,
          });
        } catch (error) {
          nextResults.push({
            requestedModel: option.value,
            label: option.label,
            pipeline: option.group,
            error: error instanceof Error ? error.message : String(error),
          });
        }
        setBacktestResults([...nextResults]);
      }
      const successes = nextResults.filter((item) => item.rmse !== undefined).length;
      setStatus(`Holdout backtest complete: ${successes.toLocaleString()} scored, ${(nextResults.length - successes).toLocaleString()} reported constraints.`);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : String(error));
    } finally {
      setIsRunning(false);
    }
  }, [
    frequency,
    horizon,
    modelOptions,
    seasonLength,
    selectedColumnsReady,
    seriesCol,
    table,
    targetCol,
    timestampCol,
    wasmBinaryUrl,
    wasmJsUrl,
  ]);

  const runRegression = useCallback(async () => {
    if (!table || !selectedColumnsReady || featureCols.length === 0) {
      setStatus('Choose a target and at least one numeric feature before running modeling.');
      return;
    }
    setIsRunning(true);
    setResult(null);
    setComparisonResults([]);
    setBacktestResults([]);
    setRegressionResult(null);
    setNeuralResult(null);
    setStatus('Fitting CartoBoost regression in WebAssembly.');
    try {
      const response = await runBrowserRegression({
        wasmJsUrl,
        wasmBinaryUrl,
        table,
        targetCol,
        featureCols,
        sparseFeatureCols,
        modelingMode,
        modelingLoss,
      });
      setRegressionResult(response);
      setStatus(`Modeling complete: ${response.metrics.trainRows.toLocaleString()} train rows, ${response.metrics.holdoutRows.toLocaleString()} holdout rows.`);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : String(error));
    } finally {
      setIsRunning(false);
    }
  }, [featureCols, modelingLoss, modelingMode, selectedColumnsReady, sparseFeatureCols, table, targetCol, wasmBinaryUrl, wasmJsUrl]);

  const runNeural = useCallback(async () => {
    if (!table || !selectedColumnsReady || featureCols.length === 0) {
      setStatus('Choose a target and at least one dense feature before running neural modeling.');
      return;
    }
    if (neuralPipeline === 'embedding' && !neuralIdCol) {
      setStatus('Choose an ID column before running embedding modeling.');
      return;
    }
    if (graphNeuralPipelines.has(neuralPipeline) && (!graphSourceCol || !graphTargetCol)) {
      setStatus(`Choose source and target node columns before running ${neuralPipelineLabels[neuralPipeline] ?? neuralPipeline}.`);
      return;
    }
    setIsRunning(true);
    setResult(null);
    setComparisonResults([]);
    setBacktestResults([]);
    setRegressionResult(null);
    setNeuralResult(null);
    setStatus(`Fitting ${neuralPipelineLabels[neuralPipeline] ?? neuralPipeline} in WebAssembly.`);
    try {
      const response = await runBrowserNeural({
        wasmJsUrl,
        wasmBinaryUrl,
        table,
        targetCol,
        featureCols,
        neuralPipeline,
        neuralIdCol,
        graphSourceCol,
        graphTargetCol,
        graphWeightCol,
      });
      setNeuralResult(response);
      setStatus(`Neural modeling complete: ${response.metrics.trainRows.toLocaleString()} train rows, ${response.metrics.holdoutRows.toLocaleString()} holdout rows.`);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : String(error));
    } finally {
      setIsRunning(false);
    }
  }, [
    featureCols,
    graphSourceCol,
    graphTargetCol,
    graphWeightCol,
    neuralIdCol,
    neuralPipeline,
    selectedColumnsReady,
    table,
    targetCol,
    wasmBinaryUrl,
    wasmJsUrl,
  ]);

  const exportSuggestedConfig = useCallback(() => {
    if (!table) {
      setStatus('Load a dataset before exporting a suggested config.');
      return;
    }
    const config = buildSuggestedConfig({
      table,
      timestampCol,
      targetCol,
      seriesCol,
      frequency,
      horizon,
      seasonLength,
      model,
      modelOptions,
      featureCols,
      sparseFeatureCols,
      modelingMode,
      modelingLoss,
      neuralPipeline,
      neuralIdCol,
      graphSourceCol,
      graphTargetCol,
      graphWeightCol,
    });
    downloadJson(`${stripExtension(table.fileName)}-cartoboost-config.json`, config);
    setStatus(`Exported suggested config for ${table.fileName}.`);
  }, [featureCols, frequency, graphSourceCol, graphTargetCol, graphWeightCol, horizon, model, modelOptions, modelingLoss, modelingMode, neuralIdCol, neuralPipeline, seasonLength, seriesCol, sparseFeatureCols, table, targetCol, timestampCol]);

  const chartRows = useMemo(() => {
    if (!result) {
      return [];
    }
    const firstSeries = result.forecast.records[0]?.series_id;
    return result.forecast.records.filter((row) => row.series_id === firstSeries);
  }, [result]);
  const actualRows = useMemo(
    () => actualRowsForFirstSeries(table, timestampCol, targetCol, seriesCol),
    [seriesCol, table, targetCol, timestampCol],
  );

  return (
    <main className={styles.shell}>
      <section className={styles.header}>
        <div>
          <span className={styles.eyebrow}>WebAssembly forecast lab</span>
          <h1>Drag in demand data and forecast in the browser</h1>
          <p>
            This page runs CartoBoost's Rust forecasting core locally through WebAssembly. No dataset
            leaves the browser.
          </p>
        </div>
        <button className={styles.secondaryButton} type="button" onClick={() => loadText(sampleCsv, 'sample-taxi-demand.csv')}>
          Load sample
        </button>
      </section>

      <section className={styles.workspace}>
        <div className={styles.controlPanel}>
          <label
            className={styles.dropZone}
            onDragOver={(event) => event.preventDefault()}
            onDrop={onDrop}
          >
            <input
              accept=".csv,.tsv,.parquet,text/csv,text/tab-separated-values,application/vnd.apache.parquet"
              type="file"
              onChange={(event) => {
                const file = event.target.files?.item(0);
                if (file) {
                  void loadFile(file);
                }
              }}
            />
            <strong>{table ? table.fileName : 'Drop CSV, TSV, or Parquet'}</strong>
            <span>{table ? `${table.rows.length.toLocaleString()} rows ready` : 'Choose CSV, TSV, or Parquet, then map timestamp, target, and series columns.'}</span>
          </label>

          <div className={styles.controlStack}>
            <ControlSection title="Dataset mapping">
              <div className={styles.controlsGrid}>
                <Select label="Timestamp" value={timestampCol} onChange={setTimestampCol} options={table?.columns ?? []} />
                <Select label="Target" value={targetCol} onChange={setTargetCol} options={table?.columns ?? []} />
                <Select label="Series" value={seriesCol} onChange={setSeriesCol} options={table?.columns ?? []} allowBlank blankLabel="Single series" />
              </div>
            </ControlSection>

            <ControlSection title="Forecast">
              <div className={styles.controlsGrid}>
                <Select label="Frequency" value={frequency} onChange={setFrequency} options={['hourly', 'daily', 'weekly']} />
                <GroupedSelect
                  label="Model"
                  value={model}
                  onChange={setModel}
                  groups={forecastModelGroups(modelOptions)}
                />
                <NumberInput label="Horizon" value={horizon} min={1} max={365} onChange={setHorizon} />
                <NumberInput label="Season Length" value={seasonLength} min={1} max={366} onChange={setSeasonLength} />
              </div>
            </ControlSection>

            <ControlSection title="Regression modeling">
              <div className={styles.controlsGrid}>
                <Select
                  label="Splitter Menu"
                  value={modelingMode}
                  onChange={setModelingMode}
                  options={['full', 'auto', 'axis', 'spatial', 'periodic']}
                  labels={{
                    full: 'Spatial + periodic toolkit',
                    auto: 'Auto dense',
                    axis: 'Axis only',
                    spatial: 'Spatial splitters',
                    periodic: 'Periodic splitters',
                  }}
                />
                <Select
                  label="Loss"
                  value={modelingLoss}
                  onChange={setModelingLoss}
                  options={['l2', 'l1', 'huber', 'log_l2', 'quantile']}
                  labels={{
                    l2: 'L2 mean',
                    l1: 'L1 median',
                    huber: 'Huber robust',
                    log_l2: 'Log-L2 positive',
                    quantile: 'Quantile median',
                  }}
                />
              </div>
            </ControlSection>

            <ControlSection title="Graph and neural">
              <div className={styles.controlsGrid}>
                <GroupedSelect
                  label="Neural Menu"
                  value={neuralPipeline}
                  onChange={setNeuralPipeline}
                  groups={[
                    {label: 'Embeddings', options: [{value: 'embedding', label: neuralPipelineLabels.embedding}]},
                    {
                      label: 'Graph pipelines',
                      options: [
                        {value: 'node2vec', label: neuralPipelineLabels.node2vec},
                        {value: 'graphsage', label: neuralPipelineLabels.graphsage},
                        {value: 'hetero_graphsage', label: neuralPipelineLabels.hetero_graphsage},
                        {value: 'hinsage', label: neuralPipelineLabels.hinsage},
                      ],
                    },
                  ]}
                />
                <Select label="ID" value={neuralIdCol} onChange={setNeuralIdCol} options={table?.columns ?? []} allowBlank blankLabel="No ID column" />
                <Select label="Graph Source" value={graphSourceCol} onChange={setGraphSourceCol} options={table?.columns ?? []} allowBlank blankLabel="No source column" />
                <Select label="Graph Target" value={graphTargetCol} onChange={setGraphTargetCol} options={table?.columns ?? []} allowBlank blankLabel="No target column" />
                <Select label="Graph Weight" value={graphWeightCol} onChange={setGraphWeightCol} options={table?.columns ?? []} allowBlank blankLabel="Unweighted graph" />
              </div>
            </ControlSection>
          </div>
          {table && (
            <FeatureSelector
              columns={numericFeatureColumns(table, targetCol, timestampCol, seriesCol)}
              selected={featureCols}
              onChange={setFeatureCols}
            />
          )}
          {table && (
            <FeatureSelector
              title="Sparse set features"
              columns={sparseFeatureColumns(table, targetCol, timestampCol, seriesCol)}
              selected={sparseFeatureCols}
              onChange={setSparseFeatureCols}
            />
          )}

          <button className={styles.primaryButton} type="button" disabled={!selectedColumnsReady || isRunning} onClick={() => void runForecast()}>
            {isRunning ? 'Running forecast' : 'Run forecast'}
          </button>
          <button className={styles.secondaryActionButton} type="button" disabled={!selectedColumnsReady || isRunning} onClick={() => void runComparison()}>
            Compare native roster
          </button>
          <button className={styles.secondaryActionButton} type="button" disabled={!selectedColumnsReady || isRunning} onClick={() => void runBacktest()}>
            Backtest native roster
          </button>
          <button className={styles.secondaryActionButton} type="button" disabled={!selectedColumnsReady || featureCols.length === 0 || isRunning} onClick={() => void runRegression()}>
            Run modeling pipeline
          </button>
          <button className={styles.secondaryActionButton} type="button" disabled={!selectedColumnsReady || featureCols.length === 0 || isRunning} onClick={() => void runNeural()}>
            Run neural pipeline
          </button>
          <button className={styles.secondaryActionButton} type="button" disabled={!table || isRunning} onClick={exportSuggestedConfig}>
            Export suggested config
          </button>
          <p className={styles.status}>{status}</p>
        </div>

        <div className={styles.outputPanel}>
          {neuralResult ? (
            <>
              <div className={styles.resultHeader}>
                <div>
                  <span className={styles.eyebrow}>Neural modeling</span>
                  <h2>{neuralResult.metadata.model}</h2>
                </div>
                <dl>
                  <div>
                    <dt>Train</dt>
                    <dd>{neuralResult.metrics.trainRows.toLocaleString()}</dd>
                  </div>
                  <div>
                    <dt>Holdout</dt>
                    <dd>{neuralResult.metrics.holdoutRows.toLocaleString()}</dd>
                  </div>
                </dl>
              </div>
              <RegressionMetricSummary result={neuralResult} />
              <RegressionPredictionChart result={neuralResult} />
              <FeatureImportanceChart result={neuralResult} />
              <RegressionPredictionTable result={neuralResult} />
            </>
          ) : regressionResult ? (
            <>
              <div className={styles.resultHeader}>
                <div>
                  <span className={styles.eyebrow}>Modeling</span>
                  <h2>CartoBoost regression</h2>
                </div>
                <dl>
                  <div>
                    <dt>Train</dt>
                    <dd>{regressionResult.metrics.trainRows.toLocaleString()}</dd>
                  </div>
                  <div>
                    <dt>Holdout</dt>
                    <dd>{regressionResult.metrics.holdoutRows.toLocaleString()}</dd>
                  </div>
                </dl>
              </div>
              <RegressionMetricSummary result={regressionResult} />
              <RegressionPredictionChart result={regressionResult} />
              <FeatureImportanceChart result={regressionResult} />
              <RegressionPredictionTable result={regressionResult} />
            </>
          ) : backtestResults.length > 0 ? (
            <>
              <div className={styles.resultHeader}>
                <div>
                  <span className={styles.eyebrow}>Backtest</span>
                  <h2>Holdout metrics</h2>
                </div>
                <dl>
                  <div>
                    <dt>Scored</dt>
                    <dd>{backtestResults.filter((item) => item.rmse !== undefined).length}</dd>
                  </div>
                  <div>
                    <dt>Checked</dt>
                    <dd>{backtestResults.length}</dd>
                  </div>
                </dl>
              </div>
              <BacktestMetricChart results={backtestResults} />
              <BacktestTable results={backtestResults} />
            </>
          ) : comparisonResults.length > 0 ? (
            <>
              <div className={styles.resultHeader}>
                <div>
                  <span className={styles.eyebrow}>Comparison</span>
                  <h2>Native roster</h2>
                </div>
                <dl>
                  <div>
                    <dt>Passed</dt>
                    <dd>{comparisonResults.filter((item) => item.response).length}</dd>
                  </div>
                  <div>
                    <dt>Checked</dt>
                    <dd>{comparisonResults.length}</dd>
                  </div>
                </dl>
              </div>
              <ComparisonChart actualRows={actualRows} results={comparisonResults} />
              <ComparisonTable results={comparisonResults} />
            </>
          ) : result ? (
            <>
              <div className={styles.resultHeader}>
                <div>
                  <span className={styles.eyebrow}>Result</span>
                  <h2>{result.metadata.model}</h2>
                </div>
                <dl>
                  <div>
                    <dt>Rows</dt>
                    <dd>{result.metadata.input.n_rows.toLocaleString()}</dd>
                  </div>
                  <div>
                    <dt>Series</dt>
                    <dd>{result.metadata.input.series_ids.length.toLocaleString()}</dd>
                  </div>
                </dl>
              </div>
              <ForecastChart actualRows={actualRows} rows={chartRows} />
              <ForecastTable records={result.forecast.records.slice(0, 12)} />
            </>
          ) : (
            <>
              <div className={styles.emptyState}>
                <span className={styles.eyebrow}>Preview</span>
                <h2>{table ? 'Dataset loaded' : 'Waiting for data'}</h2>
              </div>
              {table && (
                <>
                  <DatasetProfile table={table} timestampCol={timestampCol} targetCol={targetCol} seriesCol={seriesCol} />
                  <GeoDatasetVisualization table={table} targetCol={targetCol} />
                  <PreviewTable columns={table.columns} rows={previewRows} />
                </>
              )}
            </>
          )}
        </div>
      </section>
    </main>
  );
}

type SelectGroup = {
  label: string;
  options: {value: string; label: string}[];
};

function ControlSection({title, children}: {title: string; children: React.ReactNode}) {
  return (
    <section className={styles.controlSection}>
      <h2>{title}</h2>
      {children}
    </section>
  );
}

function Select({
  label,
  value,
  options,
  labels = {},
  allowBlank = false,
  blankLabel = 'None',
  onChange,
}: {
  label: string;
  value: string;
  options: string[];
  labels?: Record<string, string>;
  allowBlank?: boolean;
  blankLabel?: string;
  onChange: (value: string) => void;
}) {
  return (
    <label className={styles.field}>
      <span>{label}</span>
      <select value={value} onChange={(event) => onChange(event.target.value)}>
        {allowBlank && <option value="">{blankLabel}</option>}
        {options.map((option) => (
          <option value={option} key={option}>
            {labels[option] ?? option}
          </option>
        ))}
      </select>
    </label>
  );
}

function GroupedSelect({
  label,
  value,
  groups,
  onChange,
}: {
  label: string;
  value: string;
  groups: SelectGroup[];
  onChange: (value: string) => void;
}) {
  return (
    <label className={styles.field}>
      <span>{label}</span>
      <select value={value} onChange={(event) => onChange(event.target.value)}>
        {groups.map((group) => (
          <optgroup label={group.label} key={group.label}>
            {group.options.map((option) => (
              <option value={option.value} key={option.value}>
                {option.label}
              </option>
            ))}
          </optgroup>
        ))}
      </select>
    </label>
  );
}

function NumberInput({
  label,
  value,
  min,
  max,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  onChange: (value: number) => void;
}) {
  return (
    <label className={styles.field}>
      <span>{label}</span>
      <input
        type="number"
        min={min}
        max={max}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
      />
    </label>
  );
}

function forecastModelGroups(modelOptions: ModelOption[]): SelectGroup[] {
  const grouped = new Map<string, {value: string; label: string}[]>();
  for (const option of modelOptions) {
    const groupLabel = forecastPipelineLabels[option.group] ?? option.group;
    const group = grouped.get(groupLabel) ?? [];
    group.push({value: option.value, label: `${option.label} (${option.value})`});
    grouped.set(groupLabel, group);
  }
  return Array.from(grouped, ([label, options]) => ({label, options}));
}

function FeatureSelector({
  title = 'Model features',
  columns,
  selected,
  onChange,
}: {
  title?: string;
  columns: string[];
  selected: string[];
  onChange: (value: string[]) => void;
}) {
  if (columns.length === 0) {
    return null;
  }
  return (
    <div className={styles.featureSelector}>
      <span>{title}</span>
      <div>
        {columns.map((column) => (
          <label key={column}>
            <input
              type="checkbox"
              checked={selected.includes(column)}
              onChange={(event) => {
                if (event.target.checked) {
                  onChange([...selected, column]);
                } else {
                  onChange(selected.filter((item) => item !== column));
                }
              }}
            />
            {column}
          </label>
        ))}
      </div>
    </div>
  );
}

function ForecastChart({actualRows, rows}: {actualRows: ActualRecord[]; rows: ForecastRecord[]}) {
  if (rows.length === 0 && actualRows.length === 0) {
    return null;
  }
  const series = buildLineSeries(actualRows, [{label: rows[0]?.model ?? 'forecast', records: rows}]);
  return <LineChart caption={rows[0]?.series_id ? `First series: ${rows[0].series_id}` : 'Forecast'} series={series} />;
}

function RegressionMetricSummary({result}: {result: RegressionResponse}) {
  return (
    <div className={styles.metricCards}>
      <p>
        <span>RMSE</span>
        {result.metrics.rmse.toFixed(3)}
      </p>
      <p>
        <span>MAE</span>
        {result.metrics.mae.toFixed(3)}
      </p>
      <p>
        <span>R2</span>
        {result.metrics.r2.toFixed(3)}
      </p>
      <p>
        <span>Trees</span>
        {result.metadata.treeCount.toLocaleString()}
      </p>
      <p>
        <span>Mode</span>
        {result.metadata.splitterMode ?? 'auto'}
      </p>
      <p>
        <span>Loss</span>
        {result.metadata.loss ?? 'l2'}
      </p>
    </div>
  );
}

function RegressionPredictionChart({result}: {result: RegressionResponse}) {
  const values = result.predictions.flatMap((row) =>
    [row.actual, row.prediction, row.lowerPrediction, row.upperPrediction].filter((value): value is number => typeof value === 'number'),
  );
  if (values.length === 0) {
    return null;
  }
  const width = 820;
  const height = 300;
  const padding = 38;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || 1;
  const xFor = (value: number) => padding + ((value - min) / span) * (width - padding * 2);
  const yFor = (value: number) => height - padding - ((value - min) / span) * (height - padding * 2);
  return (
    <figure className={styles.chart}>
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Actual versus predicted holdout values">
        <line x1={padding} y1={height - padding} x2={width - padding} y2={height - padding} />
        <line x1={padding} y1={padding} x2={padding} y2={height - padding} />
        <line className={styles.referenceLine} x1={padding} y1={height - padding} x2={width - padding} y2={padding} />
        {result.predictions
          .filter((row) => typeof row.lowerPrediction === 'number' && typeof row.upperPrediction === 'number')
          .map((row) => (
            <line
              className={styles.intervalLine}
              x1={xFor(row.actual)}
              x2={xFor(row.actual)}
              y1={yFor(row.lowerPrediction as number)}
              y2={yFor(row.upperPrediction as number)}
              key={`interval-${row.rowIndex}`}
            />
          ))}
        {result.predictions.map((row) => (
          <circle cx={xFor(row.actual)} cy={yFor(row.prediction)} r="4" key={row.rowIndex} />
        ))}
      </svg>
      <figcaption>Actual versus predicted holdout values</figcaption>
    </figure>
  );
}

function FeatureImportanceChart({result}: {result: RegressionResponse}) {
  const rows = result.featureImportance.filter((item) => item.splitCount > 0).slice(0, 12);
  if (rows.length === 0) {
    return null;
  }
  const maxSplits = Math.max(...rows.map((row) => row.splitCount), 1);
  return (
    <figure className={styles.metricChart}>
      <figcaption>Split-count feature importance</figcaption>
      {rows.map((row, index) => (
        <div className={styles.metricRow} key={row.feature}>
          <span>{index + 1}. {row.feature}</span>
          <div>
            <i style={{width: `${Math.max((row.splitCount / maxSplits) * 100, 2)}%`}} />
          </div>
          <strong>{row.splitCount.toLocaleString()}</strong>
        </div>
      ))}
    </figure>
  );
}

function RegressionPredictionTable({result}: {result: RegressionResponse}) {
  return (
    <div className={styles.tableScroller}>
      <table>
        <thead>
          <tr>
            <th>Row</th>
            <th>Actual</th>
            <th>Prediction</th>
            <th>Lower</th>
            <th>Upper</th>
            <th>Residual</th>
          </tr>
        </thead>
        <tbody>
          {result.predictions.slice(0, 20).map((row) => (
            <tr key={row.rowIndex}>
              <td>{row.rowIndex.toLocaleString()}</td>
              <td>{row.actual.toFixed(3)}</td>
              <td>{row.prediction.toFixed(3)}</td>
              <td>{typeof row.lowerPrediction === 'number' ? row.lowerPrediction.toFixed(3) : '-'}</td>
              <td>{typeof row.upperPrediction === 'number' ? row.upperPrediction.toFixed(3) : '-'}</td>
              <td>{row.residual.toFixed(3)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ComparisonChart({actualRows, results}: {actualRows: ActualRecord[]; results: ComparisonResult[]}) {
  const forecastSeries = results
    .filter((result) => result.response)
    .slice(0, 8)
    .map((result) => ({
      label: result.label,
      records: firstSeriesForecastRows(result.response as ForecastResponse),
    }));
  const series = buildLineSeries(actualRows, forecastSeries);
  return <LineChart caption="First series comparison" series={series} />;
}

function LineChart({caption, series}: {caption: string; series: ChartSeries[]}) {
  const drawable = series.filter((item) => item.points.length > 0);
  if (drawable.length === 0) {
    return null;
  }
  const width = 820;
  const height = 300;
  const padding = 38;
  const values = drawable.flatMap((item) => item.points.map((point) => point.value));
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || 1;
  const maxIndex = Math.max(...drawable.flatMap((item) => item.points.map((point) => point.index)), 1);

  return (
    <figure className={styles.chart}>
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={caption}>
        <line x1={padding} y1={height - padding} x2={width - padding} y2={height - padding} />
        <line x1={padding} y1={padding} x2={padding} y2={height - padding} />
        {drawable.map((item, seriesIndex) => {
          const points = item.points
            .map((point) => {
              const x = padding + (point.index / maxIndex) * (width - padding * 2);
              const y = height - padding - ((point.value - min) / span) * (height - padding * 2);
              return `${x},${y}`;
            })
            .join(' ');
          return (
            <polyline
              className={seriesIndex === 0 ? styles.actualLine : styles.forecastLine}
              points={points}
              style={{stroke: chartColor(seriesIndex)}}
              key={item.label}
            />
          );
        })}
      </svg>
      <figcaption>{caption}</figcaption>
      <div className={styles.legend}>
        {drawable.map((item, index) => (
          <span key={item.label}>
            <i style={{background: chartColor(index)}} />
            {item.label}
          </span>
        ))}
      </div>
    </figure>
  );
}

function PreviewTable({columns, rows}: {columns: string[]; rows: Record<string, string>[]}) {
  return (
    <div className={styles.tableScroller}>
      <table>
        <thead>
          <tr>{columns.slice(0, 8).map((column) => <th key={column}>{column}</th>)}</tr>
        </thead>
        <tbody>
          {rows.map((row, index) => (
            <tr key={index}>
              {columns.slice(0, 8).map((column) => <td key={column}>{row[column]}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function DatasetProfile({
  table,
  timestampCol,
  targetCol,
  seriesCol,
}: {
  table: ParsedTable;
  timestampCol: string;
  targetCol: string;
  seriesCol: string;
}) {
  const numericColumns = table.columns.filter((column) =>
    table.rows.some((row) => {
      const value = row[column]?.trim();
      return Boolean(value) && Number.isFinite(Number(value));
    }),
  );
  const seriesCount = seriesCol
    ? new Set(table.rows.map((row) => row[seriesCol]).filter(Boolean)).size
    : 1;
  const targetValues = table.rows
    .map((row) => Number(row[targetCol]))
    .filter((value) => Number.isFinite(value));
  const timestamps = table.rows
    .map((row) => row[timestampCol])
    .filter(Boolean)
    .sort((a, b) => a.localeCompare(b));
  const minTarget = targetValues.length ? Math.min(...targetValues) : null;
  const maxTarget = targetValues.length ? Math.max(...targetValues) : null;
  return (
    <div className={styles.datasetProfile}>
      <dl>
        <div>
          <dt>Rows</dt>
          <dd>{table.rows.length.toLocaleString()}</dd>
        </div>
        <div>
          <dt>Columns</dt>
          <dd>{table.columns.length.toLocaleString()}</dd>
        </div>
        <div>
          <dt>Series</dt>
          <dd>{seriesCount.toLocaleString()}</dd>
        </div>
        <div>
          <dt>Numeric</dt>
          <dd>{numericColumns.length.toLocaleString()}</dd>
        </div>
      </dl>
      <div className={styles.profileGrid}>
        <p>
          <span>Time range</span>
          {timestamps[0] ?? '-'} to {timestamps[timestamps.length - 1] ?? '-'}
        </p>
        <p>
          <span>Target range</span>
          {minTarget === null || maxTarget === null ? '-' : `${minTarget.toFixed(3)} to ${maxTarget.toFixed(3)}`}
        </p>
      </div>
      <div className={styles.columnChips}>
        {table.columns.map((column) => (
          <span
            className={
              column === timestampCol || column === targetCol || column === seriesCol
                ? styles.columnChipSelected
                : undefined
            }
            key={column}
          >
            {column}
          </span>
        ))}
      </div>
    </div>
  );
}

function GeoDatasetVisualization({table, targetCol}: {table: ParsedTable; targetCol: string}) {
  const pickup = coordinatePair(table.columns, ['pickup', 'pu', 'origin', '']);
  const dropoff = coordinatePair(table.columns, ['dropoff', 'do', 'destination']);
  const h3Columns = table.columns.filter((column) => {
    const normalized = column.toLowerCase();
    return normalized.includes('h3') || table.rows.some((row) => isH3Like(row[column] ?? ''));
  });
  if (!pickup && h3Columns.length === 0) {
    return null;
  }
  const points = pickup
    ? table.rows
        .map((row) => ({
          lon: Number(row[pickup.lon]),
          lat: Number(row[pickup.lat]),
          target: Number(row[targetCol]),
          dropLon: dropoff ? Number(row[dropoff.lon]) : Number.NaN,
          dropLat: dropoff ? Number(row[dropoff.lat]) : Number.NaN,
        }))
        .filter((point) => Number.isFinite(point.lon) && Number.isFinite(point.lat) && Number.isFinite(point.target))
        .slice(0, 600)
    : [];
  const routePoints = points.filter((point) => Number.isFinite(point.dropLon) && Number.isFinite(point.dropLat)).slice(0, 140);
  const allLon = points.flatMap((point) => (Number.isFinite(point.dropLon) ? [point.lon, point.dropLon] : [point.lon]));
  const allLat = points.flatMap((point) => (Number.isFinite(point.dropLat) ? [point.lat, point.dropLat] : [point.lat]));
  const minLon = allLon.length ? Math.min(...allLon) : 0;
  const maxLon = allLon.length ? Math.max(...allLon) : 1;
  const minLat = allLat.length ? Math.min(...allLat) : 0;
  const maxLat = allLat.length ? Math.max(...allLat) : 1;
  const targetValues = points.map((point) => point.target);
  const minTarget = targetValues.length ? Math.min(...targetValues) : 0;
  const maxTarget = targetValues.length ? Math.max(...targetValues) : 1;
  const width = 820;
  const height = 330;
  const padding = 28;
  const xFor = (lon: number) => padding + ((lon - minLon) / (maxLon - minLon || 1)) * (width - padding * 2);
  const yFor = (lat: number) => height - padding - ((lat - minLat) / (maxLat - minLat || 1)) * (height - padding * 2);
  const h3Rows = h3Columns
    .flatMap((column) => h3Summary(table, column).map((row) => ({...row, column})))
    .filter((row) => h3CellLabel(row) !== '' && h3CellCount(row) > 0)
    .slice(0, 10);
  const h3DisplayRows =
    h3Rows.length > 0
      ? h3Rows
      : h3Columns
          .map((column) => ({
            column,
            cell: 'detected column',
            count: table.rows.filter((row) => (row[column]?.trim() ?? '') !== '').length || table.rows.length,
          }))
          .slice(0, 4);
  return (
    <section className={styles.geoPanel}>
      <div className={styles.geoHeader}>
        <div>
          <span className={styles.eyebrow}>Geography</span>
          <h3>{pickup ? 'Spatial demand view' : 'H3 coverage view'}</h3>
        </div>
        <span>{points.length > 0 ? `${points.length.toLocaleString()} plotted rows` : `${h3DisplayRows.length.toLocaleString()} H3 cells`}</span>
      </div>
      {points.length > 0 && (
        <svg className={styles.geoSvg} viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Geographic dataset preview">
          <rect x="0" y="0" width={width} height={height} />
          {routePoints.map((point, index) => (
            <line
              x1={xFor(point.lon)}
              y1={yFor(point.lat)}
              x2={xFor(point.dropLon)}
              y2={yFor(point.dropLat)}
              key={`route-${index}`}
            />
          ))}
          {points.map((point, index) => (
            <circle
              cx={xFor(point.lon)}
              cy={yFor(point.lat)}
              r="4"
              style={{fill: targetColor(point.target, minTarget, maxTarget)}}
              key={`point-${index}`}
            />
          ))}
        </svg>
      )}
      {h3DisplayRows.length > 0 && (
        <div className={styles.h3List}>
          {h3DisplayRows.map((row) => (
            <span key={`${row.column}-${h3CellLabel(row)}`}>
              <strong>{row.column}</strong>
              {h3CellLabel(row)}
              <i>{h3CellCount(row).toLocaleString()}</i>
            </span>
          ))}
        </div>
      )}
    </section>
  );
}

function ForecastTable({records}: {records: ForecastRecord[]}) {
  return (
    <div className={styles.tableScroller}>
      <table>
        <thead>
          <tr>
            <th>series_id</th>
            <th>timestamp</th>
            <th>horizon</th>
            <th>prediction</th>
          </tr>
        </thead>
        <tbody>
          {records.map((record) => (
            <tr key={`${record.series_id}-${record.timestamp}-${record.horizon}`}>
              <td>{record.series_id}</td>
              <td>{record.timestamp}</td>
              <td>{record.horizon}</td>
              <td>{record.prediction.toFixed(3)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ComparisonTable({results}: {results: ComparisonResult[]}) {
  return (
    <div className={styles.tableScroller}>
      <table>
        <thead>
          <tr>
            <th>Model</th>
            <th>Pipeline</th>
            <th>Rows</th>
            <th>First prediction</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody>
          {results.map((result) => {
            const first = result.response?.forecast.records[0];
            return (
              <tr key={result.requestedModel}>
                <td>{result.label}</td>
                <td>{result.pipeline}</td>
                <td>{result.response ? result.response.forecast.records.length.toLocaleString() : '-'}</td>
                <td>{first ? first.prediction.toFixed(3) : '-'}</td>
                <td>{result.error ?? 'ok'}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function BacktestMetricChart({results}: {results: BacktestResult[]}) {
  const scored = [...results]
    .filter((result) => result.rmse !== undefined)
    .sort((a, b) => (a.rmse as number) - (b.rmse as number))
    .slice(0, 12);
  if (scored.length === 0) {
    return null;
  }
  const maxRmse = Math.max(...scored.map((result) => result.rmse as number), 1);
  return (
    <figure className={styles.metricChart}>
      <figcaption>RMSE by model, lower is better</figcaption>
      {scored.map((result, index) => (
        <div className={styles.metricRow} key={result.requestedModel}>
          <span>{index + 1}. {result.label}</span>
          <div>
            <i style={{width: `${Math.max(((result.rmse as number) / maxRmse) * 100, 2)}%`}} />
          </div>
          <strong>{(result.rmse as number).toFixed(3)}</strong>
        </div>
      ))}
    </figure>
  );
}

function BacktestTable({results}: {results: BacktestResult[]}) {
  return (
    <div className={styles.tableScroller}>
      <table>
        <thead>
          <tr>
            <th>Model</th>
            <th>Pipeline</th>
            <th>RMSE</th>
            <th>MAE</th>
            <th>WAPE</th>
            <th>Rows</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody>
          {[...results].sort(sortBacktestRows).map((result) => (
            <tr key={result.requestedModel}>
              <td>{result.label}</td>
              <td>{result.pipeline}</td>
              <td>{formatMetric(result.rmse)}</td>
              <td>{formatMetric(result.mae)}</td>
              <td>{result.wape === undefined ? '-' : `${(result.wape * 100).toFixed(2)}%`}</td>
              <td>{result.comparedRows?.toLocaleString() ?? '-'}</td>
              <td>{result.error ?? 'ok'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function sortBacktestRows(a: BacktestResult, b: BacktestResult) {
  if (a.rmse === undefined && b.rmse === undefined) {
    return a.label.localeCompare(b.label);
  }
  if (a.rmse === undefined) {
    return 1;
  }
  if (b.rmse === undefined) {
    return -1;
  }
  return a.rmse - b.rmse;
}

function formatMetric(value: number | undefined) {
  return value === undefined ? '-' : value.toFixed(3);
}

function guessColumn(columns: string[], candidates: string[]) {
  const normalized = new Map(columns.map((column) => [column.toLowerCase(), column]));
  for (const candidate of candidates) {
    const match = normalized.get(candidate.toLowerCase());
    if (match) {
      return match;
    }
  }
  return null;
}

function coordinatePair(columns: string[], prefixes: string[]) {
  for (const prefix of prefixes) {
    const prefixPart = prefix ? `${prefix}_` : '';
    const lat = guessColumn(columns, [
      `${prefixPart}latitude`,
      `${prefixPart}lat`,
      `${prefixPart}y`,
      prefix ? `${prefix}Latitude` : 'latitude',
      prefix ? `${prefix}Lat` : 'lat',
    ]);
    const lon = guessColumn(columns, [
      `${prefixPart}longitude`,
      `${prefixPart}lon`,
      `${prefixPart}lng`,
      `${prefixPart}x`,
      prefix ? `${prefix}Longitude` : 'longitude',
      prefix ? `${prefix}Lon` : 'lon',
    ]);
    if (lat && lon) {
      return {lat, lon};
    }
  }
  return null;
}

function h3Summary(table: ParsedTable, column: string) {
  const counts = new Map<string, number>();
  const acceptNamedH3Values = column.toLowerCase().includes('h3');
  for (const row of table.rows) {
    const cells = row[column]
      ?.split(/[|;,\s]+/)
      .map((cell) => cell.trim())
      .filter((cell) => cell !== '' && (acceptNamedH3Values || isH3Like(cell)));
    for (const cell of cells ?? []) {
      counts.set(cell, (counts.get(cell) ?? 0) + 1);
    }
  }
  return [...counts.entries()]
    .sort((left, right) => right[1] - left[1] || left[0].localeCompare(right[0]))
    .slice(0, 8)
    .map(([cell, count]) => ({cell, count}));
}

function h3CellLabel(row: {cell?: string; count?: number; 0?: string; 1?: number}) {
  return row.cell ?? row[0] ?? '';
}

function h3CellCount(row: {cell?: string; count?: number; 0?: string; 1?: number}) {
  return Number(row.count ?? row[1] ?? 0);
}

function targetColor(value: number, min: number, max: number) {
  const ratio = Math.max(0, Math.min(1, (value - min) / (max - min || 1)));
  const hue = 190 - ratio * 150;
  return `hsl(${hue.toFixed(1)} 78% 48%)`;
}

function defaultFeatureColumns(table: ParsedTable, targetCol: string, timestampCol: string, seriesCol: string) {
  return numericFeatureColumns(table, targetCol, timestampCol, seriesCol).slice(0, 12);
}

function defaultSparseFeatureColumns(table: ParsedTable, targetCol: string, timestampCol: string, seriesCol: string) {
  return sparseFeatureColumns(table, targetCol, timestampCol, seriesCol).slice(0, 4);
}

function numericFeatureColumns(table: ParsedTable, targetCol: string, timestampCol: string, seriesCol: string) {
  const excluded = new Set([targetCol, timestampCol, seriesCol].filter(Boolean));
  return table.columns.filter((column) => {
    if (excluded.has(column)) {
      return false;
    }
    return table.rows.some((row) => {
      const value = row[column]?.trim();
      return Boolean(value) && Number.isFinite(Number(value));
    });
  });
}

function sparseFeatureColumns(table: ParsedTable, targetCol: string, timestampCol: string, seriesCol: string) {
  const excluded = new Set([targetCol, timestampCol, seriesCol].filter(Boolean));
  return table.columns.filter((column) => {
    if (excluded.has(column) || numericFeatureColumns(table, targetCol, timestampCol, seriesCol).includes(column)) {
      return false;
    }
    return looksSparseSetColumn(table, column);
  });
}

function looksSparseSetColumn(table: ParsedTable, column: string) {
  const normalized = column.toLowerCase();
  const namedLikeSparse =
    normalized.includes('h3') ||
    normalized.includes('s2') ||
    normalized.includes('membership') ||
    normalized.includes('zone_ids') ||
    normalized.includes('cell') ||
    normalized.includes('cell_ids') ||
    normalized.includes('sparse') ||
    normalized.endsWith('_set') ||
    normalized.endsWith('_ids');
  return table.rows.some((row) => {
    const value = row[column]?.trim() ?? '';
    return value !== '' && (namedLikeSparse || /[|;,\s]/.test(value) || isH3Like(value)) && parseSparseSet(value).length > 0;
  });
}

function parseSparseSet(value: string) {
  return Array.from(
    new Set(
      value
        .split(/[|;,\s]+/)
        .map((item) => item.trim())
        .filter(Boolean)
        .map((item) => {
          const numeric = Number(item);
          if (Number.isInteger(numeric) && numeric >= 0) {
            return numeric;
          }
          return isH3Like(item) ? stableStringId(item) : Number.NaN;
        })
        .filter((item) => Number.isInteger(item) && item >= 0),
    ),
  );
}

function isH3Like(value: string) {
  const normalized = value.trim().toLowerCase().replace(/^0x/, '');
  return /^[0-9a-f]{15,16}$/.test(normalized);
}

function inferFeatureKind(column: string) {
  const normalized = column.toLowerCase();
  if (
    normalized.includes('hour') ||
    normalized === 'dow' ||
    normalized.includes('day_of_week') ||
    normalized.includes('weekday')
  ) {
    return 'periodic';
  }
  if (
    normalized.includes('lat') ||
    normalized.includes('lon') ||
    normalized.includes('longitude') ||
    normalized.includes('latitude') ||
    normalized.endsWith('_x') ||
    normalized.endsWith('_y')
  ) {
    return 'spatial';
  }
  return 'numeric';
}

function inferPeriodicPeriod(column: string) {
  return column.toLowerCase().includes('hour') ? 24 : 7;
}

function buildSuggestedConfig({
  table,
  timestampCol,
  targetCol,
  seriesCol,
  frequency,
  horizon,
  seasonLength,
  model,
  modelOptions,
  featureCols,
  sparseFeatureCols,
  modelingMode,
  modelingLoss,
  neuralPipeline,
  neuralIdCol,
  graphSourceCol,
  graphTargetCol,
  graphWeightCol,
}: {
  table: ParsedTable;
  timestampCol: string;
  targetCol: string;
  seriesCol: string;
  frequency: string;
  horizon: number;
  seasonLength: number;
  model: string;
  modelOptions: ModelOption[];
  featureCols: string[];
  sparseFeatureCols: string[];
  modelingMode: string;
  modelingLoss: string;
  neuralPipeline: string;
  neuralIdCol: string;
  graphSourceCol: string;
  graphTargetCol: string;
  graphWeightCol: string;
}) {
  const featureKinds = Object.fromEntries(featureCols.map((column) => [column, inferFeatureKind(column)]));
  const sparseFeatureKinds = Object.fromEntries(sparseFeatureCols.map((column) => [column, 'sparse_set']));
  const periodicPeriods = Object.fromEntries(
    featureCols
      .filter((column) => inferFeatureKind(column) === 'periodic')
      .map((column) => [column, inferPeriodicPeriod(column)]),
  );
  const selectedModel = modelOptions.find((option) => option.value === model);
  return {
    cartoboostConfigVersion: 1,
    source: {
      fileName: table.fileName,
      rowCount: table.rows.length,
      columns: table.columns,
      acceptedFormats: ['csv', 'tsv', 'parquet'],
    },
    browserWasm: {
      page: '/forecast-lab',
      crate: 'cartoboost-wasm',
      entrypoints: ['runForecast', 'runRegressionModel', 'runNeuralModel', 'availableForecastModels'],
    },
    geography: {
      enabled: true,
      h3FeatureCols: table.columns.filter((column) => table.rows.some((row) => isH3Like(row[column] ?? ''))),
      coordinatePairs: {
        pickup: coordinatePair(table.columns, ['pickup', 'pu', 'origin', '']),
        dropoff: coordinatePair(table.columns, ['dropoff', 'do', 'destination']),
      },
      visualizations: ['spatial_target_scatter', 'route_endpoint_lines', 'h3_cell_frequency'],
    },
    forecast: {
      enabled: true,
      timestampCol,
      targetCol,
      seriesIdCol: seriesCol || null,
      frequency,
      horizon,
      model,
      modelLabel: selectedModel?.label ?? model,
      pipeline: selectedModel?.group ?? 'custom',
      options: {
        seasonLength,
      },
      roster: modelOptions.map((option) => ({
        model: option.value,
        label: option.label,
        pipeline: option.group,
      })),
    },
    modeling: {
      enabled: featureCols.length > 0,
      targetCol,
      featureCols,
      sparseFeatureCols,
      splitterMode: modelingMode,
      loss: modelingLoss,
      featureKinds,
      sparseFeatureKinds,
      sparseSetFormat: {
        delimiter: 'one or more of | ; , or whitespace',
        valueType: 'non-negative integer ids',
      },
      periodicPeriods,
      holdoutFraction: 0.2,
      booster: {
        nEstimators: 120,
        learningRate: 0.06,
        maxDepth: 3,
        minSamplesLeafRule: 'max(2, min(20, floor(finite_rows / 20)))',
        intervalLowerAlpha: 0.1,
        intervalUpperAlpha: 0.9,
      },
      metrics: ['rmse', 'mae', 'r2'],
      visualizations: ['actual_vs_predicted', 'split_count_feature_importance', 'prediction_table'],
    },
    neuralModeling: {
      enabled: featureCols.length > 0,
      pipeline: neuralPipeline,
      targetCol,
      denseFeatureCols: featureCols,
      idCol: neuralIdCol || null,
      graphSourceCol: graphSourceCol || null,
      graphTargetCol: graphTargetCol || null,
      graphWeightCol: graphWeightCol || null,
      holdoutFraction: 0.2,
      embeddingDim: 8,
      supportedPipelines: Object.keys(neuralPipelineLabels),
      node2vec: {
        walkLength: 8,
        walksPerNode: 4,
        windowSize: 3,
        epochs: 2,
      },
      graphSage: {
        epochs: 3,
        negativeSamples: 2,
        nodeFeatures: 'mean of selected denseFeatureCols by source/target node',
        nodeTypes: 'inferred from graph source/target column roles',
        edgeTypes: 'inferred from source node type and target node type',
      },
      visualizations: ['actual_vs_predicted', 'embedding_or_graph_split_importance', 'prediction_table'],
    },
  };
}

function downloadJson(fileName: string, payload: unknown) {
  const blob = new Blob([`${JSON.stringify(payload, null, 2)}\n`], {type: 'application/json'});
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = fileName;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function stripExtension(fileName: string) {
  return fileName.replace(/\.[^.]+$/, '') || 'dataset';
}

function numericCovariates(row: Record<string, string>, excludedColumns: string[]) {
  const excluded = new Set(excludedColumns.filter(Boolean));
  return Object.fromEntries(
    Object.entries(row)
      .filter(([column]) => !excluded.has(column))
      .map(([column, value]) => [column, Number(value)])
      .filter(([, value]) => Number.isFinite(value)),
  );
}

async function importExternalModule(url: string) {
  const dynamicImport = new Function('url', 'return import(url);') as (url: string) => Promise<unknown>;
  return dynamicImport(url);
}

let initializedWasmModule:
  | {
      key: string;
      promise: Promise<WasmModule>;
    }
  | null = null;

async function getInitializedWasmModule(wasmJsUrl: string, wasmBinaryUrl: string) {
  const key = `${wasmJsUrl}\u0000${wasmBinaryUrl}`;
  if (!initializedWasmModule || initializedWasmModule.key !== key) {
    initializedWasmModule = {
      key,
      promise: (async () => {
        const wasmModule = (await importExternalModule(wasmJsUrl)) as WasmModule;
        await wasmModule.default({module_or_path: wasmBinaryUrl});
        return wasmModule;
      })(),
    };
  }
  try {
    return await initializedWasmModule.promise;
  } catch (error) {
    initializedWasmModule = null;
    throw error;
  }
}

async function getForecastModelOptions(wasmJsUrl: string, wasmBinaryUrl: string): Promise<ModelOption[]> {
  const wasmModule = await getInitializedWasmModule(wasmJsUrl, wasmBinaryUrl);
  const registry = wasmModule.availableForecastModels?.() ?? [];
  const options = registry
    .filter((model) => model.name && model.label && model.pipeline)
    .map((model) => ({
      value: model.name,
      label: model.label,
      group: model.pipeline,
    }));
  return options.length > 0 ? options : fallbackModelOptions;
}

async function runBrowserForecast({
  wasmJsUrl,
  wasmBinaryUrl,
  table,
  timestampCol,
  targetCol,
  seriesCol,
  frequency,
  horizon,
  model,
  seasonLength,
}: {
  wasmJsUrl: string;
  wasmBinaryUrl: string;
  table: ParsedTable;
  timestampCol: string;
  targetCol: string;
  seriesCol: string;
  frequency: string;
  horizon: number;
  model: string;
  seasonLength: number;
}) {
  const wasmModule = await getInitializedWasmModule(wasmJsUrl, wasmBinaryUrl);
  return wasmModule.runForecast({
    rows: table.rows.map((row) => ({
      timestamp: row[timestampCol],
      target: Number(row[targetCol]),
      seriesId: seriesCol ? row[seriesCol] : undefined,
      covariates: numericCovariates(row, [timestampCol, targetCol, seriesCol]),
    })),
    frequency,
    horizon,
    model,
    options: {
      seasonLength,
    },
    metadata: {
      timestampCol,
      targetCol,
      seriesIdCol: seriesCol || undefined,
    },
  });
}

async function runBrowserRegression({
  wasmJsUrl,
  wasmBinaryUrl,
  table,
  targetCol,
  featureCols,
  sparseFeatureCols,
  modelingMode,
  modelingLoss,
}: {
  wasmJsUrl: string;
  wasmBinaryUrl: string;
  table: ParsedTable;
  targetCol: string;
  featureCols: string[];
  sparseFeatureCols: string[];
  modelingMode: string;
  modelingLoss: string;
}) {
  const rows = table.rows
    .map((row) => ({
      features: featureCols.map((column) => Number(row[column])),
      sparseSets: sparseFeatureCols.map((column) => parseSparseSet(row[column] ?? '')),
      target: Number(row[targetCol]),
    }))
    .filter((row) => Number.isFinite(row.target) && row.features.every(Number.isFinite));
  if (rows.length < 4) {
    throw new Error('CartoBoost modeling needs at least four rows with finite target and feature values.');
  }
  const wasmModule = await getInitializedWasmModule(wasmJsUrl, wasmBinaryUrl);
  return wasmModule.runRegressionModel({
    rows,
    featureNames: featureCols,
    sparseFeatureNames: sparseFeatureCols,
    options: {
      holdoutFraction: 0.2,
      splitterMode: modelingMode,
      loss: modelingLoss,
      quantileAlpha: 0.5,
      huberDelta: 8,
      logOffset: 1,
      intervalLowerAlpha: 0.1,
      intervalUpperAlpha: 0.9,
      featureKinds: Object.fromEntries(featureCols.map((column) => [column, inferFeatureKind(column)])),
      periodicPeriods: Object.fromEntries(
        featureCols
          .filter((column) => inferFeatureKind(column) === 'periodic')
          .map((column) => [column, inferPeriodicPeriod(column)]),
      ),
      nEstimators: 120,
      learningRate: 0.06,
      maxDepth: 3,
      minSamplesLeaf: Math.max(2, Math.min(20, Math.floor(rows.length / 20))),
    },
  });
}

async function runBrowserNeural({
  wasmJsUrl,
  wasmBinaryUrl,
  table,
  targetCol,
  featureCols,
  neuralPipeline,
  neuralIdCol,
  graphSourceCol,
  graphTargetCol,
  graphWeightCol,
}: {
  wasmJsUrl: string;
  wasmBinaryUrl: string;
  table: ParsedTable;
  targetCol: string;
  featureCols: string[];
  neuralPipeline: string;
  neuralIdCol: string;
  graphSourceCol: string;
  graphTargetCol: string;
  graphWeightCol: string;
}) {
  const graphTopology = buildGraphTopology(table, graphSourceCol, graphTargetCol, featureCols);
  const rows = table.rows
    .map((row, rowIndex) => ({
      id: neuralIdCol ? parseBrowserId(row[neuralIdCol]) : undefined,
      source: graphSourceCol ? graphTopology.nodeIndex.get(row[graphSourceCol]) : undefined,
      targetNode: graphTargetCol ? graphTopology.nodeIndex.get(row[graphTargetCol]) : undefined,
      edgeWeight: graphWeightCol ? Number(row[graphWeightCol]) : undefined,
      edgeType: graphTopology.edgeTypesByRow[rowIndex] ?? 0,
      dense: featureCols.map((column) => Number(row[column])),
      target: Number(row[targetCol]),
    }))
    .filter((row) => {
      const validCommon = Number.isFinite(row.target) && row.dense.every(Number.isFinite);
      if (!validCommon) {
        return false;
      }
      if (graphNeuralPipelines.has(neuralPipeline)) {
        return row.source !== undefined && row.targetNode !== undefined && (row.edgeWeight === undefined || Number.isFinite(row.edgeWeight));
      }
      return row.id !== undefined;
    });
  if (rows.length < 4) {
    throw new Error('CartoBoost neural modeling needs at least four usable rows.');
  }
  const wasmModule = await getInitializedWasmModule(wasmJsUrl, wasmBinaryUrl);
  return wasmModule.runNeuralModel({
    rows,
    denseFeatureNames: featureCols,
    nodeFeatures: graphTopology.nodeFeatures,
    nodeTypes: graphTopology.nodeTypes,
    edgeTypeTriples: graphTopology.edgeTypeTriples,
    pipeline: neuralPipeline,
    options: {
      holdoutFraction: 0.2,
      embeddingDim: 8,
      randomState: 42,
      nEstimators: 80,
      learningRate: 0.07,
      maxDepth: 4,
      minSamplesLeaf: Math.max(2, Math.min(20, Math.floor(rows.length / 20))),
      node2vecWalkLength: 8,
      node2vecWalksPerNode: 4,
      node2vecWindowSize: 3,
      node2vecEpochs: 2,
      node2vecSeed: 42,
      graphSageEpochs: 3,
      graphSageNegativeSamples: 2,
      graphSageSeed: 42,
    },
  });
}

function parseBrowserId(value: string | undefined) {
  if (!value) {
    return undefined;
  }
  const numeric = Number(value);
  if (Number.isInteger(numeric) && numeric >= 0) {
    return numeric;
  }
  return stableStringId(value);
}

function buildNodeIndex(table: ParsedTable, columns: string[]) {
  const index = new Map<string, number>();
  for (const column of columns.filter(Boolean)) {
    for (const row of table.rows) {
      const value = row[column];
      if (value && !index.has(value)) {
        index.set(value, index.size);
      }
    }
  }
  return index;
}

function buildGraphTopology(table: ParsedTable, sourceCol: string, targetCol: string, featureCols: string[]) {
  const nodeIndex = buildNodeIndex(table, [sourceCol, targetCol]);
  const width = Math.max(1, featureCols.length);
  const sums = Array.from({length: nodeIndex.size}, () => Array.from({length: width}, () => 0));
  const counts = Array.from({length: nodeIndex.size}, () => 0);
  const sourceNodes = new Set<number>();
  const targetNodes = new Set<number>();
  for (const row of table.rows) {
    const dense = featureCols.map((column) => Number(row[column]));
    const values = dense.length > 0 ? dense : [1];
    if (!values.every(Number.isFinite)) {
      continue;
    }
    const source = sourceCol ? nodeIndex.get(row[sourceCol]) : undefined;
    const target = targetCol ? nodeIndex.get(row[targetCol]) : undefined;
    for (const node of [source, target]) {
      if (node === undefined) {
        continue;
      }
      counts[node] += 1;
      for (const [index, value] of values.entries()) {
        sums[node][index] += value;
      }
    }
    if (source !== undefined) {
      sourceNodes.add(source);
    }
    if (target !== undefined) {
      targetNodes.add(target);
    }
  }
  const nodeFeatures = sums.map((row, index) => {
    const count = counts[index] || 1;
    return row.map((value) => value / count);
  });
  const nodeTypes = Array.from({length: nodeIndex.size}, (_, index) => (targetNodes.has(index) && !sourceNodes.has(index) ? 1 : 0));
  const relationIndex = new Map<string, number>();
  const edgeTypeTriples: number[][] = [];
  const edgeTypesByRow = table.rows.map((row) => {
    const source = sourceCol ? nodeIndex.get(row[sourceCol]) : undefined;
    const target = targetCol ? nodeIndex.get(row[targetCol]) : undefined;
    const sourceType = source === undefined ? 0 : nodeTypes[source] ?? 0;
    const targetType = target === undefined ? 0 : nodeTypes[target] ?? 0;
    const key = `${sourceType}->${targetType}`;
    const existing = relationIndex.get(key);
    if (existing !== undefined) {
      return existing;
    }
    const next = relationIndex.size;
    relationIndex.set(key, next);
    edgeTypeTriples.push([sourceType, next, targetType]);
    return next;
  });
  return {
    nodeIndex,
    nodeFeatures,
    nodeTypes,
    edgeTypesByRow,
    edgeTypeTriples: edgeTypeTriples.length > 0 ? edgeTypeTriples : [[0, 0, 0]],
  };
}

function stableStringId(value: string) {
  let hash = 2166136261;
  for (const char of value.trim()) {
    hash ^= char.codePointAt(0) ?? 0;
    hash = Math.imul(hash, 16777619) >>> 0;
  }
  return hash;
}

type ActualRecord = {
  series_id: string;
  timestamp: string;
  index: number;
  value: number;
};

type ChartSeries = {
  label: string;
  points: {index: number; value: number}[];
};

function actualRowsForFirstSeries(
  table: ParsedTable | null,
  timestampCol: string,
  targetCol: string,
  seriesCol: string,
): ActualRecord[] {
  if (!table || !table.columns.includes(timestampCol) || !table.columns.includes(targetCol)) {
    return [];
  }
  const firstSeries = seriesCol ? table.rows[0]?.[seriesCol] : '__single__';
  return table.rows
    .filter((row) => !seriesCol || row[seriesCol] === firstSeries)
    .map((row, index) => ({
      series_id: seriesCol ? row[seriesCol] : '__single__',
      timestamp: row[timestampCol],
      index,
      value: Number(row[targetCol]),
    }))
    .filter((row) => Number.isFinite(row.value));
}

function firstSeriesForecastRows(response: ForecastResponse) {
  const firstSeries = response.forecast.records[0]?.series_id;
  return response.forecast.records.filter((row) => row.series_id === firstSeries);
}

function buildLineSeries(
  actualRows: ActualRecord[],
  forecastSeries: {label: string; records: ForecastRecord[]}[],
): ChartSeries[] {
  const actualLength = actualRows.length;
  return [
    {
      label: 'Actual',
      points: actualRows.map((row) => ({index: row.index, value: row.value})),
    },
    ...forecastSeries.map((series) => ({
      label: series.label,
      points: series.records.map((row) => ({
        index: actualLength + row.horizon - 1,
        value: row.prediction,
      })),
    })),
  ];
}

function chartColor(index: number) {
  return ['#526274', '#168f86', '#ffb454', '#4f8cff', '#d84d7f', '#8b6cff', '#2f9d62', '#a15c38', '#65c9ff'][index % 9];
}

function hasCoordinateColumns(columns: string[]) {
  const normalized = new Set(columns.map((column) => column.toLowerCase()));
  return (
    (normalized.has('longitude') || normalized.has('lon') || normalized.has('lng') || normalized.has('x')) &&
    (normalized.has('latitude') || normalized.has('lat') || normalized.has('y'))
  );
}

type HoldoutActual = {
  series_id: string;
  timestamp: string;
  value: number;
};

function holdoutSplit(
  table: ParsedTable,
  timestampCol: string,
  targetCol: string,
  seriesCol: string,
  horizon: number,
) {
  const groups = new Map<string, Record<string, string>[]>();
  for (const row of table.rows) {
    const seriesId = seriesCol ? row[seriesCol] : '__single__';
    const rows = groups.get(seriesId) ?? [];
    rows.push(row);
    groups.set(seriesId, rows);
  }
  const trainRows: Record<string, string>[] = [];
  const actuals: HoldoutActual[] = [];
  for (const [seriesId, rows] of groups) {
    if (rows.length <= horizon + 2) {
      throw new Error(`Series ${seriesId} needs more than ${horizon + 2} rows for a ${horizon}-step holdout backtest.`);
    }
    const sortedRows = [...rows].sort((a, b) => a[timestampCol].localeCompare(b[timestampCol]));
    const cutoff = sortedRows.length - horizon;
    trainRows.push(...sortedRows.slice(0, cutoff));
    actuals.push(
      ...sortedRows.slice(cutoff).map((row) => ({
        series_id: seriesId,
        timestamp: normalizeTimestamp(row[timestampCol]),
        value: Number(row[targetCol]),
      })),
    );
  }
  return {
    train: {
      ...table,
      fileName: `${table.fileName} holdout train`,
      rows: trainRows,
    },
    actuals: actuals.filter((actual) => Number.isFinite(actual.value)),
  };
}

function evaluateHoldout(predictions: ForecastRecord[], actuals: HoldoutActual[]) {
  const actualByKey = new Map(actuals.map((actual) => [`${actual.series_id}\u0000${actual.timestamp}`, actual]));
  const pairs = predictions
    .map((prediction) => ({
      prediction,
      actual: actualByKey.get(`${prediction.series_id}\u0000${normalizeTimestamp(prediction.timestamp)}`),
    }))
    .filter((pair): pair is {prediction: ForecastRecord; actual: HoldoutActual} => pair.actual !== undefined);
  if (pairs.length === 0) {
    throw new Error('No forecast rows aligned with the holdout timestamps.');
  }
  const absErrors = pairs.map((pair) => Math.abs(pair.prediction.prediction - pair.actual.value));
  const squaredErrors = pairs.map((pair) => (pair.prediction.prediction - pair.actual.value) ** 2);
  const actualMagnitude = pairs.reduce((sum, pair) => sum + Math.abs(pair.actual.value), 0);
  return {
    rmse: Math.sqrt(squaredErrors.reduce((sum, value) => sum + value, 0) / pairs.length),
    mae: absErrors.reduce((sum, value) => sum + value, 0) / pairs.length,
    wape: actualMagnitude === 0 ? 0 : absErrors.reduce((sum, value) => sum + value, 0) / actualMagnitude,
    comparedRows: pairs.length,
  };
}

function normalizeTimestamp(value: string) {
  return value.length === 10 ? `${value}T00:00:00` : value.replace(' ', 'T').replace(/\.\d{3}Z$/, '');
}

function buildSampleCsv() {
  const rows = [
    'timestamp,series_id,target,trip_distance,pickup_hour,route_pressure,pickup_zone_id,dropoff_zone_id,pickup_lon,pickup_lat,dropoff_lon,dropoff_lat,h3_cell,zone_memberships,edge_weight',
  ];
  const h3Cells = ['882a100d2dfffff', '882a100d69fffff', '882a1072c7fffff', '882a1008b3fffff'];
  for (let index = 0; index < 56; index += 1) {
    const timestamp = new Date(Date.UTC(2026, 0, 1 + index));
    const date = timestamp.toISOString().slice(0, 10);
    const tripDistance = 1.5 + (index % 9) * 0.4 + index * 0.03;
    const pickupHour = (index * 3) % 24;
    const routePressure = (index * 7) % 13;
    const pickupZoneId = 101 + (index % 4);
    const dropoffZoneId = 205 + ((index * 3) % 4);
    const pickupLon = -73.98 + Math.sin(index / 6) * 0.04;
    const pickupLat = 40.74 + Math.cos(index / 7) * 0.03;
    const dropoffLon = -73.94 + Math.cos(index / 5) * 0.05;
    const dropoffLat = 40.71 + Math.sin(index / 8) * 0.04;
    const trend = 124 + index * 2.5;
    const weekdayLift = [0, 5, 18, 23, 11, 16, 34][index % 7];
    const spatialLift = (pickupLon + 74.0) * 110 + (pickupLat - 40.7) * 140 + (dropoffLon + 74.0) * 60;
    const h3Cell = h3Cells[index % h3Cells.length];
    const zoneMemberships = `${pickupZoneId}|${dropoffZoneId}|${h3Cell}`;
    const edgeWeight = 1 + (index % 5) * 0.25;
    const sparseLift = index % 3 === 0 ? 14 : 0;
    const target = Math.round(trend + tripDistance * 9 + pickupHour * 1.7 + routePressure * 4.5 + spatialLift + weekdayLift + sparseLift + (dropoffZoneId - 204) * 3);
    rows.push(`${date},pickup_zone_1,${target},${tripDistance.toFixed(2)},${pickupHour},${routePressure},${pickupZoneId},${dropoffZoneId},${pickupLon.toFixed(5)},${pickupLat.toFixed(5)},${dropoffLon.toFixed(5)},${dropoffLat.toFixed(5)},${h3Cell},${zoneMemberships},${edgeWeight.toFixed(2)}`);
  }
  return rows.join('\n');
}

function parseDelimited(text: string, fileName: string): ParsedTable {
  const delimiter = text.includes('\t') ? '\t' : ',';
  const rows = parseRows(text.trim(), delimiter);
  if (rows.length < 2) {
    throw new Error('The uploaded file must include a header row and at least one data row.');
  }
  const columns = rows[0].map((column) => column.trim());
  return {
    columns,
    rows: rows.slice(1).filter((row) => row.some(Boolean)).map((row) => Object.fromEntries(columns.map((column, index) => [column, row[index] ?? '']))),
    fileName,
  };
}

async function parseParquet(file: File): Promise<ParsedTable> {
  const [{tableFromIPC}, parquet] = await Promise.all([
    import('apache-arrow'),
    import('parquet-wasm') as Promise<ParquetWasmModule>,
  ]);
  await parquet.default();
  const wasmTable = parquet.readParquet(new Uint8Array(await file.arrayBuffer()));
  try {
    const arrowTable = tableFromIPC(wasmTable.intoIPCStream()) as unknown as {
      numRows: number;
      schema: {fields: {name: string}[]};
      getChild: (name: string) => {get: (index: number) => unknown} | null;
      getChildAt: (index: number) => {get: (rowIndex: number) => unknown} | null;
    };
    const columns = arrowTable.schema.fields.map((field) => field.name);
    const vectors = columns.map((column, index) => arrowTable.getChild(column) ?? arrowTable.getChildAt(index));
    const rows = Array.from({length: arrowTable.numRows}, (_, rowIndex) =>
      Object.fromEntries(
        columns.map((column, columnIndex) => [
          column,
          formatCellValue(vectors[columnIndex]?.get(rowIndex)),
        ]),
      ),
    );
    if (columns.length === 0 || rows.length === 0) {
      throw new Error('The uploaded Parquet file must include at least one column and one row.');
    }
    return {columns, rows, fileName: file.name};
  } finally {
    wasmTable.free?.();
  }
}

function formatCellValue(value: unknown) {
  if (value == null) {
    return '';
  }
  if (value instanceof Date) {
    return value.toISOString().replace(/\.\d{3}Z$/, '');
  }
  if (typeof value === 'bigint') {
    return value.toString();
  }
  if (ArrayBuffer.isView(value)) {
    return Array.from(value as Uint8Array).join(',');
  }
  return String(value);
}

function parseRows(text: string, delimiter: string): string[][] {
  const rows: string[][] = [];
  let row: string[] = [];
  let cell = '';
  let quoted = false;

  for (let index = 0; index < text.length; index += 1) {
    const char = text[index];
    const next = text[index + 1];
    if (char === '"' && quoted && next === '"') {
      cell += '"';
      index += 1;
    } else if (char === '"') {
      quoted = !quoted;
    } else if (char === delimiter && !quoted) {
      row.push(cell);
      cell = '';
    } else if ((char === '\n' || char === '\r') && !quoted) {
      if (char === '\r' && next === '\n') {
        index += 1;
      }
      row.push(cell);
      rows.push(row);
      row = [];
      cell = '';
    } else {
      cell += char;
    }
  }
  row.push(cell);
  rows.push(row);
  return rows;
}
