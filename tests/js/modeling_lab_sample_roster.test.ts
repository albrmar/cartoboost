import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import test from 'node:test';

import {tableFromIPC} from 'apache-arrow';
import initParquet, {readParquet} from 'parquet-wasm/esm';

import initCartoboost, {
  availableForecastModels,
  runForecast,
} from '../../static/wasm/cartoboost/cartoboost_wasm.js';

type ArrowRow = Record<string, unknown>;

type ForecastRecord = {
  series_id: string;
  timestamp: string;
  horizon: number;
  prediction: number;
};

function scalar(value: unknown): unknown {
  return typeof value === 'bigint' ? Number(value) : value;
}

function numeric(value: unknown): number | null {
  const parsed = Number(scalar(value));
  return Number.isFinite(parsed) ? parsed : null;
}

const shippedSamples = [
  'static/samples/yellow_taxi_2024-01-single-lane-5000.parquet',
  'static/samples/yellow_taxi_2024-01-varied-routes-2500.parquet',
] as const;

test(
  'shipped taxi samples return finite records for every registered forecast model without fallback',
  {timeout: 360_000},
  async () => {
    await initParquet({
      module_or_path: await readFile('node_modules/parquet-wasm/esm/parquet_wasm_bg.wasm'),
    });
    await initCartoboost({
      module_or_path: await readFile('static/wasm/cartoboost/cartoboost_wasm_bg.wasm'),
    });

    const options = {
      seasonLength: 24,
      nEstimators: 4,
      maxDepth: 2,
      minSamplesLeaf: 16,
      lags: [1, 24],
      rollingMeanWindows: [24],
      calendarFeatures: ['day_of_week'],
      maxDirectHorizon: 14,
      coordinateX: 'pickup_longitude',
      coordinateY: 'pickup_latitude',
      windowSize: 24,
      windowCount: 3,
      validationWindow: 24,
      maxP: 1,
      maxD: 1,
      maxQ: 1,
      mstlSeasonLengths: [24],
    };

    for (const samplePath of shippedSamples) {
      const parquet = await readFile(samplePath);
      const arrowTable = tableFromIPC(readParquet(new Uint8Array(parquet)).intoIPCStream());
      const rows = Array.from({length: arrowTable.numRows}, (_, rowIndex) => {
        const source = arrowTable.get(rowIndex)?.toJSON() as ArrowRow | undefined;
        assert.ok(source, `${samplePath} row ${rowIndex} should exist`);
        const covariates: Record<string, number> = {};
        for (const [key, value] of Object.entries(source)) {
          if (['timestamp', 'target', 'series_id'].includes(key)) {
            continue;
          }
          const parsed = numeric(value);
          if (parsed !== null) {
            covariates[key] = parsed;
          }
        }
        return {
          timestamp: String(source.timestamp),
          target: Number(scalar(source.target)),
          seriesId: String(source.series_id),
          covariates,
        };
      });

      const expectedRecords = 14 * new Set(rows.map((row) => row.seriesId)).size;
      for (const model of availableForecastModels() as Array<{name: string}>) {
        const started = performance.now();
        const autoOptions = model.name === 'auto_forecast' ? {maxAutoCandidateCount: 4} : {};
        const response = runForecast({
          rows,
          frequency: 'hourly',
          horizon: 14,
          model: model.name,
          options: {...options, ...autoOptions},
          metadata: {
            timestampCol: 'timestamp',
            targetCol: 'target',
            seriesIdCol: 'series_id',
          },
        }) as {metadata: {warning?: unknown}; forecast: {records: ForecastRecord[]}};
        const elapsedMs = Math.round(performance.now() - started);
        console.log(`${samplePath} ${model.name} ${elapsedMs}ms ${response.forecast.records.length} records`);

        assert.equal(
          response.metadata.warning,
          undefined,
          `${model.name} should fit ${samplePath} directly without browser fallback`,
        );
        assert.equal(response.forecast.records.length, expectedRecords, `${model.name} ${samplePath} record count`);
        for (const [index, record] of response.forecast.records.entries()) {
          assert.ok(record.series_id, `${model.name} ${samplePath} record ${index} series_id`);
          assert.ok(record.timestamp, `${model.name} ${samplePath} record ${index} timestamp`);
          assert.ok(record.horizon > 0, `${model.name} ${samplePath} record ${index} horizon`);
          assert.ok(
            Number.isFinite(record.prediction),
            `${model.name} ${samplePath} record ${index} finite prediction`,
          );
        }
      }
    }
  },
);
