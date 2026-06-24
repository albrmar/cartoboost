export function coerceFiniteNumber(value: unknown): number | null {
  if (typeof value === 'number') {
    return Number.isFinite(value) ? value : null;
  }
  if (typeof value === 'string') {
    const trimmed = value.trim();
    if (trimmed === '') {
      return null;
    }
    const parsed = Number(trimmed);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
}

export function formatFixed(value: unknown, digits = 3) {
  const numeric = coerceFiniteNumber(value);
  return numeric === null ? '-' : numeric.toFixed(digits);
}

export function formatPercent(value: unknown, digits = 1) {
  const numeric = coerceFiniteNumber(value);
  return numeric === null ? '-' : `${numeric >= 0 ? '+' : ''}${(numeric * 100).toFixed(digits)}%`;
}

export function assertForecastResponseRecords(response: unknown, requestedModel: string): void {
  const records = (response as {forecast?: {records?: unknown[]}} | null)?.forecast?.records;
  if (!Array.isArray(records) || records.length === 0) {
    throw new Error(`${requestedModel} returned no forecast records.`);
  }
  for (const [index, record] of records.entries()) {
    const row = record as {
      series_id?: unknown;
      timestamp?: unknown;
      horizon?: unknown;
      prediction?: unknown;
    };
    if (
      typeof row.series_id !== 'string' ||
      row.series_id.length === 0 ||
      typeof row.timestamp !== 'string' ||
      row.timestamp.length === 0 ||
      !Number.isInteger(row.horizon) ||
      Number(row.horizon) <= 0 ||
      coerceFiniteNumber(row.prediction) === null
    ) {
      throw new Error(`${requestedModel} returned an invalid forecast record at index ${index}.`);
    }
  }
}
