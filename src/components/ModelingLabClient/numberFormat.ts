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
