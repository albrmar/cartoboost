import assert from 'node:assert/strict';
import test from 'node:test';

import {coerceFiniteNumber, formatFixed, formatPercent} from '../../src/components/ModelingLabClient/numberFormat';

test('coerceFiniteNumber accepts finite numbers and numeric strings', () => {
  assert.equal(coerceFiniteNumber(12.5), 12.5);
  assert.equal(coerceFiniteNumber('12.5'), 12.5);
  assert.equal(coerceFiniteNumber(' 0 '), 0);
});

test('coerceFiniteNumber rejects non-finite and non-numeric values', () => {
  assert.equal(coerceFiniteNumber(Number.NaN), null);
  assert.equal(coerceFiniteNumber(Number.POSITIVE_INFINITY), null);
  assert.equal(coerceFiniteNumber(''), null);
  assert.equal(coerceFiniteNumber('not a number'), null);
  assert.equal(coerceFiniteNumber(null), null);
});

test('formatters do not call toFixed on non-number payloads', () => {
  assert.equal(formatFixed('424.1114', 3), '424.111');
  assert.equal(formatFixed(undefined, 3), '-');
  assert.equal(formatFixed('bad-payload', 3), '-');
  assert.equal(formatPercent('0.1874', 1), '+18.7%');
  assert.equal(formatPercent('bad-payload', 1), '-');
});
