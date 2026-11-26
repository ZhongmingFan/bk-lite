import { describe, expect, it } from 'vitest';

import {
  FLAT_ZERO_STROKE_LIFT_PX,
  FLAT_ZERO_STROKE_SERIES_PREFIX,
  flatZeroStrokeLift,
  liftStrokeY,
  tooltipItemsWithoutFlatZeroStroke,
  tooltipSampleValue
} from '../flat-zero-stroke';

describe('flat zero stroke', () => {
  it('lifts an all-zero window 8px above the x axis', () => {
    expect(flatZeroStrokeLift([0, 0, 0], FLAT_ZERO_STROKE_LIFT_PX, 0)).toBe(8);
    const axisY = 240;
    expect(liftStrokeY(axisY, FLAT_ZERO_STROKE_LIFT_PX)).toBe(232);
  });

  it('reads 0 from the original series and drops the lifted stroke', () => {
    const rows = tooltipItemsWithoutFlatZeroStroke([
      { seriesName: 'connections', value: [1_700_000_000, 0] },
      { seriesName: `${FLAT_ZERO_STROKE_SERIES_PREFIX}0`, value: [1_700_000_000, 8] }
    ]);
    expect(rows.map((item) => item.seriesName)).toEqual(['connections']);
    expect(rows.map((item) => tooltipSampleValue(item.value))).toEqual([0]);
  });

  it('does not lift a series that is not entirely zero', () => {
    expect(flatZeroStrokeLift([0, 1, 0], FLAT_ZERO_STROKE_LIFT_PX, 0)).toBeNull();
    expect(flatZeroStrokeLift([0, 0, 0], undefined, 0)).toBeNull();
    expect(flatZeroStrokeLift([], FLAT_ZERO_STROKE_LIFT_PX, 0)).toBeNull();
    expect(flatZeroStrokeLift([0, 0, 0], FLAT_ZERO_STROKE_LIFT_PX, 1)).toBeNull();
    const rows = tooltipItemsWithoutFlatZeroStroke([
      { seriesName: 'cpu', value: [10, 4] }
    ]);
    expect(tooltipSampleValue(rows[0]?.value)).toBe(4);
  });

  it('keeps a zero series inside a mixed stack and lifts when the whole stack is zero', () => {
    expect(flatZeroStrokeLift([0, 0], FLAT_ZERO_STROKE_LIFT_PX, 0, true, false)).toBeNull();
    expect(flatZeroStrokeLift([0, 0], FLAT_ZERO_STROKE_LIFT_PX, 0, true, true)).toBe(8);
  });
});
