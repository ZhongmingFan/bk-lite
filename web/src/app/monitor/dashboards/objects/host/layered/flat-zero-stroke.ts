/** 全 0 折线彩色笔画中心离开 X 轴的距离。调用方必须显式传入这个数。 */
export const FLAT_ZERO_STROKE_LIFT_PX = 8;

/** 绘制层附加序列的名字。悬浮框会跳过它，避免把同一条 0 读两遍。 */
export const FLAT_ZERO_STROKE_SERIES_PREFIX = '\0flat-zero:';

export const isFlatZeroStrokeSeriesName = (name: unknown) => (
  String(name ?? '').startsWith(FLAT_ZERO_STROKE_SERIES_PREFIX)
);

/** 可视窗口里的采样整段都是 0。空序列不算。 */
export const seriesIsFlatZero = (samples: readonly number[]) => (
  samples.length > 0 && samples.every((value) => value === 0)
);

/** 图上每一条都有采样且全是 0。空图或夹着空序列时不算。 */
export const everySeriesFlatZero = (groups: readonly (readonly number[])[]) => (
  groups.length > 0 && groups.every((samples) => seriesIsFlatZero(samples))
);

/**
 * 整条序列在可视窗口内全是 0、且数值轴下限就是 0 时，返回只用于绘制的上移像素。
 * 堆叠图里只要还有不是全 0 的序列，这条全 0 线留在堆里，不单独抬。
 * 其它情况返回 null：不改采样、不改轴、不抬非零线。
 */
export const flatZeroStrokeLift = (
  samples: readonly number[],
  requested: number | undefined,
  axisMin: number,
  stacked = false,
  chartAllZero = false
): number | null => {
  if (requested !== FLAT_ZERO_STROKE_LIFT_PX) return null;
  if (axisMin !== 0) return null;
  if (!seriesIsFlatZero(samples)) return null;
  if (stacked && !chartAllZero) return null;
  return FLAT_ZERO_STROKE_LIFT_PX;
};

/** 像素 Y 向下增长，减去间距就是离开 X 轴向上。 */
export const liftStrokeY = (pixelY: number, liftPx: number) => pixelY - liftPx;

export interface FlatZeroTooltipItem {
  seriesName?: unknown;
  value?: unknown;
}

/** 丢掉抬升绘制层。剩下的 value 仍是采样，全 0 序列读出来是 0。 */
export const tooltipItemsWithoutFlatZeroStroke = <T extends FlatZeroTooltipItem>(items: readonly T[]): T[] => (
  items.filter((item) => !isFlatZeroStrokeSeriesName(item?.seriesName))
);

/** 悬浮读数。数组值取 Y，不取绘制层上移后的像素。 */
export const tooltipSampleValue = (value: unknown): number | null => {
  const raw = Array.isArray(value) ? value[1] : value;
  if (typeof raw === 'number') return Number.isFinite(raw) ? raw : null;
  if (typeof raw === 'string' && raw.trim() !== '') {
    const numeric = Number(raw);
    return Number.isFinite(numeric) ? numeric : null;
  }
  return null;
};
