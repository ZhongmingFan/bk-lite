type CopyValues = Record<string, string | number>;
type Translate = (id: string, defaultMessage?: string, values?: CopyValues) => string;

const textId = (suffix: string) => `monitor.dashboards.text.${suffix}`;

/** Redis 专业盘文案。中文是缺词条时的回退。 */
export const REDIS_CATEGORY_COPY = {
  tabOverview: [textId('redisCategoryTabOverview'), '概览'],
  tabFocus: [textId('redisCategoryTabFocus'), '重点关注'],
  tabPerformance: [textId('redisCategoryTabPerformance'), '缓存效率'],
  tabMemory: [textId('redisCategoryTabMemory'), '内存'],
  tabClients: [textId('redisCategoryTabClients'), '客户端'],
  tabThroughput: [textId('redisCategoryTabThroughput'), '吞吐与网络'],
  tabReplication: [textId('redisCategoryTabReplication'), '复制 · 哨兵'],
  tabCluster: [textId('redisCategoryTabCluster'), 'Cluster'],
  tabSlowlog: [textId('redisCategoryTabSlowlog'), '慢命令'],
  healthOverview: [textId('redisCategoryHealthOverview'), '健康概览'],
  overviewTrends: [textId('redisCategoryOverviewTrends'), '趋势'],
  focusMiss: [textId('redisCategoryFocusMiss'), '缓存未命中频率'],
  focusFrag: [textId('redisCategoryFocusFrag'), '内存碎片率'],
  focusBlocked: [textId('redisCategoryFocusBlocked'), '阻塞等待客户端'],
  focusOps: [textId('redisCategoryFocusOps'), '实时操作频率'],
  viewHitTrend: [textId('redisCategoryViewHitTrend'), '查看缓存效率'],
  viewMemory: [textId('redisCategoryViewMemory'), '查看内存'],
  viewClients: [textId('redisCategoryViewClients'), '查看客户端'],
  viewThroughput: [textId('redisCategoryViewThroughput'), '查看吞吐'],
  cardUsedVsLimit: [textId('redisCategoryCardUsedVsLimit'), '已用内存 vs 内存上限'],
  cardHitVsMiss: [textId('redisCategoryCardHitVsMiss'), '命中 vs 未命中'],
  cardActiveVsBlocked: [textId('redisCategoryCardActiveVsBlocked'), '活跃连接 vs 阻塞等待'],
  cardNetCompare: [textId('redisCategoryCardNetCompare'), '入站 vs 出站'],
  cardHitRate: [textId('redisCategoryCardHitRate'), '缓存命中率'],
  fragHigh: [textId('redisCategoryFragHigh'), '碎片偏高'],
  fragOk: [textId('redisCategoryFragOk'), '1.5 以内'],
  memoryLimitUnset: [textId('redisCategoryMemoryLimitUnset'), '未配置内存上限'],
  chartQuiet: [textId('redisCategoryChartQuiet'), '当前无采集'],
  chartCommand: [textId('redisCategoryChartCommand'), '命令吞吐趋势'],
  chartHit: [textId('redisCategoryChartHit'), '缓存命中趋势'],
  chartBlocked: [textId('redisCategoryChartBlocked'), '阻塞等待趋势'],
  chartMemory: [textId('redisCategoryChartMemory'), '内存压力'],
  chartFrag: [textId('redisCategoryChartFrag'), '内存碎片率'],
  chartLifecycle: [textId('redisCategoryChartLifecycle'), '键生命周期'],
  chartClients: [textId('redisCategoryChartClients'), '客户端连接趋势'],
  chartRejected: [textId('redisCategoryChartRejected'), '连接拒绝频率'],
  chartNet: [textId('redisCategoryChartNet'), '网络流量'],
  seriesCommands: [textId('redisCategorySeriesCommands'), '命令速率'],
  seriesOps: [textId('redisCategorySeriesOps'), '实时操作频率'],
  seriesHit: [textId('redisCategorySeriesHit'), '命中频率'],
  seriesMiss: [textId('redisCategorySeriesMiss'), '未命中频率'],
  seriesBlocked: [textId('redisCategorySeriesBlocked'), '阻塞客户端'],
  seriesFrag: [textId('redisCategorySeriesFrag'), '碎片率'],
  seriesExpired: [textId('redisCategorySeriesExpired'), '键过期频率'],
  seriesEvicted: [textId('redisCategorySeriesEvicted'), '键驱逐频率'],
  seriesUsed: [textId('redisCategorySeriesUsed'), '已用内存'],
  seriesLimit: [textId('redisCategorySeriesLimit'), '内存上限'],
  seriesClients: [textId('redisCategorySeriesClients'), '活跃连接'],
  seriesRejected: [textId('redisCategorySeriesRejected'), '连接拒绝'],
  seriesNetIn: [textId('redisCategorySeriesNetIn'), '网络流入'],
  seriesNetOut: [textId('redisCategorySeriesNetOut'), '网络流出'],
  focusStepCards: [textId('redisCategoryFocusStepCards'), '看本区四卡'],
  focusStepCardsHint: [textId('redisCategoryFocusStepCardsHint'), '未命中、碎片、阻塞和吞吐是否一起异常。'],
  focusStepPerf: [textId('redisCategoryFocusStepPerf'), '看缓存效率'],
  focusStepPerfHint: [textId('redisCategoryFocusStepPerfHint'), '命中和未命中是否失衡。'],
  focusStepMemory: [textId('redisCategoryFocusStepMemory'), '看内存'],
  focusStepMemoryHint: [textId('redisCategoryFocusStepMemoryHint'), '使用量是否顶到上限，碎片是否超过 1.5。'],
  focusStepClients: [textId('redisCategoryFocusStepClients'), '看客户端与吞吐'],
  focusStepClientsHint: [textId('redisCategoryFocusStepClientsHint'), '阻塞、拒绝、流量和操作频率是否在堆积。'],
  alertTitle: [textId('redisCategoryAlertTitle'), 'Redis 告警'],
  alertFooter: [textId('redisCategoryAlertFooter'), '仅显示本实例活跃告警'],
  alertEmpty: [textId('redisCategoryAlertEmpty'), '当前无活跃告警'],
  alertEmptyHint: [textId('redisCategoryAlertEmptyHint'), '本实例运行正常']
} as const;

export type RedisCategoryCopyKey = keyof typeof REDIS_CATEGORY_COPY;

export const redisCategoryText = (
  t: Translate,
  key: RedisCategoryCopyKey,
  values?: CopyValues
) => {
  const [id, fallback] = REDIS_CATEGORY_COPY[key];
  return t(id, fallback, values);
};
