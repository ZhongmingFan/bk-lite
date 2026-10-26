type CopyValues = Record<string, string | number>;
type Translate = (id: string, defaultMessage?: string, values?: CopyValues) => string;

const textId = (suffix: string) => `monitor.dashboards.text.${suffix}`;

/** Redis 专业盘文案。中文是缺词条时的回退。 */
export const REDIS_CATEGORY_COPY = {
  tabOverview: [textId('redisCategoryTabOverview'), '概览'],
  tabFocus: [textId('redisCategoryTabFocus'), '重点关注'],
  tabPerformance: [textId('redisCategoryTabPerformance'), '性能与缓存'],
  tabMemory: [textId('redisCategoryTabMemory'), '内存'],
  tabClients: [textId('redisCategoryTabClients'), '连接与网络'],
  tabReplication: [textId('redisCategoryTabReplication'), '复制 · 哨兵'],
  tabCluster: [textId('redisCategoryTabCluster'), 'Cluster'],
  tabSlowlog: [textId('redisCategoryTabSlowlog'), '慢命令'],
  healthOverview: [textId('redisCategoryHealthOverview'), '健康概览'],
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
