/**
 * 概览五 KPI 的指标名。query 与单位仍在 config.ts，专业盘只按这个顺序取传统卡。
 * 内存使用率用已有 redis_memory_utilization（used/maxmemory 派生），不另写查询。
 */
export const REDIS_OVERVIEW_METRIC_NAMES = [
  'redis_uptime',
  'redis_memory_utilization',
  'redis_keyspace_hitrate',
  'redis_evicted_keys_rate',
  'redis_clients'
] as const;

export const REDIS_OVERVIEW_KPI_TITLES = [
  '运行时长',
  '内存使用率',
  '缓存命中率',
  '键驱逐频率',
  '客户端连接数'
] as const;
