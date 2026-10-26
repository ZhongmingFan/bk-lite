'use client';

import React from 'react';
import { Progress } from 'antd';
import { ProDashLoading } from '../../host/layered/pro-chrome';
import { RingChartPanel, StatCard } from '../../../shared/widgets';
import { HealthIcon, MemoryIcon, PublishIcon } from '../../../shared/widgets/metric-icons';
import { useDashboardText } from '../../../shared/utils/content-i18n';
import type { MetricUnit } from '../../../shared/types';
import { REDIS_PALETTE } from '../config';
import styles from '../index.module.scss';
import { redisCategoryText, type RedisCategoryCopyKey } from './redis-copy';
import {
  redisDisplay,
  redisLatest,
  redisMemoryLimitUnset,
  type RedisMetricMap
} from './series';
import RedisTrendChart, { type RedisTrendSpec } from './trend-chart';

const commandTrend: RedisTrendSpec = {
  titleKey: 'chartCommand',
  unit: 'cps',
  series: [
    { name: 'redis_total_commands_processed_rate', labelKey: 'seriesCommands', color: REDIS_PALETTE.cyan }
  ]
};

const hitTrend: RedisTrendSpec = {
  titleKey: 'chartHit',
  unit: 'cps',
  series: [
    { name: 'redis_keyspace_hits_rate', labelKey: 'seriesHit', color: REDIS_PALETTE.emerald },
    { name: 'redis_keyspace_misses_rate', labelKey: 'seriesMiss', color: REDIS_PALETTE.rose }
  ]
};

const lifecycleTrend: RedisTrendSpec = {
  titleKey: 'chartLifecycle',
  unit: 'cps',
  series: [
    { name: 'redis_expired_keys_rate', labelKey: 'seriesExpired', color: REDIS_PALETTE.indigo },
    { name: 'redis_evicted_keys_rate', labelKey: 'seriesEvicted', color: REDIS_PALETTE.rose }
  ]
};

const rejectTrend: RedisTrendSpec = {
  titleKey: 'chartRejected',
  unit: 'cps',
  series: [
    { name: 'redis_rejected_connections_rate', labelKey: 'seriesRejected', color: REDIS_PALETTE.rose }
  ]
};

const opsTrend: RedisTrendSpec = {
  titleKey: 'chartCommand',
  unit: 'cps',
  series: [
    { name: 'redis_total_commands_processed_rate', labelKey: 'seriesCommands', color: REDIS_PALETTE.cyan },
    { name: 'redis_instantaneous_ops_per_sec', labelKey: 'seriesOps', color: REDIS_PALETTE.blue }
  ]
};

function useRedisText() {
  const { t } = useDashboardText();
  return (key: RedisCategoryCopyKey) => redisCategoryText(t, key);
}

function BigNumber({
  title,
  metric,
  unit,
  color,
  icon,
  footer,
  metricMap,
  pending
}: {
  title: string;
  metric: string;
  unit: MetricUnit;
  color: string;
  icon: React.ReactNode;
  footer?: string;
  metricMap: RedisMetricMap;
  pending: ReadonlySet<string>;
}) {
  const shown = redisDisplay(metricMap, metric, unit);
  const loading = pending.has(metric) && shown.value == null;
  return (
    <StatCard
      title={title}
      value={loading ? ' ' : shown.text}
      unit={loading ? '' : shown.unit}
      icon={icon}
      iconStyle={{ color }}
      color={shown.value == null ? 'var(--color-text-3)' : color}
      hideTrend
      footer={footer}
      extra={loading ? <ProDashLoading size="small" layout="inline" /> : undefined}
      styles={styles}
    />
  );
}

function PairBars({
  title,
  quiet,
  loading,
  quietText,
  rows
}: {
  title: string;
  quiet: boolean;
  loading: boolean;
  quietText: string;
  rows: ReadonlyArray<{ label: string; display: string; percent: number; color: string }>;
}) {
  return (
    <section className={styles.panel} data-redis-feature="compare">
      <div className={styles.panelHeading}>
        <h3 className={styles.panelTitle}>{title}</h3>
      </div>
      {loading ? (
        <ProDashLoading />
      ) : quiet ? (
        <p className="m-0 text-[13px] text-[var(--color-text-3)]">{quietText}</p>
      ) : (
        <div className="mt-3 flex flex-col gap-3">
          {rows.map((row) => (
            <div key={row.label}>
              <div className="mb-1 flex justify-between gap-3 text-[12px]">
                <span className="text-[var(--color-text-1)]">{row.label}</span>
                <span className="tabular-nums text-[var(--color-text-3)]">{row.display}</span>
              </div>
              <Progress
                percent={row.percent}
                showInfo={false}
                strokeColor={row.color}
                trailColor="var(--color-fill-2)"
              />
            </div>
          ))}
        </div>
      )}
    </section>
  );
}

function reading(metricMap: RedisMetricMap, name: string, unit: MetricUnit) {
  const shown = redisDisplay(metricMap, name, unit);
  return shown.unit ? `${shown.text}${shown.unit}` : shown.text;
}

export function MemoryPane({
  metricMap,
  pending
}: {
  metricMap: RedisMetricMap;
  pending: ReadonlySet<string>;
}) {
  const text = useRedisText();
  const unset = redisMemoryLimitUnset(metricMap);
  const used = redisDisplay(metricMap, 'redis_used_memory', 'bytes');
  const limit = redisDisplay(metricMap, 'redis_maxmemory', 'bytes');
  const frag = redisLatest(metricMap, 'redis_mem_fragmentation_ratio');
  const loading = pending.has('redis_used_memory') && used.value == null && !unset;
  const percent = !unset && limit.value != null && limit.value > 0 && used.value != null
    ? Math.min(100, Math.max(0, (used.value / limit.value) * 100))
    : 0;
  const fragColor = frag == null ? 'var(--color-text-3)' : frag > 1.5 ? REDIS_PALETTE.rose : REDIS_PALETTE.emerald;
  return (
    <div data-redis-pane="memory" className="flex flex-col gap-4">
      <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
        {unset ? (
          <section className={styles.panel} data-redis-feature="memory-limit">
            <div className={styles.panelHeading}>
              <h3 className={styles.panelTitle}>{text('cardUsedVsLimit')}</h3>
            </div>
            <p className="m-0 mt-3 text-[13px] text-[var(--color-text-3)]">{text('memoryLimitUnset')}</p>
            <p className="m-0 mt-1 text-[13px] tabular-nums text-[var(--color-text-1)]">
              {text('seriesUsed')} {reading(metricMap, 'redis_used_memory', 'bytes')}
            </p>
          </section>
        ) : (
          <PairBars
            title={text('cardUsedVsLimit')}
            quiet={!loading && used.value == null}
            loading={loading}
            quietText={text('chartQuiet')}
            rows={[{
              label: text('seriesUsed'),
              display: `${reading(metricMap, 'redis_used_memory', 'bytes')} / ${reading(metricMap, 'redis_maxmemory', 'bytes')}`,
              percent,
              color: REDIS_PALETTE.blue
            }]}
          />
        )}
        <BigNumber
          title={text('focusFrag')}
          metric="redis_mem_fragmentation_ratio"
          unit="none"
          color={fragColor}
          icon={<MemoryIcon />}
          footer={frag == null ? text('chartQuiet') : frag > 1.5 ? text('fragHigh') : text('fragOk')}
          metricMap={metricMap}
          pending={pending}
        />
      </div>
      <RedisTrendChart spec={lifecycleTrend} metricMap={metricMap} pending={pending} />
    </div>
  );
}

export function CachePane({
  metricMap,
  pending
}: {
  metricMap: RedisMetricMap;
  pending: ReadonlySet<string>;
}) {
  const text = useRedisText();
  const hits = redisDisplay(metricMap, 'redis_keyspace_hits_rate', 'cps');
  const misses = redisDisplay(metricMap, 'redis_keyspace_misses_rate', 'cps');
  const hitRate = redisDisplay(metricMap, 'redis_keyspace_hitrate', 'percent');
  const loading = (pending.has('redis_keyspace_hits_rate') || pending.has('redis_keyspace_misses_rate'))
    && hits.value == null && misses.value == null;
  const empty = !loading && hits.value == null && misses.value == null;
  return (
    <div data-redis-pane="cache" className="flex flex-col gap-4">
      <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
        {loading ? (
          <section className={styles.panel}><ProDashLoading /></section>
        ) : (
          <RingChartPanel
            title={text('cardHitVsMiss')}
            data={[
              { name: text('seriesHit'), value: Math.max(hits.value ?? 0, 0), color: REDIS_PALETTE.emerald, display: hits.text },
              { name: text('seriesMiss'), value: Math.max(misses.value ?? 0, 0), color: REDIS_PALETTE.rose, display: misses.text }
            ]}
            centerValue={hitRate.unit ? `${hitRate.text}${hitRate.unit}` : hitRate.text}
            centerCaption={text('cardHitRate')}
            isEmpty={empty}
            emptyDescription={text('chartQuiet')}
            styles={styles}
          />
        )}
        <BigNumber
          title={text('cardHitRate')}
          metric="redis_keyspace_hitrate"
          unit="percent"
          color={REDIS_PALETTE.emerald}
          icon={<HealthIcon />}
          metricMap={metricMap}
          pending={pending}
        />
      </div>
      <RedisTrendChart spec={hitTrend} metricMap={metricMap} pending={pending} />
    </div>
  );
}

export function ClientsPane({
  metricMap,
  pending
}: {
  metricMap: RedisMetricMap;
  pending: ReadonlySet<string>;
}) {
  const text = useRedisText();
  const active = redisLatest(metricMap, 'redis_clients');
  const blocked = redisLatest(metricMap, 'redis_blocked_clients');
  const loading = pending.has('redis_clients') && active == null && blocked == null;
  const max = Math.max(active ?? 0, blocked ?? 0);
  const share = (value: number | null) => (max > 0 && value != null ? (value / max) * 100 : 0);
  return (
    <div data-redis-pane="clients" className="flex flex-col gap-4">
      <PairBars
        title={text('cardActiveVsBlocked')}
        quiet={!loading && active == null && blocked == null}
        loading={loading}
        quietText={text('chartQuiet')}
        rows={[
          {
            label: text('seriesClients'),
            display: reading(metricMap, 'redis_clients', 'counts'),
            percent: share(active),
            color: REDIS_PALETTE.blue
          },
          {
            label: text('seriesBlocked'),
            display: reading(metricMap, 'redis_blocked_clients', 'counts'),
            percent: share(blocked),
            color: REDIS_PALETTE.orange
          }
        ]}
      />
      <RedisTrendChart spec={rejectTrend} metricMap={metricMap} pending={pending} />
    </div>
  );
}

export function ThroughputPane({
  metricMap,
  pending
}: {
  metricMap: RedisMetricMap;
  pending: ReadonlySet<string>;
}) {
  const text = useRedisText();
  const inbound = redisLatest(metricMap, 'redis_total_net_input_bytes_rate');
  const outbound = redisLatest(metricMap, 'redis_total_net_output_bytes_rate');
  const loading = pending.has('redis_total_net_input_bytes_rate') && inbound == null && outbound == null;
  const max = Math.max(inbound ?? 0, outbound ?? 0);
  const share = (value: number | null) => (max > 0 && value != null ? (value / max) * 100 : 0);
  return (
    <div data-redis-pane="throughput" className="flex flex-col gap-4">
      <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
        <PairBars
          title={text('cardNetCompare')}
          quiet={!loading && inbound == null && outbound == null}
          loading={loading}
          quietText={text('chartQuiet')}
          rows={[
            {
              label: text('seriesNetIn'),
              display: reading(metricMap, 'redis_total_net_input_bytes_rate', 'byteps'),
              percent: share(inbound),
              color: REDIS_PALETTE.blue
            },
            {
              label: text('seriesNetOut'),
              display: reading(metricMap, 'redis_total_net_output_bytes_rate', 'byteps'),
              percent: share(outbound),
              color: REDIS_PALETTE.emerald
            }
          ]}
        />
        <BigNumber
          title={text('focusOps')}
          metric="redis_instantaneous_ops_per_sec"
          unit="cps"
          color={REDIS_PALETTE.blue}
          icon={<PublishIcon />}
          metricMap={metricMap}
          pending={pending}
        />
      </div>
      <RedisTrendChart spec={opsTrend} metricMap={metricMap} pending={pending} />
    </div>
  );
}

export function CompactTrends({
  metricMap,
  pending
}: {
  metricMap: RedisMetricMap;
  pending: ReadonlySet<string>;
}) {
  return (
    <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
      <RedisTrendChart spec={commandTrend} metricMap={metricMap} pending={pending} />
      <RedisTrendChart spec={hitTrend} metricMap={metricMap} pending={pending} />
    </div>
  );
}
