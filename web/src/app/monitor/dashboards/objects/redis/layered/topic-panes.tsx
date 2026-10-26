'use client';

import type { ReactNode } from 'react';
import { Progress } from 'antd';
import { ProDashLoading, ProSection } from '../../host/layered/pro-chrome';
import chart from '../../host/layered/layered.module.scss';
import { RingChartPanel } from '../../../shared/widgets';
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

/**
 * 环图壳仍写 panelTitle 槽。CSS Module 导出不是可枚举普通对象，展开会丢掉
 * panel / ring 类，所以只按键改挂 chartTitle，其余壳层原样走。
 */
const ringStyles = new Proxy(styles, {
  get(target, key) {
    if (key === 'panelHeading') return chart.chartHead;
    if (key === 'panelTitle') return `${chart.chartTitle} m-0`;
    return Reflect.get(target, key);
  }
});

const featureGrid = 'grid min-w-0 grid-cols-1 items-stretch gap-3 lg:grid-cols-2';

function useRedisText() {
  const { t } = useDashboardText();
  return (key: RedisCategoryCopyKey) => redisCategoryText(t, key);
}

/** 与 ThinTimeSeries 卡头同构：chartHead + chartTitle（13px/600）。加载和空态也留着。 */
function FeatureHead({ title }: { title: string }) {
  return (
    <div className={chart.chartHead}>
      <div className={chart.chartTitleGroup}>
        <div className={chart.chartTitleBlock}>
          <div className={chart.chartTitle}>{title}</div>
        </div>
      </div>
    </div>
  );
}

function TopicSections({
  pane,
  readings,
  trend
}: {
  pane: string;
  readings: ReactNode;
  trend: ReactNode;
}) {
  const text = useRedisText();
  return (
    <div data-redis-pane={pane} className="contents">
      <ProSection title={text('sectionReadings')}>
        {readings}
      </ProSection>
      <ProSection title={text('sectionTrends')}>
        {trend}
      </ProSection>
    </div>
  );
}

function BigNumber({
  title,
  metric,
  unit,
  color,
  footer,
  metricMap,
  pending
}: {
  title: string;
  metric: string;
  unit: MetricUnit;
  color: string;
  footer?: string;
  metricMap: RedisMetricMap;
  pending: ReadonlySet<string>;
}) {
  const shown = redisDisplay(metricMap, metric, unit);
  const loading = pending.has(metric) && shown.value == null;
  const valueColor = shown.value == null ? 'var(--color-text-3)' : color;
  return (
    <section className={`${styles.panel} h-full`} data-redis-feature="bignum">
      <FeatureHead title={title} />
      {loading ? (
        <ProDashLoading />
      ) : (
        <div className="flex flex-col gap-2">
          <div className={styles.statValue} style={{ color: valueColor }}>
            {shown.text}
            {shown.unit ? <span className={styles.statUnit}>{shown.unit}</span> : null}
          </div>
          {footer ? <div className={styles.statMeta}>{footer}</div> : null}
        </div>
      )}
    </section>
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
    <section className={`${styles.panel} h-full`} data-redis-feature="compare">
      <FeatureHead title={title} />
      {loading ? (
        <ProDashLoading />
      ) : quiet ? (
        <p className="m-0 text-[13px] text-[var(--color-text-3)]">{quietText}</p>
      ) : (
        <div className="flex flex-col gap-3">
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
    <TopicSections
      pane="memory"
      readings={(
        <div className={featureGrid}>
          {unset ? (
            <section className={`${styles.panel} h-full`} data-redis-feature="memory-limit">
              <FeatureHead title={text('cardUsedVsLimit')} />
              <p className="m-0 text-[13px] text-[var(--color-text-3)]">{text('memoryLimitUnset')}</p>
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
            footer={frag == null ? text('chartQuiet') : frag > 1.5 ? text('fragHigh') : text('fragOk')}
            metricMap={metricMap}
            pending={pending}
          />
        </div>
      )}
      trend={<RedisTrendChart spec={lifecycleTrend} metricMap={metricMap} pending={pending} />}
    />
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
    <TopicSections
      pane="cache"
      readings={(
        <div className={featureGrid}>
          {loading ? (
            <section className={`${styles.panel} h-full`} data-redis-feature="compare">
              <FeatureHead title={text('cardHitVsMiss')} />
              <ProDashLoading />
            </section>
          ) : (
            <RingChartPanel
              className="h-full"
              title={text('cardHitVsMiss')}
              data={[
                { name: text('seriesHit'), value: Math.max(hits.value ?? 0, 0), color: REDIS_PALETTE.emerald, display: hits.text },
                { name: text('seriesMiss'), value: Math.max(misses.value ?? 0, 0), color: REDIS_PALETTE.rose, display: misses.text }
              ]}
              centerValue={hitRate.unit ? `${hitRate.text}${hitRate.unit}` : hitRate.text}
              centerCaption={text('cardHitRate')}
              isEmpty={empty}
              emptyDescription={text('chartQuiet')}
              styles={ringStyles}
            />
          )}
          <BigNumber
            title={text('cardHitRate')}
            metric="redis_keyspace_hitrate"
            unit="percent"
            color={REDIS_PALETTE.emerald}
            metricMap={metricMap}
            pending={pending}
          />
        </div>
      )}
      trend={<RedisTrendChart spec={hitTrend} metricMap={metricMap} pending={pending} />}
    />
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
    <TopicSections
      pane="clients"
      readings={(
        <div className={featureGrid}>
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
          <BigNumber
            title={text('cardRejected')}
            metric="redis_rejected_connections_rate"
            unit="cps"
            color={REDIS_PALETTE.rose}
            metricMap={metricMap}
            pending={pending}
          />
        </div>
      )}
      trend={<RedisTrendChart spec={rejectTrend} metricMap={metricMap} pending={pending} />}
    />
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
    <TopicSections
      pane="throughput"
      readings={(
        <div className={featureGrid}>
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
            metricMap={metricMap}
            pending={pending}
          />
        </div>
      )}
      trend={<RedisTrendChart spec={opsTrend} metricMap={metricMap} pending={pending} />}
    />
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
