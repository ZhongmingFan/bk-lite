'use client';

import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Tabs } from 'antd';
import useMonitorApi from '@/app/monitor/api';
import type { TimeValuesProps } from '@/app/monitor/types';
import HostAlertDrawer from '../../host/layered/host-alert-drawer';
import { hostAlertRailEvents, type HostAlertRailEvent } from '../../host/layered/events-rail';
import { overviewHostAlertParams } from '../../host/layered/host-alert-list';
import { ProTimeWindowProvider } from '../../host/layered/pro-time-window';
import { selectProChartWindow } from '../../../shared/utils/pro-chart-window';
import type { PreparedSummaryCard } from '../../common/simple-dashboard-core';
import { useDashboardText } from '../../../shared/utils/content-i18n';
import layered from '../../host/layered/layered.module.scss';
import OverviewPane from './overview-pane';
import type { RedisTopicTab } from './focus-pane';
import { CachePane, ClientsPane, MemoryPane, ThroughputPane } from './topic-panes';
import { redisCategoryText, type RedisCategoryCopyKey } from './redis-copy';
import type { RedisMetricMap } from './series';

type RedisContentTab = 'overview' | RedisTopicTab;

const TABS: ReadonlyArray<{ key: RedisContentTab; copy: RedisCategoryCopyKey }> = [
  { key: 'overview', copy: 'tabOverview' },
  { key: 'memory', copy: 'tabMemory' },
  { key: 'cache', copy: 'tabPerformance' },
  { key: 'clients', copy: 'tabClients' },
  { key: 'throughput', copy: 'tabThroughput' }
];

interface RedisCategoryLayoutProps {
  instanceId: React.Key;
  monitorObjectId: React.Key;
  timeValues: TimeValuesProps;
  queryTimeRange: { startMs: number; endMs: number } | null;
  loading: boolean;
  pendingMetricNames: ReadonlySet<string>;
  summaryCards: PreparedSummaryCard[];
  metricMap: RedisMetricMap;
}

export default function RedisCategoryLayout({
  instanceId,
  monitorObjectId,
  timeValues,
  queryTimeRange,
  loading,
  pendingMetricNames,
  summaryCards,
  metricMap
}: RedisCategoryLayoutProps) {
  const { t } = useDashboardText();
  const { getMonitorAlert } = useMonitorApi();
  const text = (key: RedisCategoryCopyKey) => redisCategoryText(t, key);
  const [tab, setTab] = useState<RedisContentTab>('overview');
  const [alertCount, setAlertCount] = useState<number | null>(null);
  const [alertEvents, setAlertEvents] = useState<HostAlertRailEvent[]>([]);
  const [eventsPhase, setEventsPhase] = useState<'loading' | 'ready'>('loading');
  const eventsScopeRef = useRef<string | null>(null);
  const [alertDrawerOpen, setAlertDrawerOpen] = useState(false);
  const alertDrawerOpenRef = useRef(false);
  alertDrawerOpenRef.current = alertDrawerOpen;
  const [alertRefreshKey, setAlertRefreshKey] = useState(0);
  const [focusAlertId, setFocusAlertId] = useState('');
  const instanceKey = String(instanceId || '');
  const [seenInstance, setSeenInstance] = useState(instanceKey);
  if (instanceKey !== seenInstance) {
    setSeenInstance(instanceKey);
    setAlertCount(null);
    setAlertEvents([]);
    setFocusAlertId('');
  }
  const chartWindow = useMemo(
    () => selectProChartWindow(timeValues, queryTimeRange),
    [queryTimeRange, timeValues]
  );
  const chartWindowKey = chartWindow ? `${chartWindow.startSec}|${chartWindow.endSec}` : '';
  const eventsScope = `${instanceKey}|${
    chartWindow ? Math.round(chartWindow.endSec - chartWindow.startSec) : ''
  }`;
  const kpiLoading = (loading && pendingMetricNames.size === 0)
    || summaryCards.some((card) => pendingMetricNames.has(card.card.metric));

  useEffect(() => {
    if (eventsScopeRef.current === eventsScope) return;
    const first = eventsScopeRef.current == null;
    eventsScopeRef.current = eventsScope;
    if (first) return;
    setAlertEvents([]);
    setEventsPhase('loading');
  }, [eventsScope]);

  useEffect(() => {
    if (tab !== 'overview') return undefined;
    if (!instanceKey) {
      setAlertCount(null);
      setAlertEvents([]);
      setEventsPhase('ready');
      return undefined;
    }
    let active = true;
    getMonitorAlert(overviewHostAlertParams(instanceKey))
      .then((data) => {
        if (!active) return;
        setAlertEvents(hostAlertRailEvents(data?.results));
        setEventsPhase('ready');
        if (!alertDrawerOpenRef.current) {
          setAlertCount(typeof data?.count === 'number' ? data.count : 0);
        }
      })
      .catch(() => {
        if (!active) return;
        setAlertEvents([]);
        setEventsPhase('ready');
        if (!alertDrawerOpenRef.current) setAlertCount(null);
      });
    return () => {
      active = false;
    };
  }, [chartWindowKey, getMonitorAlert, instanceKey, tab]);

  return (
    <ProTimeWindowProvider
      timeValues={timeValues}
      queryTimeRange={queryTimeRange}
      suspend={loading && pendingMetricNames.size === 0}
    >
      <div
        className={layered.root}
        data-redis-pro="category"
        data-redis-alert-count={alertCount ?? undefined}
      >
        <div className={layered.tabBand}>
          <Tabs
            className={layered.partitionTabs}
            activeKey={tab}
            onChange={(key) => setTab(key as RedisContentTab)}
            items={TABS.map((item) => ({
              key: item.key,
              label: text(item.copy)
            }))}
          />
        </div>
        <div className={layered.main}>
          {tab === 'overview' ? (
            <OverviewPane
              summaryCards={summaryCards}
              loading={kpiLoading}
              metricMap={metricMap}
              pendingMetricNames={pendingMetricNames}
              alertEvents={alertEvents}
              eventsLoading={eventsPhase === 'loading'}
              onFocusAlert={(alertId) => {
                setFocusAlertId(alertId);
                setAlertDrawerOpen(true);
                setAlertRefreshKey((key) => key + 1);
              }}
              onOpen={setTab}
            />
          ) : null}
          {tab === 'memory' ? (
            <MemoryPane metricMap={metricMap} pending={pendingMetricNames} />
          ) : null}
          {tab === 'cache' ? (
            <CachePane metricMap={metricMap} pending={pendingMetricNames} />
          ) : null}
          {tab === 'clients' ? (
            <ClientsPane metricMap={metricMap} pending={pendingMetricNames} />
          ) : null}
          {tab === 'throughput' ? (
            <ThroughputPane metricMap={metricMap} pending={pendingMetricNames} />
          ) : null}
        </div>
        <HostAlertDrawer
          open={alertDrawerOpen}
          refreshKey={alertRefreshKey}
          instanceId={instanceKey}
          monitorObjectId={monitorObjectId}
          focusAlertId={focusAlertId}
          onClose={() => setAlertDrawerOpen(false)}
          onCount={setAlertCount}
          title={text('alertTitle')}
          footerHint={text('alertFooter')}
          emptyCopy={text('alertEmpty')}
          emptyHint={text('alertEmptyHint')}
        />
      </div>
    </ProTimeWindowProvider>
  );
}
