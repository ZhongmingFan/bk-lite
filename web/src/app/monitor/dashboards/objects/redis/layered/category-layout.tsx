'use client';

import React, { useState } from 'react';
import { Tabs } from 'antd';
import type { TimeValuesProps } from '@/app/monitor/types';
import HostAlertDrawer from '../../host/layered/host-alert-drawer';
import { ProTimeWindowProvider } from '../../host/layered/pro-time-window';
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
  const text = (key: RedisCategoryCopyKey) => redisCategoryText(t, key);
  const [tab, setTab] = useState<RedisContentTab>('overview');
  const kpiLoading = (loading && pendingMetricNames.size === 0)
    || summaryCards.some((card) => pendingMetricNames.has(card.card.metric));

  return (
    <ProTimeWindowProvider
      timeValues={timeValues}
      queryTimeRange={queryTimeRange}
      suspend={loading && pendingMetricNames.size === 0}
    >
      <div className={layered.root} data-redis-pro="category">
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
        {/* 事件轨下一刀才打开。这里只复用公共抽屉，并按实例过滤。 */}
        <HostAlertDrawer
          open={false}
          refreshKey={0}
          instanceId={String(instanceId || '')}
          monitorObjectId={monitorObjectId}
          onClose={() => undefined}
          onCount={() => undefined}
          title={text('alertTitle')}
          footerHint={text('alertFooter')}
          emptyCopy={text('alertEmpty')}
          emptyHint={text('alertEmptyHint')}
        />
      </div>
    </ProTimeWindowProvider>
  );
}
