'use client';

import React, { useState } from 'react';
import { Tabs } from 'antd';
import type { TimeValuesProps } from '@/app/monitor/types';
import HostAlertDrawer from '../../host/layered/host-alert-drawer';
import { GrayTabEmptyState, ProSection } from '../../host/layered/pro-chrome';
import { ProTimeWindowProvider } from '../../host/layered/pro-time-window';
import type { PreparedSummaryCard } from '../../common/simple-dashboard-core';
import { useDashboardText } from '../../../shared/utils/content-i18n';
import layered from '../../host/layered/layered.module.scss';
import OverviewPane from './overview-pane';
import { redisCategoryText, type RedisCategoryCopyKey } from './redis-copy';

type RedisContentTab = 'overview' | 'focus' | 'performance' | 'memory' | 'clients';

interface RedisTabItem {
  key: RedisContentTab | 'replication' | 'cluster' | 'slowlog';
  copy: RedisCategoryCopyKey;
  /** HOLD：灰掉且不可进入，避免用空数据冒充已采集。 */
  hold?: boolean;
}

const TABS: readonly RedisTabItem[] = [
  { key: 'overview', copy: 'tabOverview' },
  { key: 'focus', copy: 'tabFocus' },
  { key: 'performance', copy: 'tabPerformance' },
  { key: 'memory', copy: 'tabMemory' },
  { key: 'clients', copy: 'tabClients' },
  { key: 'replication', copy: 'tabReplication', hold: true },
  { key: 'cluster', copy: 'tabCluster', hold: true },
  { key: 'slowlog', copy: 'tabSlowlog', hold: true }
];

const CONTENT_TABS = new Set<string>(['overview', 'focus', 'performance', 'memory', 'clients']);

interface RedisCategoryLayoutProps {
  instanceId: React.Key;
  monitorObjectId: React.Key;
  timeValues: TimeValuesProps;
  queryTimeRange: { startMs: number; endMs: number } | null;
  loading: boolean;
  pendingMetricNames: ReadonlySet<string>;
  summaryCards: PreparedSummaryCard[];
}

export default function RedisCategoryLayout({
  instanceId,
  monitorObjectId,
  timeValues,
  queryTimeRange,
  loading,
  pendingMetricNames,
  summaryCards
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
            onChange={(key) => {
              if (CONTENT_TABS.has(key)) setTab(key as RedisContentTab);
            }}
            items={TABS.map((item) => ({
              key: item.key,
              disabled: Boolean(item.hold),
              label: (
                <span title={item.hold ? text('holdHint') : undefined}>
                  {text(item.copy)}
                </span>
              )
            }))}
          />
        </div>
        <div className={layered.main}>
          {tab === 'overview' ? (
            <OverviewPane summaryCards={summaryCards} loading={kpiLoading} />
          ) : (
            <div data-redis-pane={tab}>
              <ProSection title={text(TABS.find((item) => item.key === tab)?.copy || 'tabFocus')}>
                <GrayTabEmptyState title={text('paneEmpty')} />
              </ProSection>
            </div>
          )}
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
