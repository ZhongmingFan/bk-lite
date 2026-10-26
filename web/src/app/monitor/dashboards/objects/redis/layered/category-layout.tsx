'use client';

import React from 'react';
import { Tabs } from 'antd';
import type { TimeValuesProps } from '@/app/monitor/types';
import HostAlertDrawer from '../../host/layered/host-alert-drawer';
import { ProTimeWindowProvider } from '../../host/layered/pro-time-window';
import type { PreparedSummaryCard } from '../../common/simple-dashboard-core';
import { useDashboardText } from '../../../shared/utils/content-i18n';
import layered from '../../host/layered/layered.module.scss';
import OverviewPane from './overview-pane';
import { redisCategoryText, type RedisCategoryCopyKey } from './redis-copy';

/**
 * S1 只露出已绑指标的概览。
 * 重点关注 / 性能与缓存 / 内存 / 连接与网络尚未绑指标，复制·哨兵 / Cluster / 慢命令为 HOLD，
 * 都不进 items，避免灰态或「暂无数据」占位。
 */
const VISIBLE_TABS = [
  { key: 'overview', copy: 'tabOverview' }
] as const satisfies ReadonlyArray<{ key: string; copy: RedisCategoryCopyKey }>;

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
            activeKey="overview"
            items={VISIBLE_TABS.map((item) => ({
              key: item.key,
              label: text(item.copy)
            }))}
          />
        </div>
        <div className={layered.main}>
          <OverviewPane summaryCards={summaryCards} loading={kpiLoading} />
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
