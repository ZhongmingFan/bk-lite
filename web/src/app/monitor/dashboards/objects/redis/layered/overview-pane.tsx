'use client';

import React, { useMemo } from 'react';
import {
  KpiTileRow,
  ProSection,
  type KpiTileModel
} from '../../host/layered/pro-chrome';
import type { PreparedSummaryCard } from '../../common/simple-dashboard-core';
import { useDashboardText } from '../../../shared/utils/content-i18n';
import { redisCategoryText, type RedisCategoryCopyKey } from './redis-copy';

interface OverviewPaneProps {
  summaryCards: PreparedSummaryCard[];
  loading: boolean;
}

const headline = (main: PreparedSummaryCard['mainValue']) => (
  main.unit ? `${main.value}${main.unit}` : main.value
);

/** 概览只放传统五 KPI。重点关注、趋势留给后续切片。 */
export default function OverviewPane({ summaryCards, loading }: OverviewPaneProps) {
  const { t, dt } = useDashboardText();
  const text = (key: RedisCategoryCopyKey) => redisCategoryText(t, key);
  const cards = useMemo<KpiTileModel[]>(() => (
    summaryCards.map((summary) => {
      const empty = summary.mainValue.value === '--';
      return {
        key: summary.card.metric,
        label: dt(summary.card.title),
        value: headline(summary.mainValue),
        tone: empty ? undefined : (summary.valueColor ?? summary.card.color),
        subtitle: summary.card.isUptimeCard && summary.uptimeState
          ? dt(summary.uptimeState.label)
          : undefined
      };
    })
  ), [dt, summaryCards]);

  return (
    <div data-redis-pane="overview">
      <ProSection title={text('healthOverview')}>
        <KpiTileRow items={cards} loading={loading} columns={5} />
      </ProSection>
    </div>
  );
}
