// @vitest-environment node

import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';

import en from '@/app/monitor/locales/en.json';
import zh from '@/app/monitor/locales/zh.json';
import { buildCollectionPolicyBulkConfig } from '@/app/monitor/(pages)/event/template/templateBulkUtils';
import {
  COLLECTION_POLICY_FIELD,
  buildCollectionPolicyApplyPayload,
  defaultSelectedTemplateKeys,
  extractCollectInstanceIds,
  omitCollectionPolicyField,
  resolvePolicyTemplateList,
  shouldSkipPolicyCreate
} from '../automaticPolicyApply';

describe('automatic collection monitoring policy apply', () => {
  it('names the field in both supported languages', () => {
    expect(zh.monitor.integrations.monitoringPolicy).toBe('监控策略');
    expect(en.monitor.integrations.monitoringPolicy).toBe('Monitoring Policy');
    expect(zh.monitor.integrations.policyCreateFailed).toContain('{error}');
    expect(en.monitor.integrations.policyCreateFailed).toContain('{error}');
  });

  it('keeps only templates of the current plugin and defaults them selected', () => {
    const templates = resolvePolicyTemplateList(
      [
        { template_key: 'builtin:1', name: 'CPU', plugin_id: 12 },
        { template_key: 'builtin:2', name: 'Disk', plugin_id: 12 },
        { template_key: 'builtin:3', name: 'Remote CPU', plugin_id: 11 }
      ],
      12
    );

    expect(templates.map((item) => item.template_key)).toEqual([
      'builtin:1',
      'builtin:2'
    ]);
    expect(defaultSelectedTemplateKeys(templates)).toEqual([
      'builtin:1',
      'builtin:2'
    ]);
    expect(shouldSkipPolicyCreate([])).toBe(true);
  });

  it('builds a bulk payload with no-data alerts disabled', () => {
    const config = buildCollectionPolicyBulkConfig();
    const payload = buildCollectionPolicyApplyPayload({
      monitorObjectId: 3,
      templates: [
        { template_key: 'builtin:1', name: 'CPU', plugin_id: 12 }
      ],
      instanceIds: ["('host-a',)"]
    });

    expect(config.enable_alerts).toEqual(['threshold']);
    expect(config.no_data_enabled).toBe(false);
    expect(config).not.toHaveProperty('no_data_level');
    expect(config).not.toHaveProperty('no_data_period');
    expect(payload?.config.enable_alerts).toEqual(['threshold']);
    expect(payload?.config).not.toHaveProperty('no_data_alert_name');
    expect(
      buildCollectionPolicyApplyPayload({
        monitorObjectId: 3,
        templates: [{ template_key: 'builtin:1' }],
        instanceIds: []
      })
    ).toBeNull();
  });

  it('prefers stored instance ids from collect result and omits the form field from collect values', () => {
    expect(
      extractCollectInstanceIds(
        { instance_ids: ["('host-1',)"] },
        { instances: [{ instance_id: 'host-1' }] }
      )
    ).toEqual(["('host-1',)"]);
    expect(
      COLLECTION_POLICY_FIELD in
        omitCollectionPolicyField({
          interval: 60,
          [COLLECTION_POLICY_FIELD]: ['builtin:1']
        })
    ).toBe(false);
  });

  it('places the field between interval form items and the monitored object table', () => {
    const source = readFileSync(new URL('../automatic.tsx', import.meta.url), 'utf8');
    const formItemsPosition = source.indexOf('{formItems}');
    const policyFieldPosition = source.indexOf(
      'name={COLLECTION_POLICY_FIELD}',
      formItemsPosition
    );
    const basicInfoPosition = source.indexOf(
      "t('monitor.integrations.basicInformation')",
      formItemsPosition
    );

    expect(formItemsPosition).toBeGreaterThanOrEqual(0);
    expect(policyFieldPosition).toBeGreaterThan(formItemsPosition);
    expect(basicInfoPosition).toBeGreaterThan(policyFieldPosition);
  });
});
