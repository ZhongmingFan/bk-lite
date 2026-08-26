import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

import {
  buildBulkApplyPayload,
  buildCollectionPolicyBulkConfig,
  buildPolicyPreview,
  changeBulkAssetPage,
  clearTemplateSelection,
  canDeleteTemplates,
  containsBuiltinTemplate,
  getAssetCollectionTemplateLabels,
  getAssetOrganizationText,
  getPrimaryNoticeType,
  groupPolicyTemplates,
  normalizeBulkConfig,
  reconcileBulkAssetSelection,
  resetBulkAssetPageForSearch,
  selectTemplateGroup,
  toggleTemplateSelection,
} from '../src/app/monitor/(pages)/event/template/templateBulkUtils';
import {
  COLLECTION_POLICY_FIELD,
  buildCollectionPolicyApplyPayload,
  defaultSelectedTemplateKeys,
  extractCollectInstanceIds,
  omitCollectionPolicyField,
  resolvePolicyTemplateList,
  selectedPolicyTemplates,
  shouldSkipPolicyCreate,
} from '../src/app/monitor/(pages)/integration/list/detail/configure/automaticPolicyApply';

const templates = [
  {
    template_key: 'host-remote:0',
    name: 'CPU 使用率过高',
    description: 'CPU too high',
    metric_name: 'cpu_usage_total',
    template_group: '主机（Host Remote）',
    plugin_id: 11,
    plugin_display_name: 'Host Remote',
  },
  {
    template_key: 'host-remote:1',
    name: '内存使用率过高',
    description: 'Memory too high',
    metric_name: 'mem_used_percent',
    template_group: '主机（Host Remote）',
    plugin_id: 11,
    plugin_display_name: 'Host Remote',
  },
  {
    template_key: 'telegraf:0',
    name: '磁盘使用率过高',
    description: 'Disk too high',
    metric_name: 'disk_used_percent',
    template_group: '主机（Telegraf）',
    plugin_id: 12,
    plugin_display_name: 'Telegraf',
  },
];

const templatePageSource = readFileSync(
  new URL('../src/app/monitor/(pages)/event/template/page.tsx', import.meta.url),
  'utf8'
);
const searchPosition = templatePageSource.indexOf('<Input');
const importPosition = templatePageSource.indexOf('<Upload', searchPosition);
const bulkPosition = templatePageSource.indexOf('<Dropdown', importPosition);
assert.ok(searchPosition >= 0 && searchPosition < importPosition);
assert.ok(importPosition < bulkPosition);
assert.match(templatePageSource, /disabled: !selectedTemplateKeys\.length \|\| containsBuiltin/);
assert.match(templatePageSource, /内置模版不可删除/);

const strategyPageSource = readFileSync(
  new URL('../src/app/monitor/(pages)/event/strategy/detail/page.tsx', import.meta.url),
  'utf8'
);
const confirmPosition = strategyPageSource.indexOf("{t('common.confirm')}");
const saveTemplatePosition = strategyPageSource.indexOf('保存模版', confirmPosition);
const cancelPosition = strategyPageSource.indexOf("{t('common.cancel')}", saveTemplatePosition);
assert.ok(confirmPosition >= 0 && confirmPosition < saveTemplatePosition);
assert.ok(saveTemplatePosition < cancelPosition);
assert.match(strategyPageSource, /validateFields\(templateFields\)/);

const groups = groupPolicyTemplates(templates);
assert.deepEqual(groups.map((group) => ({
  name: group.name,
  total: group.templates.length,
  selected: group.selectedCount,
})), [
  { name: '主机（Host Remote）', total: 2, selected: 0 },
  { name: '主机（Telegraf）', total: 1, selected: 0 },
]);

const selectedOne = toggleTemplateSelection([], templates[0]);
assert.deepEqual(selectedOne, ['host-remote:0']);

const selectedGroup = selectTemplateGroup(selectedOne, groups[0].templates, true);
assert.deepEqual(selectedGroup, ['host-remote:0', 'host-remote:1']);
assert.deepEqual(clearTemplateSelection(), []);
assert.equal(canDeleteTemplates([]), false);
assert.equal(canDeleteTemplates([{ template_type: 'custom', deletable: true }]), true);
assert.equal(
  containsBuiltinTemplate([
    { template_type: 'custom', deletable: true },
    { template_type: 'builtin', deletable: false },
  ]),
  true
);
assert.equal(
  canDeleteTemplates([
    { template_type: 'custom', deletable: true },
    { template_type: 'builtin', deletable: false },
  ]),
  false
);

const selectedTemplates = templates.filter((item) => selectedGroup.includes(item.template_key));
const assets = [
  {
    instance_id: "('host-a',)",
    instance_name: 'host-a',
    organization: [7],
    plugins: [{ id: 11, display_name: 'Host Remote' }],
  },
  {
    instance_id: "('host-b',)",
    instance_name: 'host-b',
    organization: [8],
    plugins: [{ id: 11, display_name: 'Host Remote' }],
  },
];
const config = {
  name_prefix: '批量策略',
  enable: true,
  schedule: { type: 'min', value: 5 },
  period: { type: 'min', value: 10 },
  trigger_count: 2,
  notice: true,
  notice_type_ids: [1, 2],
  notice_type: 'email',
  notice_users: ['alice'],
};

const preview = buildPolicyPreview(selectedTemplates, assets, config);
assert.deepEqual(preview.map((item) => item.name), [
  '批量策略-CPU 使用率过高',
  '批量策略-内存使用率过高',
]);
assert.equal(preview[0].metricLabel, 'Host Remote - cpu_usage_total');
assert.equal(preview[0].assetScopeLabel, '覆盖 2 个实例：host-a、host-b');
assert.equal(preview[0].statusLabel, '启用');

const payload = buildBulkApplyPayload({
  monitorObjectId: 3,
  templates: selectedTemplates,
  assets,
  config,
});
assert.equal(payload.monitor_object, 3);
assert.deepEqual(payload.asset_ids, ["('host-a',)", "('host-b',)"]);
assert.deepEqual(payload.template_keys, ['host-remote:0', 'host-remote:1']);
assert.deepEqual(payload.config.notice_type_ids, [1, 2]);
assert.equal(payload.config.notice_type, 'email');
assert.equal(payload.config.trigger_count, 2);
assert.equal('no_data_level' in payload.config, false);
assert.equal('no_data_alert_name' in payload.config, false);

const defaultTriggerConfig = normalizeBulkConfig({
  ...config,
  trigger_count: undefined,
});
assert.equal(defaultTriggerConfig.trigger_count, 1);

const noDataDisabledConfig = normalizeBulkConfig({
  ...config,
  no_data_enabled: false,
  enable_alerts: ['threshold', 'no_data'],
  no_data_level: 'warning',
  no_data_alert_name: '无数据告警',
});
assert.deepEqual(noDataDisabledConfig.enable_alerts, ['threshold']);
assert.equal('no_data_level' in noDataDisabledConfig, false);
assert.equal('no_data_alert_name' in noDataDisabledConfig, false);

const noDataEnabledConfig = normalizeBulkConfig({
  ...config,
  no_data_enabled: true,
  no_data_period: { type: 'min', value: 5 },
  no_data_level: 'warning',
  no_data_alert_name: '无数据告警',
});
assert.deepEqual(noDataEnabledConfig.enable_alerts, ['threshold', 'no_data']);
assert.deepEqual(noDataEnabledConfig.no_data_period, { type: 'min', value: 5 });
assert.deepEqual(noDataEnabledConfig.no_data_recovery_period, { type: 'min', value: 5 });
assert.equal(noDataEnabledConfig.no_data_level, 'warning');
assert.equal(noDataEnabledConfig.no_data_alert_name, '无数据告警');

const noDataPayload = buildBulkApplyPayload({
  monitorObjectId: 3,
  templates: selectedTemplates,
  assets,
  config: noDataEnabledConfig,
});
assert.equal(noDataPayload.config.no_data_level, 'warning');
assert.equal(noDataPayload.config.no_data_alert_name, '无数据告警');
assert.equal(
  getPrimaryNoticeType([2, 1], [
    { id: 1, channel_type: 'email' },
    { id: 2, channel_type: 'nats' },
  ]),
  'nats'
);
assert.equal(getPrimaryNoticeType([], [{ id: 1, channel_type: 'email' }]), '');

assert.equal(
  getAssetOrganizationText(
    { organization: [7, 8] },
    [
      {
        value: 7,
        label: '生产环境',
        children: [{ value: 8, label: '核心业务' }],
      },
    ]
  ),
  '生产环境,核心业务'
);
assert.deepEqual(
  getAssetCollectionTemplateLabels({
    plugins: [
      { id: 11, display_name: 'Host Remote' },
      { id: 12, name: 'Telegraf' },
      { id: 13 },
    ],
  }),
  ['Host Remote', 'Telegraf', '13']
);

const firstAssetPage = [
  { instance_id: 'host-a', instance_name: 'host-a' },
  { instance_id: 'host-b', instance_name: 'host-b' },
];
const secondAssetPage = [
  { instance_id: 'host-c', instance_name: 'host-c' },
  { instance_id: 'host-d', instance_name: 'host-d' },
];

const firstPageSelection = reconcileBulkAssetSelection(
  [],
  firstAssetPage,
  ['host-a']
);
const crossPageSelection = reconcileBulkAssetSelection(
  firstPageSelection.selectedAssets,
  secondAssetPage,
  ['host-a', 'host-c']
);
assert.deepEqual(crossPageSelection.selectedAssetIds, ['host-a', 'host-c']);
assert.deepEqual(
  crossPageSelection.selectedAssets.map((asset) => asset.instance_id),
  ['host-a', 'host-c']
);

// 翻页或切换搜索结果本身不会触发选择变更，已选缓存仍包含完整记录。
assert.deepEqual(
  crossPageSelection.selectedAssets.map((asset) => asset.instance_id),
  ['host-a', 'host-c']
);
const selectionAfterCancel = reconcileBulkAssetSelection(
  crossPageSelection.selectedAssets,
  firstAssetPage,
  ['host-c']
);
assert.deepEqual(selectionAfterCancel.selectedAssetIds, ['host-c']);
assert.deepEqual(
  selectionAfterCancel.selectedAssets.map((asset) => asset.instance_id),
  ['host-c']
);

const assetPagination = { current: 3, pageSize: 8, total: 64 };
assert.deepEqual(changeBulkAssetPage(assetPagination, 4, 8), {
  current: 4,
  pageSize: 8,
  total: 64,
});
assert.deepEqual(changeBulkAssetPage(assetPagination, 3, 20), {
  current: 1,
  pageSize: 20,
  total: 64,
});
assert.deepEqual(resetBulkAssetPageForSearch(assetPagination), {
  current: 1,
  pageSize: 8,
  total: 64,
});

const bulkApplyModalSource = readFileSync(
  resolve('src/app/monitor/(pages)/event/template/bulkApplyModal.tsx'),
  'utf8'
);
assert.match(bulkApplyModalSource, /name: assetNameQuery/, '资产搜索应传递 name 参数');
assert.match(
  bulkApplyModalSource,
  /page_size: assetPagination\.pageSize/,
  '资产请求应使用受控的每页数量'
);
assert.doesNotMatch(
  bulkApplyModalSource,
  /page_size:\s*1000/,
  '资产列表不应再受 1000 条硬上限约束'
);
assert.match(
  bulkApplyModalSource,
  /preserveSelectedRowKeys: true/,
  '资产表格应保留跨页选择状态'
);
assert.match(
  bulkApplyModalSource,
  /requestId !== assetRequestIdRef\.current/,
  '旧资产请求响应不应覆盖新查询'
);

const collectionConfig = buildCollectionPolicyBulkConfig();
assert.deepEqual(collectionConfig.enable_alerts, ['threshold']);
assert.equal(collectionConfig.no_data_enabled, false);
assert.equal('no_data' in (collectionConfig.enable_alerts || []), false);
assert.equal('no_data_level' in collectionConfig, false);
assert.equal('no_data_period' in collectionConfig, false);
assert.equal('no_data_recovery_period' in collectionConfig, false);
assert.equal('no_data_alert_name' in collectionConfig, false);
assert.equal(collectionConfig.name_prefix, '');
assert.equal(collectionConfig.notice, false);
assert.equal(collectionConfig.enable, true);
assert.deepEqual(collectionConfig.schedule, { type: 'min', value: 5 });
assert.deepEqual(collectionConfig.period, { type: 'min', value: 5 });
assert.equal(collectionConfig.trigger_count, 1);

const collectionPayload = buildCollectionPolicyApplyPayload({
  monitorObjectId: 3,
  templates: selectedTemplates,
  instanceIds: ["('host-a',)", "('host-b',)"],
});
assert.ok(collectionPayload);
assert.equal(collectionPayload.monitor_object, 3);
assert.deepEqual(collectionPayload.template_keys, ['host-remote:0', 'host-remote:1']);
assert.deepEqual(collectionPayload.asset_ids, ["('host-a',)", "('host-b',)"]);
assert.deepEqual(collectionPayload.config.enable_alerts, ['threshold']);
assert.equal(collectionPayload.config.no_data_enabled, false);
assert.equal('no_data_level' in collectionPayload.config, false);
assert.equal('no_data_alert_name' in collectionPayload.config, false);

assert.equal(
  buildCollectionPolicyApplyPayload({
    monitorObjectId: 3,
    templates: selectedTemplates,
    instanceIds: [],
  }),
  null
);
assert.equal(
  buildCollectionPolicyApplyPayload({
    monitorObjectId: 3,
    templates: [],
    instanceIds: ["('host-a',)"],
  }),
  null
);

const mixedPluginTemplates = [
  { template_key: 'builtin:1', name: 'CPU', plugin_id: 12 },
  { template_key: 'builtin:2', name: 'Memory', plugin_id: 12 },
  { template_key: 'builtin:3', name: 'Remote CPU', plugin_id: 11 },
  { name: 'no-key', plugin_id: 12 },
];
const pluginTemplates = resolvePolicyTemplateList(mixedPluginTemplates, '12');
assert.deepEqual(
  pluginTemplates.map((item) => item.template_key),
  ['builtin:1', 'builtin:2']
);
assert.deepEqual(defaultSelectedTemplateKeys(pluginTemplates), ['builtin:1', 'builtin:2']);
assert.equal(shouldSkipPolicyCreate([]), true);
assert.equal(shouldSkipPolicyCreate(undefined), true);
assert.equal(shouldSkipPolicyCreate(['builtin:1']), false);
assert.deepEqual(
  selectedPolicyTemplates(pluginTemplates, ['builtin:2']).map((item) => item.template_key),
  ['builtin:2']
);
assert.deepEqual(
  extractCollectInstanceIds({ instance_ids: ["('host-1',)"] }, { instances: [{ instance_id: 'host-1' }] }),
  ["('host-1',)"]
);
assert.deepEqual(
  extractCollectInstanceIds({}, { instances: [{ instance_id: 'plain-id' }] }),
  ['plain-id']
);
assert.equal(
  COLLECTION_POLICY_FIELD in omitCollectionPolicyField({
    interval: 60,
    [COLLECTION_POLICY_FIELD]: ['builtin:1'],
  }),
  false
);

const automaticSource = readFileSync(
  resolve('src/app/monitor/(pages)/integration/list/detail/configure/automatic.tsx'),
  'utf8'
);
const formItemsPosition = automaticSource.indexOf('{formItems}');
const policyFieldPosition = automaticSource.indexOf(
  'name={COLLECTION_POLICY_FIELD}',
  formItemsPosition
);
const basicInfoPosition = automaticSource.indexOf(
  "t('monitor.integrations.basicInformation')",
  formItemsPosition
);
assert.ok(formItemsPosition >= 0 && formItemsPosition < policyFieldPosition);
assert.ok(policyFieldPosition < basicInfoPosition);
assert.match(automaticSource, /mode="multiple"/);
assert.match(automaticSource, /bulkCreatePoliciesFromTemplates/);
assert.match(automaticSource, /buildCollectionPolicyApplyPayload/);
assert.doesNotMatch(
  automaticSource,
  /enable_alerts:\s*\[[^\]]*no_data/,
  '接入页不得为该路径开启无数据告警'
);

console.log('monitor-template-bulk logic validation passed');
