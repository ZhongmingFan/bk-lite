import { MODULE_OBJECT_QUERY_PARAM } from '@/app/monitor/utils/monitorObjectQuery';

const ASSET_PATH = '/monitor/integration/asset';

export const buildCollectNeedUpdateAssetUrl = (options?: {
  monitorObjectId?: number | string | null;
  pluginId?: number | string | null;
}): string => {
  const params = new URLSearchParams();
  params.set('need_update', '1');
  const objectId = options?.monitorObjectId;
  if (objectId != null && String(objectId).trim() !== '') {
    params.set(MODULE_OBJECT_QUERY_PARAM, String(objectId));
  }
  const pluginId = options?.pluginId;
  if (pluginId != null && String(pluginId).trim() !== '') {
    params.set('monitor_plugin_id', String(pluginId));
  }
  return `${ASSET_PATH}?${params.toString()}`;
};
