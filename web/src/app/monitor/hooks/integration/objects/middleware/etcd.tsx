export const useEtcdBkpullConfig = () => {
  return {
    instance_type: 'etcd',
    dashboardDisplay: [
      '集群健康',
      '请求性能',
      '网络',
      '存储',
    ],
    tableDiaplay: [
      { type: 'enum', key: 'etcd_server_has_leader_gauge' },
      { type: 'value', key: 'etcd_mvcc_db_total_size_in_use_in_bytes_gauge' },
      { type: 'value', key: 'etcd_server_proposals_failed_total_counter_rate' },
    ],
    groupIds: {},
    collectTypes: {
      Etcd: 'bkpull',
    },
  };
};
