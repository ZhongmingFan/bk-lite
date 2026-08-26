import importlib.util
import sys
import types
from pathlib import Path


def _install_module(monkeypatch, name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _load_policy_service_module(monkeypatch, rows):
    class _QuerySet:
        def __init__(self, items=None):
            self.items = list(rows if items is None else items)

        def filter(self, *args, **kwargs):
            items = self.items
            plugin_id = kwargs.get("plugin_id")
            if plugin_id is not None:
                items = [row for row in items if row.plugin_id == plugin_id]
            return _QuerySet(items)

        def order_by(self, *args):
            return self.items

        def values_list(self, *args, **kwargs):
            return []

    _install_module(
        monkeypatch,
        "apps.monitor.models",
        PolicyTemplate=types.SimpleNamespace(
            TYPE_BUILTIN="builtin",
            TYPE_CUSTOM="custom",
            objects=types.SimpleNamespace(select_related=lambda *args: _QuerySet()),
        ),
        MonitorPlugin=object,
        MonitorObject=object,
    )

    spec = importlib.util.spec_from_file_location(
        "policy_service_templates_module",
        Path(__file__).resolve().parents[1] / "services" / "policy.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_get_policy_templates_adds_default_trigger_count_and_preserves_explicit_value(monkeypatch):
    monitor_object = types.SimpleNamespace(id=1, name="Host", display_name="主机")
    plugin = types.SimpleNamespace(id=9, name="Telegraf", display_name="Telegraf", collector="Telegraf")
    rows = [
        types.SimpleNamespace(
            id=7,
            key="builtin:cpu",
            scope_key="builtin",
            template_type="builtin",
            monitor_object_id=1,
            monitor_object=monitor_object,
            plugin_id=9,
            plugin=plugin,
            name="CPU",
            description="",
            config={"metric_name": "cpu_usage"},
        ),
        types.SimpleNamespace(
            id=8,
            key="builtin:memory",
            scope_key="builtin",
            template_type="builtin",
            monitor_object_id=1,
            monitor_object=monitor_object,
            plugin_id=9,
            plugin=plugin,
            name="Memory",
            description="",
            config={"metric_name": "memory_usage", "trigger_count": 2},
        ),
    ]
    module = _load_policy_service_module(monkeypatch, rows)

    templates = module.PolicyService.get_policy_templates("Host")

    assert templates[0]["trigger_count"] == 1
    assert templates[1]["trigger_count"] == 2


def test_get_policy_templates_filters_by_plugin_id_and_rejects_invalid_id(monkeypatch):
    monitor_object = types.SimpleNamespace(id=1, name="Host", display_name="主机")
    telegraf = types.SimpleNamespace(id=9, name="Telegraf", display_name="Telegraf", collector="Telegraf")
    remote = types.SimpleNamespace(id=11, name="HostRemote", display_name="Host Remote", collector="NATS")
    rows = [
        types.SimpleNamespace(
            id=7,
            key="builtin:cpu",
            scope_key="builtin",
            template_type="builtin",
            monitor_object_id=1,
            monitor_object=monitor_object,
            plugin_id=9,
            plugin=telegraf,
            name="CPU",
            description="",
            config={"metric_name": "cpu_usage"},
        ),
        types.SimpleNamespace(
            id=8,
            key="builtin:remote-cpu",
            scope_key="builtin",
            template_type="builtin",
            monitor_object_id=1,
            monitor_object=monitor_object,
            plugin_id=11,
            plugin=remote,
            name="Remote CPU",
            description="",
            config={"metric_name": "cpu_usage"},
        ),
    ]
    module = _load_policy_service_module(monkeypatch, rows)

    filtered = module.PolicyService.get_policy_templates("Host", plugin_id="9")
    assert [item["plugin_id"] for item in filtered] == [9]
    assert [item["id"] for item in filtered] == [7]
    assert module.PolicyService.get_policy_templates("Host", plugin_id="not-a-number") == []
