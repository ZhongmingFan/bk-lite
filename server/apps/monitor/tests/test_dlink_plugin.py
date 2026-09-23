"""Contract tests for the D-Link DGS/DXS switch SNMP plugin.

Validates Telegraf/snmp/switch_dlink against the NEW switch layout
(switch_dasan / switch_cambium / switch_robustel):

- AGENT-GENERAL-MIB CPU scalars as top-level fields (GET instance .0).
- DRAM table fields without .0; unitID is a tag (0 = local device).
- IF-MIB comes from ``# @bk_include_ifmib_table``, not an expanded ifTable.
- Fan / PSU / temperature / Flash are out of this round unless present.
- Switch object only; no router child.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from apps.core.utils.loader import LanguageLoader
from apps.monitor.management.services.plugin_migrate import _expand_local_template_assets

SERVER_ROOT = Path(__file__).resolve().parents[3]
PLUGINS = SERVER_ROOT / "apps" / "monitor" / "support-files" / "plugins" / "Telegraf"
DLINK_DIR = PLUGINS / "snmp" / "switch_dlink"
CISCO_DIR = PLUGINS / "snmp" / "switch_cisco"

BRAND = "dlink"
COLLECT_TYPE = "snmp_dlink"
CONFIG_TYPE = "dlink"
PLUGIN_NAME = "Switch D-Link SNMP"
OBJECT_NAME = "Switch"
SYSOBJECTID_GATE = "1.3.6.1.4.1.171.10"
IFMIB_INCLUDE = "# @bk_include_ifmib_table"

SUPPORTED_SCALAR_UNITS = {
    "byteps", "bytes", "counts", "cps", "percent", "celsius", "s", "short", "none",
}

CPU_SCALARS = {
    "device_cpu_usage_5s": "1.3.6.1.4.1.171.12.1.1.6.1.0",
    "device_cpu_usage_1min": "1.3.6.1.4.1.171.12.1.1.6.2.0",
    "device_cpu_usage": "1.3.6.1.4.1.171.12.1.1.6.3.0",
}
SYSTEM_SCALARS = {
    "uptime": "1.3.6.1.2.1.1.3.0",
    "source": "1.3.6.1.2.1.1.5.0",
}
DRAM_TABLE_FIELDS = {
    "unitID": "1.3.6.1.4.1.171.12.1.1.9.1.1",
    "total": "1.3.6.1.4.1.171.12.1.1.9.1.2",
    "used": "1.3.6.1.4.1.171.12.1.1.9.1.3",
    "usage": "1.3.6.1.4.1.171.12.1.1.9.1.4",
}
EXPECTED_METRICS = {
    "device_cpu_usage",
    "device_cpu_usage_5s",
    "device_cpu_usage_1min",
    "device_memory_total",
    "device_memory_used",
    "device_memory_usage",
    "snmp_uptime",
}
ABSENT_HEALTH_METRICS = (
    "device_temperature_celsius",
    "device_fan_state",
    "device_psu_state",
)
_FIELD_RE = re.compile(
    r"\[\[inputs\.snmp\.(field|table\.field)\]\](.*?)(?=\[\[inputs\.snmp\.|\Z)",
    re.S,
)


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _snmp_fields(toml_text):
    parsed = []
    for kind, body in _FIELD_RE.findall(toml_text):
        oid = re.search(r'oid = "([^"]+)"', body).group(1)
        name = re.search(r'name = "([^"]+)"', body).group(1)
        parsed.append((kind, name, oid, "is_tag = true" in body))
    return parsed


@pytest.fixture(scope="module")
def metrics():
    return _read_json(DLINK_DIR / "metrics.json")


@pytest.fixture(scope="module")
def cisco_metrics():
    return _read_json(CISCO_DIR / "metrics.json")


@pytest.fixture(scope="module")
def policy():
    return _read_json(DLINK_DIR / "policy.json")


@pytest.fixture(scope="module")
def ui():
    return _read_json(DLINK_DIR / "UI.json")


@pytest.fixture(scope="module")
def toml_text():
    return (DLINK_DIR / "dlink.child.toml.j2").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def languages():
    return {
        lang: LanguageLoader("monitor", lang).translations
        for lang in ("zh-Hans", "en")
    }


# --------------------------------------------------------------------------- #
# cross-file identity
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_collect_type_consistent_across_files(metrics, policy, ui, toml_text):
    assert COLLECT_TYPE in metrics["status_query"]
    assert "instance_type='switch'" in metrics["status_query"]
    assert ui["collect_type"] == COLLECT_TYPE
    assert f'collect_type = "{COLLECT_TYPE}"' in toml_text
    assert metrics["plugin"] == PLUGIN_NAME
    assert policy["plugin"] == PLUGIN_NAME
    assert metrics["name"] == OBJECT_NAME
    assert ui["object_name"] == OBJECT_NAME
    assert policy["object"] == OBJECT_NAME


@pytest.mark.unit
def test_config_type_consistent(ui, toml_text):
    assert ui["config_type"] == [CONFIG_TYPE]
    assert f'config_type = "{CONFIG_TYPE}"' in toml_text
    assert f'brand = "{BRAND}"' in toml_text


@pytest.mark.unit
def test_ui_is_pure_snmp_form(ui):
    assert not any(f["name"] == "brand" for f in ui["form_fields"])


@pytest.mark.unit
def test_no_router_object():
    assert not (PLUGINS / "snmp" / "router_dlink").exists()


@pytest.mark.unit
def test_sysobjectid_gate_is_dlink_products(metrics, toml_text):
    assert SYSOBJECTID_GATE in metrics["plugin_desc"]
    assert SYSOBJECTID_GATE in toml_text


# --------------------------------------------------------------------------- #
# overlapping device_* group/unit vs Cisco (query shape may differ)
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_shared_device_metrics_match_cisco_group_and_unit(metrics, cisco_metrics):
    cisco = {m["name"]: m for m in cisco_metrics["metrics"]}
    drift = []
    for m in metrics["metrics"]:
        base = cisco.get(m["name"])
        if base is None:
            continue
        if m["metric_group"] != base["metric_group"]:
            drift.append(f'{m["name"]}.group')
        if m["unit"] != base["unit"]:
            drift.append(f'{m["name"]}.unit')
    assert drift == [], f"device_* drift vs Cisco: {drift}"


@pytest.mark.unit
def test_cpu_headline_is_snmp_prefixed_scalar_query(metrics):
    by_name = {m["name"]: m for m in metrics["metrics"]}
    for name in CPU_SCALARS:
        query = by_name[name]["query"]
        assert query.startswith("snmp_"), f"{name} must be a top-level snmp_ series"
        assert by_name[name]["unit"] == "percent"
        assert by_name[name]["dimensions"] == []


# --------------------------------------------------------------------------- #
# AGENT-GENERAL CPU scalars (.0) and DRAM table (no .0)
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_cpu_scalars_use_agent_general_oids_with_instance_zero(toml_text):
    fields = {name: oid for kind, name, oid, _tag in _snmp_fields(toml_text) if kind == "field"}
    for name, oid in {**SYSTEM_SCALARS, **CPU_SCALARS}.items():
        assert fields[name] == oid
        assert oid.endswith(".0")


@pytest.mark.unit
def test_dram_table_fields_have_no_instance_zero(toml_text):
    tables = {
        name: (oid, is_tag)
        for kind, name, oid, is_tag in _snmp_fields(toml_text)
        if kind == "table.field"
    }
    for name, oid in DRAM_TABLE_FIELDS.items():
        assert name in tables, f"missing DRAM column {name}"
        got_oid, is_tag = tables[name]
        assert got_oid == oid
        assert not got_oid.endswith(".0")
    assert tables["unitID"][1] is True
    assert "index_as_tag = true" in toml_text
    assert 'name = "device_memory"' in toml_text
    assert 'oid = "1.3.6.1.4.1.171.12.1.1.9"' not in toml_text


@pytest.mark.unit
def test_every_scalar_has_dot_zero_every_table_field_does_not(toml_text):
    for kind, name, oid, _tag in _snmp_fields(toml_text):
        if kind == "field":
            assert oid.endswith(".0"), f"scalar {name} missing .0: {oid}"
        else:
            assert not oid.endswith(".0"), f"table.field {name} has .0: {oid}"


@pytest.mark.unit
def test_memory_total_used_convert_kb_to_bytes_usage_is_vendor_percent(metrics):
    by_name = {m["name"]: m for m in metrics["metrics"]}
    assert set(by_name) == EXPECTED_METRICS
    for name in ("device_memory_total", "device_memory_used"):
        assert by_name[name]["unit"] == "bytes"
        assert "* 1024" in by_name[name]["query"]
        dims = [d["name"] for d in by_name[name]["dimensions"]]
        assert "unitID" in dims
    usage = by_name["device_memory_usage"]
    assert usage["unit"] == "percent"
    assert usage["query"] == "device_memory_usage{instance_type='switch', __$labels__}"
    assert "unitID" in [d["name"] for d in usage["dimensions"]]
    assert "device_memory_free" not in by_name


# --------------------------------------------------------------------------- #
# IF-MIB include — source must not expand ifTable
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_toml_uses_ifmib_include_and_does_not_expand_iftable(toml_text):
    assert IFMIB_INCLUDE in toml_text
    assert toml_text.count(IFMIB_INCLUDE) == 1
    assert "ifHCInOctets" not in toml_text
    assert "ifHCOutOctets" not in toml_text
    assert 'oid = "1.3.6.1.2.1.2.2"' not in toml_text
    assert "[[inputs.snmp.table]]" in toml_text  # DRAM table only in source


@pytest.mark.unit
def test_ifmib_include_expands_at_import(toml_text):
    expanded = _expand_local_template_assets(toml_text, DLINK_DIR)
    assert IFMIB_INCLUDE not in expanded
    assert "ifHCInOctets" in expanded
    assert "ifHCOutOctets" in expanded
    assert "1.3.6.1.2.1.2.2" in expanded


# --------------------------------------------------------------------------- #
# fan/psu/temp/flash are out of this round unless present
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_absent_health_metrics_are_not_declared(metrics, policy, toml_text):
    names = {m["name"] for m in metrics["metrics"]}
    policy_metrics = {t["metric_name"] for t in policy["templates"]}
    for absent in ABSENT_HEALTH_METRICS:
        assert absent not in names
        assert absent not in policy_metrics
        assert absent not in toml_text
    assert "[[processors.enum]]" not in toml_text
    assert "device_fan" not in toml_text
    assert "device_psu" not in toml_text
    assert "device_temperature" not in toml_text


# --------------------------------------------------------------------------- #
# policy / units / dimensions hygiene
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_policy_templates_reference_existing_metrics(metrics, policy):
    known = {m["name"] for m in metrics["metrics"]}
    bad = [t["metric_name"] for t in policy["templates"] if t["metric_name"] not in known]
    assert bad == [], f"policy references unknown metrics: {bad}"
    assert {t["metric_name"] for t in policy["templates"]} == {
        "device_cpu_usage",
        "device_memory_usage",
    }


@pytest.mark.unit
def test_all_metric_units_supported(metrics):
    bad = [
        f'{m["name"]}:{m["unit"]}'
        for m in metrics["metrics"]
        if m["data_type"] != "Enum" and m["unit"] not in SUPPORTED_SCALAR_UNITS
    ]
    assert bad == [], f"unsupported units: {bad}"


@pytest.mark.unit
def test_dimensions_well_formed(metrics):
    bad = [
        m["name"]
        for m in metrics["metrics"]
        for d in m.get("dimensions", [])
        if not d.get("name") or not d.get("description")
    ]
    assert bad == [], f"malformed dimensions: {bad}"


# --------------------------------------------------------------------------- #
# i18n completeness (zh-Hans + en)
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_plugin_has_bilingual_name_and_desc(languages):
    for lang, data in languages.items():
        entry = (data.get("monitor_object_plugin") or {}).get(PLUGIN_NAME) or {}
        assert entry.get("name"), f"{lang}: plugin name missing"
        assert entry.get("desc"), f"{lang}: plugin desc missing"


@pytest.mark.unit
def test_every_metric_has_bilingual_translation(metrics, languages):
    missing = []
    for lang, data in languages.items():
        group = (data.get("monitor_object_metric") or {}).get(OBJECT_NAME) or {}
        for m in metrics["metrics"]:
            entry = group.get(m["name"]) or {}
            if not entry.get("name") or not entry.get("desc"):
                missing.append(f'{lang}:{m["name"]}')
    assert missing == [], f"metrics missing translation: {missing}"


@pytest.mark.unit
def test_every_metric_group_has_bilingual_translation(metrics, languages):
    groups = {m["metric_group"] for m in metrics["metrics"]}
    missing = []
    for lang, data in languages.items():
        trans = (data.get("monitor_object_metric_group") or {}).get(OBJECT_NAME) or {}
        missing += [f"{lang}:{g}" for g in groups if not trans.get(g)]
    assert missing == [], f"metric groups missing translation: {missing}"


@pytest.mark.unit
def test_object_has_bilingual_translation(languages):
    for lang, data in languages.items():
        obj = (data.get("monitor_object") or {}).get(OBJECT_NAME)
        assert obj, f"{lang}: object {OBJECT_NAME} missing translation"


# --------------------------------------------------------------------------- #
# secrets never inlined as plaintext
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_passwords_use_sidecar_env_placeholders_not_plaintext(ui, toml_text):
    field_names = {field["name"] for field in ui["form_fields"]}
    assert "ENV_AUTH_PASSWORD" in field_names
    assert "ENV_PRIV_PASSWORD" in field_names
    assert "auth_password" not in field_names
    assert "priv_password" not in field_names
    assert 'auth_password = "${AUTH_PASSWORD__{{ config_id }}}"' in toml_text
    assert 'priv_password = "${PRIV_PASSWORD__{{ config_id }}}"' in toml_text
    assert "{{ auth_password }}" not in toml_text
    assert "{{ priv_password }}" not in toml_text
