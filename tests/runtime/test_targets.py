from __future__ import annotations

import json
from pathlib import Path

from exoswarm.runtime.targets import blind_target_summaries, load_demo_vault

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_real_demo_registry_loads_only_opaque_agent_context() -> None:
    vault, targets = load_demo_vault(PROJECT_ROOT / "data")
    assert targets == ("TARGET-X17", "TARGET-X42")
    for opaque in targets:
        context = vault.agent_context(opaque)
        serialized = json.dumps(context.model_dump(mode="json")).lower()
        assert context.opaque_target_id == opaque
        assert "tic " not in serialized
        assert "toi-" not in serialized
        assert sorted(context.available_product_roles) == ["light_curve", "target_pixel"]


def test_blind_target_list_contains_no_identity_or_catalog_status() -> None:
    summaries = blind_target_summaries(PROJECT_ROOT / "data")
    serialized = json.dumps(summaries).lower()
    assert {item["opaque_target_id"] for item in summaries} == {"TARGET-X17", "TARGET-X42"}
    assert "tic " not in serialized
    assert "toi-" not in serialized
    assert "confirmed" not in serialized
    assert "eclipsing" not in serialized
