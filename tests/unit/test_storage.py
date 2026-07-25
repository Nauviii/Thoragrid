"""Unit tests for deterministic storage path helpers (pure functions, no Supabase call needed).

create_signed_url() itself requires a live Supabase project and is covered by the
integration suite (tests/integration/test_conversation_memory.py), consistent with this
project's live-service testing approach.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from core.storage.supabase_storage import xray_path, heatmap_path


def test_xray_path_matches_upload_convention():
    """xray_path must match the f'{image_hash}.png' convention used in api/routes/image.py."""
    assert xray_path("abc123") == "abc123.png"


def test_heatmap_path_matches_upload_convention():
    """heatmap_path must match the f'{interaction_id}/{condition}.png' convention in api/routes/image.py."""
    assert heatmap_path("int-1", "Cardiomegaly") == "int-1/Cardiomegaly.png"


def test_paths_are_deterministic():
    """Same inputs must always yield the same path, since re-signing depends on recomputing it later."""
    assert xray_path("h") == xray_path("h")
    assert heatmap_path("i", "Mass") == heatmap_path("i", "Mass")