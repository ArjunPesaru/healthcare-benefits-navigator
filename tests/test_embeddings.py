"""
Unit tests for rag/embeddings.py.

Covers: JSONL chunk loading, edge cases, and error handling for
missing FAISS index/model files.
"""
import json
import pytest

from rag.embeddings import load_chunks_from_jsonl, load_index


# ── TestLoadChunksFromJsonl ───────────────────────────────────────────────────

class TestLoadChunksFromJsonl:
    """Tests for parsing JSONL chunk files."""

    def test_loads_valid_jsonl_file(self, tmp_path):
        chunks = [
            {"chunk_id": "c1", "chunk_type": "plan",         "text": "hello world"},
            {"chunk_id": "c2", "chunk_type": "connectorcare", "text": "subsidy info"},
        ]
        path = tmp_path / "chunks.jsonl"
        path.write_text("\n".join(json.dumps(c) for c in chunks) + "\n")

        loaded = load_chunks_from_jsonl(str(path))
        assert len(loaded) == 2
        assert loaded[0]["chunk_id"] == "c1"
        assert loaded[1]["chunk_type"] == "connectorcare"

    def test_returns_list_type(self, tmp_path):
        path = tmp_path / "chunks.jsonl"
        path.write_text('{"chunk_id": "c1", "text": "data"}\n')
        result = load_chunks_from_jsonl(str(path))
        assert isinstance(result, list)

    def test_skips_blank_lines(self, tmp_path):
        path = tmp_path / "chunks.jsonl"
        path.write_text(
            '{"chunk_id": "c1", "text": "first"}\n'
            "\n"
            '{"chunk_id": "c2", "text": "second"}\n'
            "\n"
        )
        loaded = load_chunks_from_jsonl(str(path))
        assert len(loaded) == 2

    def test_empty_file_returns_empty_list(self, tmp_path):
        path = tmp_path / "empty.jsonl"
        path.write_text("")
        assert load_chunks_from_jsonl(str(path)) == []

    def test_preserves_chunk_metadata(self, tmp_path):
        chunk = {
            "chunk_id": "plan_001",
            "chunk_type": "plan",
            "text": "PLAN: Test",
            "metadata": {"metal_tier": "Silver", "age_premiums": {21: 430.0}},
        }
        path = tmp_path / "chunks.jsonl"
        path.write_text(json.dumps(chunk) + "\n")
        loaded = load_chunks_from_jsonl(str(path))
        assert loaded[0]["metadata"]["metal_tier"] == "Silver"
        # JSON serialises integer keys as strings — {21: 430.0} → {"21": 430.0}
        assert loaded[0]["metadata"]["age_premiums"]["21"] == 430.0

    def test_single_line_file(self, tmp_path):
        path = tmp_path / "chunks.jsonl"
        path.write_text('{"chunk_id": "c1", "text": "only one"}')
        loaded = load_chunks_from_jsonl(str(path))
        assert len(loaded) == 1


# ── TestLoadIndex ─────────────────────────────────────────────────────────────

class TestLoadIndex:
    def test_raises_file_not_found_when_index_missing(self, tmp_path, monkeypatch):
        """load_index should raise FileNotFoundError pointing user to setup.py."""
        import rag.embeddings as emb_mod
        monkeypatch.setattr(emb_mod, "FAISS_INDEX_PATH", str(tmp_path / "no_index.faiss"))
        with pytest.raises(FileNotFoundError, match="setup.py"):
            load_index()
