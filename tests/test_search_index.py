"""Tests for Tantivy search index."""

import json
from pathlib import Path

import pytest
import yaml


@pytest.fixture
def sample_exports(tmp_path: Path) -> Path:
    """Create sample exported sessions with YAML front matter."""
    exports_dir = tmp_path / "exports"
    claude_dir = exports_dir / "claude"
    claude_dir.mkdir(parents=True)

    # Create 3 sample exports
    sessions = [
        {
            "session_id": "session-abc123",
            "agent": "claude",
            "project": "my-project",
            "branch": "main",
            "cwd": "/Users/test/my-project",
            "lines": 50,
            "modified": "2025-11-28T10:00:00",
            "content": "User asked about Python decorators and how to use them.",
        },
        {
            "session_id": "session-def456",
            "agent": "claude",
            "project": "my-project",
            "branch": "feature",
            "cwd": "/Users/test/my-project",
            "lines": 100,
            "modified": "2025-11-29T08:00:00",
            "content": "Discussion about TypeScript interfaces and type safety.",
        },
        {
            "session_id": "session-ghi789",
            "agent": "claude",
            "project": "other-project",
            "branch": "main",
            "cwd": "/Users/test/other-project",
            "lines": 30,
            "modified": "2025-11-27T12:00:00",
            "content": "Help with Rust borrow checker errors and lifetime issues.",
        },
    ]

    for session in sessions:
        content = session.pop("content")
        export_file = claude_dir / f"{session['session_id']}.txt"

        yaml_str = yaml.dump(session, default_flow_style=False)
        with open(export_file, "w") as f:
            f.write(f"---\n{yaml_str}---\n\n")
            f.write(f"> {content}\n")

    return exports_dir


@pytest.fixture
def index_path(tmp_path: Path) -> Path:
    """Return path for test index."""
    return tmp_path / "test-index"


class TestSearchIndex:
    """Tests for SessionIndex class."""

    def test_build_index_creates_index_directory(
        self, sample_exports: Path, index_path: Path
    ):
        """Should create index at specified path."""
        from claude_code_tools.search_index import SessionIndex

        index = SessionIndex(index_path)
        index.build_from_exports(sample_exports)

        assert index_path.exists()
        # Tantivy creates meta.json file
        assert (index_path / "meta.json").exists()

    def test_build_index_indexes_documents(
        self, sample_exports: Path, index_path: Path
    ):
        """Should index all exported sessions."""
        from claude_code_tools.search_index import SessionIndex

        index = SessionIndex(index_path)
        stats = index.build_from_exports(sample_exports)

        assert stats["indexed"] == 3

    def test_search_returns_matching_sessions(
        self, sample_exports: Path, index_path: Path
    ):
        """Search query should return sessions with matching content."""
        from claude_code_tools.search_index import SessionIndex

        index = SessionIndex(index_path)
        index.build_from_exports(sample_exports)

        results = index.search("Python decorators")
        assert len(results) >= 1
        # First result should be the Python session
        assert "abc123" in results[0]["session_id"]

    def test_search_returns_snippets(self, sample_exports: Path, index_path: Path):
        """Results should include snippet with match context."""
        from claude_code_tools.search_index import SessionIndex

        index = SessionIndex(index_path)
        index.build_from_exports(sample_exports)

        results = index.search("TypeScript")
        assert len(results) >= 1
        assert "snippet" in results[0]
        assert "TypeScript" in results[0]["snippet"]

    def test_search_returns_metadata(self, sample_exports: Path, index_path: Path):
        """Results should include session metadata."""
        from claude_code_tools.search_index import SessionIndex

        index = SessionIndex(index_path)
        index.build_from_exports(sample_exports)

        results = index.search("Rust")
        assert len(results) >= 1

        result = results[0]
        assert "session_id" in result
        assert "agent" in result
        assert "project" in result
        assert "modified" in result

    def test_search_empty_query_returns_recent(
        self, sample_exports: Path, index_path: Path
    ):
        """Empty query should return recent sessions."""
        from claude_code_tools.search_index import SessionIndex

        index = SessionIndex(index_path)
        index.build_from_exports(sample_exports)

        results = index.get_recent(limit=10)
        assert len(results) == 3

        # Should be sorted by recency (most recent first)
        # session-def456 is most recent (2025-11-29)
        assert "def456" in results[0]["session_id"]

    def test_search_respects_limit(self, sample_exports: Path, index_path: Path):
        """Should respect result limit."""
        from claude_code_tools.search_index import SessionIndex

        index = SessionIndex(index_path)
        index.build_from_exports(sample_exports)

        # Search for something that matches multiple sessions
        results = index.search("help", limit=1)
        assert len(results) == 1

    def test_search_filters_by_project(self, sample_exports: Path, index_path: Path):
        """Should support filtering by project/cwd."""
        from claude_code_tools.search_index import SessionIndex

        index = SessionIndex(index_path)
        index.build_from_exports(sample_exports)

        # Search with project filter
        results = index.search("", project="other-project")

        # Should only return sessions from other-project
        assert len(results) >= 1
        assert all("other-project" in r["project"] for r in results)

    def test_incremental_index_updates(self, sample_exports: Path, index_path: Path):
        """Incremental indexing should only add new/modified files."""
        from claude_code_tools.search_index import SessionIndex

        index = SessionIndex(index_path)

        # First build
        stats1 = index.build_from_exports(sample_exports)
        assert stats1["indexed"] == 3

        # Second build (no changes)
        stats2 = index.build_from_exports(sample_exports)
        assert stats2["indexed"] == 0
        assert stats2["skipped"] == 3

    def test_index_persists_across_instances(
        self, sample_exports: Path, index_path: Path
    ):
        """Index should persist and be usable by new instance."""
        from claude_code_tools.search_index import SessionIndex

        # Build index
        index1 = SessionIndex(index_path)
        index1.build_from_exports(sample_exports)

        # Create new instance and search
        index2 = SessionIndex(index_path)
        results = index2.search("Python")

        assert len(results) >= 1

    def test_search_result_has_score(self, sample_exports: Path, index_path: Path):
        """Search results should include relevance score."""
        from claude_code_tools.search_index import SessionIndex

        index = SessionIndex(index_path)
        index.build_from_exports(sample_exports)

        results = index.search("decorators")
        assert len(results) >= 1
        assert "score" in results[0]
        assert isinstance(results[0]["score"], float)


@pytest.fixture
def sample_jsonl_sessions(tmp_path: Path) -> list[Path]:
    """Create sample JSONL session files (raw Claude Code format)."""
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(parents=True)

    # Session 1: Simple user/assistant exchange
    session1 = sessions_dir / "session-jsonl-001.jsonl"
    session1_lines = [
        {
            "sessionId": "jsonl-session-001",
            "cwd": "/Users/test/my-project",
            "gitBranch": "main",
            "isSidechain": False,
            "type": "user",
            "message": {"role": "user", "content": "How do I use Python decorators?"},
            "timestamp": "2025-11-28T10:00:00Z",
        },
        {
            "sessionId": "jsonl-session-001",
            "cwd": "/Users/test/my-project",
            "gitBranch": "main",
            "isSidechain": False,
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Python decorators are functions that modify other functions."}
                ],
            },
            "timestamp": "2025-11-28T10:01:00Z",
        },
    ]
    with open(session1, "w") as f:
        for line in session1_lines:
            f.write(json.dumps(line) + "\n")

    # Session 2: With tool use
    session2 = sessions_dir / "session-jsonl-002.jsonl"
    session2_lines = [
        {
            "sessionId": "jsonl-session-002",
            "cwd": "/Users/test/rust-project",
            "gitBranch": "feature-branch",
            "isSidechain": True,
            "type": "user",
            "message": {"role": "user", "content": "Help me fix this Rust borrow checker error"},
            "timestamp": "2025-11-29T08:00:00Z",
        },
        {
            "sessionId": "jsonl-session-002",
            "cwd": "/Users/test/rust-project",
            "gitBranch": "feature-branch",
            "isSidechain": True,
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Let me read the file first."},
                    {"type": "tool_use", "name": "Read", "input": {"path": "src/main.rs"}},
                ],
            },
            "timestamp": "2025-11-29T08:01:00Z",
        },
    ]
    with open(session2, "w") as f:
        for line in session2_lines:
            f.write(json.dumps(line) + "\n")

    return [session1, session2]


class TestIndexFromJsonl:
    """Tests for direct JSONL indexing (Recall model approach)."""

    def test_parse_jsonl_session_extracts_metadata(
        self, sample_jsonl_sessions: list[Path], index_path: Path
    ):
        """Should extract metadata from JSONL session file."""
        from claude_code_tools.search_index import SessionIndex

        index = SessionIndex(index_path)
        parsed = index._parse_jsonl_session(sample_jsonl_sessions[0])

        assert parsed is not None
        # session_id is extracted from filename, not from internal sessionId field
        assert parsed["metadata"]["session_id"] == "session-jsonl-001"
        assert parsed["metadata"]["cwd"] == "/Users/test/my-project"
        assert parsed["metadata"]["branch"] == "main"
        assert parsed["metadata"]["project"] == "my-project"
        assert parsed["metadata"]["is_sidechain"] is False

    def test_parse_jsonl_session_extracts_timestamps(
        self, sample_jsonl_sessions: list[Path], index_path: Path
    ):
        """Should extract first and last timestamps."""
        from claude_code_tools.search_index import SessionIndex

        index = SessionIndex(index_path)
        parsed = index._parse_jsonl_session(sample_jsonl_sessions[0])

        assert parsed is not None
        assert parsed["metadata"]["created"] == "2025-11-28T10:00:00Z"
        assert parsed["metadata"]["modified"] == "2025-11-28T10:01:00Z"

    def test_parse_jsonl_session_extracts_content(
        self, sample_jsonl_sessions: list[Path], index_path: Path
    ):
        """Should concatenate message content for indexing."""
        from claude_code_tools.search_index import SessionIndex

        index = SessionIndex(index_path)
        parsed = index._parse_jsonl_session(sample_jsonl_sessions[0])

        assert parsed is not None
        assert "Python decorators" in parsed["content"]
        assert "functions that modify" in parsed["content"]

    def test_parse_jsonl_session_includes_tool_names(
        self, sample_jsonl_sessions: list[Path], index_path: Path
    ):
        """Should include tool names in content for searchability."""
        from claude_code_tools.search_index import SessionIndex

        index = SessionIndex(index_path)
        parsed = index._parse_jsonl_session(sample_jsonl_sessions[1])

        assert parsed is not None
        assert "[Tool: Read]" in parsed["content"]

    def test_parse_jsonl_session_tracks_first_last_msg(
        self, sample_jsonl_sessions: list[Path], index_path: Path
    ):
        """Should track first and last message for preview."""
        from claude_code_tools.search_index import SessionIndex

        index = SessionIndex(index_path)
        parsed = index._parse_jsonl_session(sample_jsonl_sessions[0])

        assert parsed is not None
        assert parsed["first_msg"]["role"] == "user"
        assert "decorators" in parsed["first_msg"]["content"]
        assert parsed["last_msg"]["role"] == "assistant"

    def test_index_from_jsonl_indexes_sessions(
        self, sample_jsonl_sessions: list[Path], index_path: Path
    ):
        """Should index JSONL sessions directly."""
        from claude_code_tools.search_index import SessionIndex

        index = SessionIndex(index_path)
        stats = index.index_from_jsonl(sample_jsonl_sessions)

        assert stats["indexed"] == 2
        assert stats["failed"] == 0

    def test_index_from_jsonl_is_searchable(
        self, sample_jsonl_sessions: list[Path], index_path: Path
    ):
        """Indexed JSONL sessions should be searchable."""
        from claude_code_tools.search_index import SessionIndex

        index = SessionIndex(index_path)
        index.index_from_jsonl(sample_jsonl_sessions)

        # Search for Python content
        results = index.search("Python decorators")
        assert len(results) >= 1
        assert "session-jsonl-001" in results[0]["session_id"]

        # Search for Rust content
        results = index.search("Rust borrow checker")
        assert len(results) >= 1
        assert "session-jsonl-002" in results[0]["session_id"]

    def test_index_from_jsonl_incremental(
        self, sample_jsonl_sessions: list[Path], index_path: Path
    ):
        """Incremental indexing should skip unchanged files."""
        from claude_code_tools.search_index import SessionIndex

        index = SessionIndex(index_path)

        # First index
        stats1 = index.index_from_jsonl(sample_jsonl_sessions)
        assert stats1["indexed"] == 2

        # Second index (no changes)
        stats2 = index.index_from_jsonl(sample_jsonl_sessions)
        assert stats2["indexed"] == 0
        assert stats2["skipped"] == 2

    def test_index_from_jsonl_detects_sidechain(
        self, sample_jsonl_sessions: list[Path], index_path: Path
    ):
        """Should correctly detect sidechain sessions."""
        from claude_code_tools.search_index import SessionIndex

        index = SessionIndex(index_path)
        index.index_from_jsonl(sample_jsonl_sessions)

        # Search for sidechain session
        results = index.search("Rust")
        assert len(results) >= 1
        # The is_sidechain field should be "true" for session 2
        rust_result = next(r for r in results if "002" in r["session_id"])
        # Note: is_sidechain is stored as string "true"/"false"


@pytest.fixture
def sample_pi_sessions(tmp_path: Path) -> list[Path]:
    """Create sample pi JSONL sessions mirroring the on-disk layout.

    Pi stores the main session at
    ``<home>/.omp/agent/sessions/<project>/<file>.jsonl`` and sub-agents one
    level deeper at ``.../<project>/<run>/<name>.jsonl``.
    """
    project_dir = (
        tmp_path / ".omp" / "agent" / "sessions" / "-Users-test-proj"
    )
    run_dir = project_dir / "2026-08-17T08-53-32-716Z_main-uuid"
    run_dir.mkdir(parents=True)

    # Main session
    main = project_dir / "2026-08-17T08-53-32-716Z_019fdb0c-30e0-7000-b6dc-8135a4374fc5.jsonl"
    main_lines = [
        {"type": "title", "v": 1, "title": "Explore pi sessions",
         "source": "auto", "updatedAt": "2026-08-17T08:54:44.462Z"},
        {"type": "session", "version": 3,
         "id": "019fdb0c-30e0-7000-b6dc-8135a4374fc5",
         "timestamp": "2026-08-17T08:53:32.716Z",
         "cwd": "/Users/test/proj", "title": "Explore pi sessions"},
        {"type": "message", "timestamp": "2026-08-17T08:54:43.748Z",
         "message": {"role": "user", "content": [
             {"type": "text", "text": "How do I use Python decorators?"}]}},
        {"type": "message", "timestamp": "2026-08-17T08:54:48.121Z",
         "message": {"role": "assistant", "content": [
             {"type": "thinking", "thinking": "Let me look this up."},
             {"type": "text", "text": "Decorators wrap functions."},
             {"type": "toolCall", "id": "t1", "name": "read",
              "arguments": {"path": "decorators.py", "i": "reading file"}}]}},
        {"type": "message", "timestamp": "2026-08-17T08:54:48.136Z",
         "message": {"role": "toolResult", "content": [
             {"type": "text", "text": "file contents here"}]}},
    ]
    with open(main, "w") as f:
        for line in main_lines:
            f.write(json.dumps(line) + "\n")

    # Sub-agent session (nested one level deeper)
    sub = run_dir / "__advisor.jsonl"
    sub_lines = [
        {"type": "session", "version": 3,
         "id": "019fdb3d-31d3-7000-8e91-c0a7526dbcbf",
         "timestamp": "2026-08-17T09:00:48.595Z",
         "cwd": "/Users/test/proj"},
        {"type": "message", "timestamp": "2026-08-17T09:00:49.000Z",
         "message": {"role": "user", "content": [
             {"type": "text", "text": "Advise on Rust borrow checker"}]}},
        {"type": "message", "timestamp": "2026-08-17T09:00:50.000Z",
         "message": {"role": "assistant", "content": [
             {"type": "text", "text": "Use references carefully."}]}},
    ]
    with open(sub, "w") as f:
        for line in sub_lines:
            f.write(json.dumps(line) + "\n")

    return [main, sub]


class TestPiSessions:
    """Tests for indexing pi agent sessions."""

    def test_detect_agent_from_path(self, sample_pi_sessions: list[Path]):
        """Pi session paths should be detected as agent 'pi'."""
        from claude_code_tools.session_utils import detect_agent_from_path

        for f in sample_pi_sessions:
            assert detect_agent_from_path(f) == "pi"

    def test_is_valid_session(self, sample_pi_sessions: list[Path]):
        """Pi session/message records should validate."""
        from claude_code_tools.session_utils import is_valid_session

        for f in sample_pi_sessions:
            assert is_valid_session(f) is True

    def test_metadata_uses_session_record_id(
        self, sample_pi_sessions: list[Path]
    ):
        """Session id comes from the 'session' record, not the filename."""
        from claude_code_tools.export_session import extract_session_metadata

        main_meta = extract_session_metadata(sample_pi_sessions[0], "pi")
        assert main_meta["session_id"] == "019fdb0c-30e0-7000-b6dc-8135a4374fc5"
        assert main_meta["cwd"] == "/Users/test/proj"
        assert main_meta["project"] == "proj"
        assert main_meta["is_sidechain"] is False
        assert main_meta["created"] == "2026-08-17T08:53:32.716Z"

        sub_meta = extract_session_metadata(sample_pi_sessions[1], "pi")
        # Sub-agent file name is not a UUID; id must come from the record
        assert sub_meta["session_id"] == "019fdb3d-31d3-7000-8e91-c0a7526dbcbf"

    def test_metadata_detects_subagent_sidechain(
        self, sample_pi_sessions: list[Path]
    ):
        """Nested pi sub-agent sessions are flagged as sidechains."""
        from claude_code_tools.export_session import extract_session_metadata

        sub_meta = extract_session_metadata(sample_pi_sessions[1], "pi")
        assert sub_meta["is_sidechain"] is True

    def test_parse_extracts_content_and_tools(
        self, sample_pi_sessions: list[Path], index_path: Path
    ):
        """Content extraction should capture text, thinking and tool names."""
        from claude_code_tools.search_index import SessionIndex

        index = SessionIndex(index_path)
        parsed = index._parse_jsonl_session(sample_pi_sessions[0])

        assert parsed is not None
        assert parsed["metadata"]["agent"] == "pi"
        assert parsed["metadata"]["session_id"] == (
            "019fdb0c-30e0-7000-b6dc-8135a4374fc5"
        )
        assert "Python decorators" in parsed["content"]
        assert "Decorators wrap functions" in parsed["content"]
        assert "Let me look this up" in parsed["content"]
        assert "[Tool: read]" in parsed["content"]
        assert "decorators.py" in parsed["content"]
        assert parsed["first_msg"]["role"] == "user"
        assert "decorators" in parsed["first_msg"]["content"]

    def test_index_and_search(
        self, sample_pi_sessions: list[Path], index_path: Path
    ):
        """Pi sessions should be indexed and searchable."""
        from claude_code_tools.search_index import SessionIndex

        index = SessionIndex(index_path)
        stats = index.index_from_jsonl(sample_pi_sessions)
        assert stats["indexed"] == 2
        assert stats["failed"] == 0

        results = index.search("Python decorators")
        assert len(results) >= 1
        assert results[0]["agent"] == "pi"
        assert results[0]["session_id"] == (
            "019fdb0c-30e0-7000-b6dc-8135a4374fc5"
        )

    def test_auto_index_discovers_pi_sessions(
        self, sample_pi_sessions: list[Path], tmp_path: Path, index_path: Path
    ):
        """auto_index should discover pi sessions under a pi home."""
        from claude_code_tools.search_index import auto_index

        stats = auto_index(
            index_path=index_path,
            claude_home=tmp_path / "nonexistent-claude",
            codex_home=tmp_path / "nonexistent-codex",
            pi_homes=[tmp_path / ".omp"],
            silent=True,
        )
        # Main + sub-agent both discovered and indexed
        assert stats["pi_files"] == 2
        assert stats["indexed"] == 2
