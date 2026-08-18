"""Tests for pi session resume (omp/pi launcher selection)."""

from pathlib import Path

from claude_code_tools.session_utils import resume_pi_session


class TestResumePiSession:
    """resume_pi_session builds the correct launcher command in shell mode."""

    def test_omp_home_uses_omp_launcher(self, capsys):
        """A ~/.omp session resumes with `omp -r <id>`."""
        session = Path.home() / ".omp/agent/sessions/-proj/2026_uuid.jsonl"
        resume_pi_session("abc123", "/tmp/other", session, shell_mode=True)
        out = capsys.readouterr().out
        assert "cd /tmp/other" in out
        assert "omp -r abc123" in out
        assert "pi -r" not in out

    def test_pi_home_uses_pi_launcher(self, capsys):
        """A ~/.pi session resumes with `pi -r <id>`."""
        session = Path.home() / ".pi/agent/sessions/-proj/2026_uuid.jsonl"
        resume_pi_session("abc123", "/tmp/other", session, shell_mode=True)
        out = capsys.readouterr().out
        assert "pi -r abc123" in out
        # Must not fall back to the omp launcher for a ~/.pi session.
        assert "omp -r" not in out

    def test_same_cwd_omits_cd(self, capsys, tmp_path, monkeypatch):
        """No `cd` is emitted when already in the session's directory."""
        monkeypatch.chdir(tmp_path)
        session = Path.home() / ".omp/agent/sessions/-proj/2026_uuid.jsonl"
        resume_pi_session("xyz", str(tmp_path), session, shell_mode=True)
        out = capsys.readouterr().out
        assert "cd " not in out
        assert "omp -r xyz" in out

    def test_session_id_is_shell_quoted(self, capsys):
        """Session ids with shell metacharacters are quoted."""
        session = Path.home() / ".omp/agent/sessions/-proj/2026_uuid.jsonl"
        resume_pi_session("a b;rm", "/tmp/other", session, shell_mode=True)
        out = capsys.readouterr().out
        assert "omp -r 'a b;rm'" in out
