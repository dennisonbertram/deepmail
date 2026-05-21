"""Tests for the Deep Email unified CLI and related changes.

Covers:
  - deep-email init writes .mcp.json with correct content
  - deep-email init merges into existing .mcp.json
  - deep-email init skips if already configured
  - Embedded OAuth defaults load when GOOGLE_CLIENT_ID is absent
  - MCP server name is 'deepmail'
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

POC_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(POC_ROOT / "src"))

from pi_email.unified_cli import cli


# ====================================================================
# deepmail init tests
# ====================================================================


class TestDeepmailInit:

    def test_init_writes_mcp_json(self, tmp_path):
        """deep-email init creates .mcp.json with correct content."""
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(cli, ["init"])
            assert result.exit_code == 0
            assert "Wrote" in result.output

            mcp_json = Path.cwd() / ".mcp.json"
            assert mcp_json.exists()

            config = json.loads(mcp_json.read_text())
            assert "mcpServers" in config
            assert "deepmail" in config["mcpServers"]
            server = config["mcpServers"]["deepmail"]
            assert server["type"] == "stdio"
            assert server["command"] == "uvx"
            assert server["args"] == ["deep-email"]

    def test_init_merges_existing(self, tmp_path):
        """deep-email init merges into existing .mcp.json with other servers."""
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            # Pre-create .mcp.json with another server
            existing = {
                "mcpServers": {
                    "other-server": {
                        "type": "stdio",
                        "command": "other-cmd",
                        "args": [],
                    }
                }
            }
            Path(".mcp.json").write_text(json.dumps(existing))

            result = runner.invoke(cli, ["init"])
            assert result.exit_code == 0
            assert "Wrote" in result.output

            config = json.loads(Path(".mcp.json").read_text())
            # Both servers should be present
            assert "other-server" in config["mcpServers"]
            assert "deepmail" in config["mcpServers"]

    def test_init_skips_if_already_configured(self, tmp_path):
        """deep-email init does not duplicate if deepmail is already in .mcp.json."""
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            existing = {
                "mcpServers": {
                    "deepmail": {
                        "type": "stdio",
                        "command": "uvx",
                        "args": ["deepmail"],
                    }
                }
            }
            Path(".mcp.json").write_text(json.dumps(existing))

            result = runner.invoke(cli, ["init"])
            assert result.exit_code == 0
            assert "already configured" in result.output

            # Content should be unchanged
            config = json.loads(Path(".mcp.json").read_text())
            assert len(config["mcpServers"]) == 1


# ====================================================================
# Embedded defaults test (no GOOGLE_CLIENT_ID required)
# ====================================================================


class TestEmbeddedDefaults:

    def test_no_env_var_loads_embedded_defaults(self):
        """OAuthConfig.from_env() returns embedded defaults when GOOGLE_CLIENT_ID is absent."""
        from pi_email.oauth import OAuthConfig, _DEFAULT_CLIENT_ID

        # Clear the env var and prevent dotenv from reloading it
        with patch.dict(os.environ, {}, clear=True), \
             patch("pi_email.oauth._find_dotenv", None), \
             patch("pi_email.oauth._load_dotenv", None):
            cfg = OAuthConfig.from_env()
            assert cfg.client_id == _DEFAULT_CLIENT_ID


# ====================================================================
# MCP server name test
# ====================================================================


class TestMcpServerName:

    def test_mcp_server_name_is_deepmail(self):
        """FastMCP server instance is named 'deepmail'."""
        from pi_email.mcp_server import mcp

        assert mcp.name == "deepmail"


# ====================================================================
# CLI --help tests
# ====================================================================


class TestAccountsCommand:

    def test_accounts_no_accounts(self):
        """deep-email accounts with no accounts -> 'No authenticated accounts'."""
        runner = CliRunner()
        with patch("pi_email.token_store.TokenStore.list_accounts", return_value=[]):
            result = runner.invoke(cli, ["accounts"])
            assert result.exit_code == 1
            assert "No authenticated accounts" in result.output

    def test_accounts_lists_accounts(self):
        """deep-email accounts with two accounts -> lists them."""
        from unittest.mock import MagicMock

        runner = CliRunner()
        mock_creds = MagicMock()
        mock_creds.expired = False

        with (
            patch("pi_email.token_store.TokenStore._maybe_migrate"),
            patch("pi_email.token_store.TokenStore.list_accounts", return_value=[
                "work@example.com",
                "personal@gmail.com",
            ]),
            patch("pi_email.token_store.TokenStore.load", return_value=mock_creds),
            patch("pi_email.oauth.refresh_if_needed"),
            patch("googleapiclient.discovery.build") as mock_build,
        ):
            # Mock the Gmail API profile call.
            mock_service = MagicMock()
            mock_build.return_value = mock_service
            mock_service.users.return_value.getProfile.return_value.execute.return_value = {
                "messagesTotal": 5000,
            }

            result = runner.invoke(cli, ["accounts"])
            assert result.exit_code == 0
            assert "work@example.com" in result.output
            assert "personal@gmail.com" in result.output
            assert "5,000 messages" in result.output

    def test_auth_remove(self):
        """deep-email auth --remove user@example.com removes the account."""
        runner = CliRunner()
        with (
            patch("pi_email.token_store.TokenStore._maybe_migrate"),
            patch("pi_email.token_store.TokenStore.remove", return_value=True) as mock_remove,
        ):
            result = runner.invoke(cli, ["auth", "--remove", "user@example.com"])
            assert result.exit_code == 0
            assert "Removed" in result.output
            mock_remove.assert_called_once_with("user@example.com")


class TestCliHelp:

    def test_deep_email_help(self):
        """deep-email --help shows all subcommands."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "auth" in result.output
        assert "whoami" in result.output
        assert "query" in result.output
        assert "run" in result.output
        assert "init" in result.output
        assert "setup" in result.output
        assert "serve" in result.output
        assert "accounts" in result.output

    def test_deep_email_init_help(self):
        """deep-email init --help works."""
        runner = CliRunner()
        result = runner.invoke(cli, ["init", "--help"])
        assert result.exit_code == 0
        assert "Configure Deep Email" in result.output

    def test_deep_email_setup_help(self):
        """deep-email setup --help works."""
        runner = CliRunner()
        result = runner.invoke(cli, ["setup", "--help"])
        assert result.exit_code == 0
        assert "wizard" in result.output.lower()

    def test_deep_email_auth_help(self):
        """deep-email auth --help works."""
        runner = CliRunner()
        result = runner.invoke(cli, ["auth", "--help"])
        assert result.exit_code == 0
        assert "OAuth" in result.output


# ====================================================================
# Token expiry error message test
# ====================================================================


class TestTokenExpiryMessage:

    def test_format_gmail_error_invalid_grant(self):
        """_format_gmail_error detects invalid_grant and returns friendly message."""
        from pi_email.mcp_server import _format_gmail_error

        exc = RuntimeError("invalid_grant: Token has been revoked")
        msg = _format_gmail_error(exc, "search_emails")
        assert "deep-email auth" in msg
        assert "7 days" in msg

    def test_format_gmail_error_401(self):
        """_format_gmail_error detects 401 errors."""
        from pi_email.mcp_server import _format_gmail_error

        exc = RuntimeError("HttpError 401: Request had invalid authentication credentials")
        msg = _format_gmail_error(exc, "read_email")
        assert "deep-email auth" in msg

    def test_format_gmail_error_generic(self):
        """_format_gmail_error returns generic message for unknown errors."""
        from pi_email.mcp_server import _format_gmail_error

        exc = RuntimeError("Connection refused")
        msg = _format_gmail_error(exc, "search_emails")
        assert "Connection refused" in msg
        assert "search_emails" in msg
