"""
Tests for OpenKeywords CLI
"""

import pytest
from click.testing import CliRunner
from unittest.mock import patch, MagicMock, AsyncMock
import os

from openkeywords.cli import main, check, generate


class TestCLICheck:
    """Tests for the check command"""

    def test_check_no_keys(self):
        """Test check with no API keys set"""
        runner = CliRunner()

        with patch.dict(os.environ, {}, clear=True):
            result = runner.invoke(check)

        assert result.exit_code == 0
        assert "GEMINI_API_KEY: Not set" in result.output

    def test_check_with_gemini_key(self):
        """Test check with Gemini key set"""
        runner = CliRunner()

        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key-12345"}, clear=True):
            result = runner.invoke(check)

        assert result.exit_code == 0
        assert "GEMINI_API_KEY: Set" in result.output

    def test_check_with_both_keys(self):
        """Test check with both keys set"""
        runner = CliRunner()

        env = {
            "GEMINI_API_KEY": "gemini-key-12345",
            "SERANKING_API_KEY": "seranking-key-12345",
        }
        with patch.dict(os.environ, env, clear=True):
            result = runner.invoke(check)

        assert result.exit_code == 0
        assert "GEMINI_API_KEY: Set" in result.output
        assert "SERANKING_API_KEY: Set" in result.output


class TestCLIGenerate:
    """Tests for the generate command"""

    def test_generate_no_api_key(self):
        """Test generate fails without API key"""
        runner = CliRunner()

        with patch.dict(os.environ, {}, clear=True):
            result = runner.invoke(generate, ["--company", "TestCorp"])

        assert result.exit_code == 1
        assert "GEMINI_API_KEY" in result.output

    def test_generate_help(self):
        """Test generate help"""
        runner = CliRunner()
        result = runner.invoke(generate, ["--help"])

        assert result.exit_code == 0
        assert "--company" in result.output
        assert "--industry" in result.output
        assert "--count" in result.output

    def test_generate_requires_company(self):
        """Test generate requires --company"""
        runner = CliRunner()

        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}, clear=True):
            result = runner.invoke(generate, [])

        assert result.exit_code != 0
        assert "Missing option" in result.output or "required" in result.output.lower()


class TestCLIMain:
    """Tests for main CLI group"""

    def test_main_help(self):
        """Test main help"""
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])

        assert result.exit_code == 0
        assert "OpenKeywords" in result.output
        assert "check" in result.output
        assert "generate" in result.output

    def test_main_version(self):
        """Test version flag"""
        runner = CliRunner()
        result = runner.invoke(main, ["--version"])

        assert result.exit_code == 0
        assert "0.2.0" in result.output
